#!/usr/bin/env python3
"""
Compute MMD (Maximum Mean Discrepancy) and JS divergence between different
performers' hand/lip features under four different processing methods.

Methods:
  1. 微调 CLIP (Fine-tuned CLIP): Conformer fine-tuned lip features (768-dim)
  2. 原生冻结 CLIP (Frozen CLIP): Raw CLIP ViT-L/14 features (1024-dim)
  3. 冻结 + CSSP: Frozen CLIP → projection + SignerNorm (768-dim)
  4. 冻结 + CSSP + VP-Align: Frozen CLIP → CSSP → fusion_proj (2048-dim)

MMD: RBF kernel, measures distance between feature centers in high-dim space.
     Lower MMD → features from different performers are more aligned.
JS divergence: Random-projection + histogram-based, measures probability
     distribution similarity. Lower JS → distributions are more similar.

Usage:
    python scripts/compute_mmd_js.py [--max_frames 5000] [--device cuda:0]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import argparse
import os
import sys
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Add fna_sra to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path('/home/uic/fengling/mccsd')
SIGNERS = ['HS', 'LF', 'WT', 'XP', 'YX', 'YZ']

# Feature paths
FROZEN_HAND_ROOT  = BASE_DIR / 'mccsd_datasets/Hand_Features/clip-vit-large-patch14_hand_feat_mccsd'
FROZEN_LIP_ROOT   = BASE_DIR / 'mccsd_datasets/Lip_Features/clip-vit-large-patch14_lip_feat_mccsd'
CONFORMER_FT_ROOT = BASE_DIR / 'fna_sra/vit_finetune_output/conformer_lip_feat_mccsd'

# Checkpoint (trained with cross_modal_align=true, use_feature_norm=true)
CKPT_PATH = BASE_DIR / 'fna_sra/logs/2026-06-01T10-39-25_mccsd_6H_cv_HS/checkpoints/epoch=00047-step=0013536-cer=5.38-wer=11.07.ckpt'

# Output
OUTPUT_DIR = BASE_DIR / 'fna_sra/scripts/mmd_js_results'

NUM_JS_PROJECTIONS = 20   # number of random projections for JS estimation
NUM_JS_BINS        = 100  # histogram bins for JS

# ---------------------------------------------------------------------------
# 1. Feature loading
# ---------------------------------------------------------------------------
def load_features_by_signer(feat_root, signers, max_frames=10000):
    """
    Load all per-frame features for each signer from .npy files.
    Returns:
        feats_dict: signer -> numpy array (N_frames, D) — pooled frames
        seq_dict:   signer -> list of numpy arrays, each (T_i, D) — per-video
    """
    feat_by_signer = defaultdict(list)
    seq_by_signer = defaultdict(list)
    for signer in signers:
        signer_dir = Path(feat_root) / signer
        if not signer_dir.exists():
            print(f'  [WARN] Directory not found: {signer_dir}')
            continue
        for fpath in sorted(signer_dir.iterdir()):
            if fpath.suffix != '.npy':
                continue
            try:
                feats = np.load(fpath)  # (T, D)
                if feats.ndim == 2 and feats.shape[0] > 0:
                    feat_by_signer[signer].append(feats.astype(np.float32))
                    seq_by_signer[signer].append(feats.astype(np.float32))
            except Exception as e:
                print(f'  [WARN] Failed to load {fpath}: {e}')

    # Concatenate and subsample for pooled features
    result_pooled = {}
    result_seq = {}
    for signer, feat_list in feat_by_signer.items():
        all_feats = np.concatenate(feat_list, axis=0)  # (total_frames, D)
        if len(all_feats) > max_frames:
            idx = np.random.RandomState(42).choice(len(all_feats), max_frames, replace=False)
            all_feats = all_feats[idx]
        result_pooled[signer] = all_feats
        result_seq[signer] = seq_by_signer[signer]
        print(f'    {signer}: {all_feats.shape[0]} frames ({len(seq_by_signer[signer])} videos), dim={all_feats.shape[1]}')
    return result_pooled, result_seq


# ---------------------------------------------------------------------------
# 2. Model-based feature extraction (CSSP / CSSP+VP-Align)
# ---------------------------------------------------------------------------
def load_model_components(device='cpu'):
    """
    Load model checkpoint and extract CSSP-related components.
    Returns: spatio_proj, spatiotemp_proj, signer_norm_spatial,
             signer_norm_spatiotem, fusion_proj
    """
    from fna_sra.t5_sra import FlanT5SLT, SignerNorm
    from fna_sra.mm_projector import build_vision_projector

    print(f'Loading checkpoint: {CKPT_PATH}')
    ckpt = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Extract relevant weights
    def get_weights(prefix):
        d = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                d[k[len(prefix)+1:]] = v
        return d

    # Build components
    spatio_proj = build_vision_projector('linear', 1024, 768)
    spatiotemp_proj = build_vision_projector('linear', 1024, 768)
    fusion_proj = build_vision_projector('mlp2x_gelu', 768, 2048)

    spatio_proj.load_state_dict(get_weights('spatio_proj'))
    spatiotemp_proj.load_state_dict(get_weights('spatiotemp_proj'))
    fusion_proj.load_state_dict(get_weights('fusion_proj'))

    signer_norm_spatial = SignerNorm(768)
    signer_norm_spatiotem = SignerNorm(768)
    signer_norm_spatial.load_state_dict(get_weights('signer_norm_spatial'))
    signer_norm_spatiotem.load_state_dict(get_weights('signer_norm_spatiotem'))

    # Move to device
    spatio_proj = spatio_proj.to(device).eval()
    spatiotemp_proj = spatiotemp_proj.to(device).eval()
    fusion_proj = fusion_proj.to(device).eval()
    signer_norm_spatial = signer_norm_spatial.to(device).eval()
    signer_norm_spatiotem = signer_norm_spatiotem.to(device).eval()

    return spatio_proj, spatiotemp_proj, signer_norm_spatial, signer_norm_spatiotem, fusion_proj


def extract_cssp_features(seq_by_signer, proj, signer_norm, max_frames=5000, device='cpu'):
    """
    Pass features through projection + SignerNorm.
    Processes each video as a sequence so SignerNorm operates correctly.
    seq_by_signer: dict signer -> list of numpy arrays, each (T_i, 1024)
    Returns: dict signer -> numpy array (N, 768) — pooled frames
    """
    result = {}
    for signer, sequences in seq_by_signer.items():
        all_out = []
        total_frames = 0
        for seq in sequences:
            if total_frames >= max_frames:
                break
            T = seq.shape[0]
            if T == 0:
                continue
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, 1024)
            out = proj(seq_tensor)                                          # (1, T, 768)
            mask = torch.ones(1, T, dtype=torch.bool, device=device)
            out = signer_norm(out, mask)                                   # (1, T, 768)
            out_np = out.squeeze(0).detach().cpu().numpy()                 # (T, 768)
            all_out.append(out_np)
            total_frames += T
        result[signer] = np.concatenate(all_out, axis=0).astype(np.float32)[:max_frames]
        print(f'    CSSP {signer}: {result[signer].shape}')
    return result


def extract_vpalign_features(seq_by_signer, proj, signer_norm, fusion_proj,
                              max_frames=5000, device='cpu'):
    """
    Pass features through CSSP + fusion_proj.
    Processes each video as a sequence.
    seq_by_signer: dict signer -> list of numpy arrays, each (T_i, 1024)
    Returns: dict signer -> numpy array (N, 2048) — pooled frames
    """
    result = {}
    for signer, sequences in seq_by_signer.items():
        all_out = []
        total_frames = 0
        for seq in sequences:
            if total_frames >= max_frames:
                break
            T = seq.shape[0]
            if T == 0:
                continue
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, 1024)
            out = proj(seq_tensor)                                          # (1, T, 768)
            mask = torch.ones(1, T, dtype=torch.bool, device=device)
            out = signer_norm(out, mask)                                   # (1, T, 768)
            out = fusion_proj(out)                                          # (1, T, 2048)
            out_np = out.squeeze(0).detach().cpu().numpy()                 # (T, 2048)
            all_out.append(out_np)
            total_frames += T
        result[signer] = np.concatenate(all_out, axis=0).astype(np.float32)[:max_frames]
        print(f'    VP-Align {signer}: {result[signer].shape}')
    return result


# ---------------------------------------------------------------------------
# 3. MMD computation (RBF kernel)
# ---------------------------------------------------------------------------
def compute_mmd(X, Y, sigma=None):
    """
    Compute squared MMD between two feature sets X (N, D) and Y (M, D).
    Uses RBF (Gaussian) kernel with median heuristic for sigma.
    Returns: (mmd2, sigma, mmd)
    """
    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))

    if sigma is None:
        # Median heuristic: use a subset if too large
        n_sample = min(2000, len(X), len(Y))
        idx_x = torch.randperm(len(X))[:n_sample]
        idx_y = torch.randperm(len(Y))[:n_sample]
        pooled = torch.cat([X[idx_x], Y[idx_y]], dim=0)
        dists = torch.cdist(pooled, pooled)
        sigma = dists.median().item() / 2.0
        sigma = max(sigma, 1e-6)

    sigma2 = 2.0 * sigma ** 2

    # For memory efficiency, process in chunks
    def kernel_XX_chunked(A):
        total = 0.0
        chunk = 1000
        for i in range(0, len(A), chunk):
            Ai = A[i:i+chunk]
            total += torch.exp(-torch.cdist(Ai, A) ** 2 / sigma2).sum().item()
        return total / (len(A) ** 2)

    def kernel_XY_chunked(A, B):
        total = 0.0
        chunk = 500
        for i in range(0, len(A), chunk):
            Ai = A[i:i+chunk]
            total += torch.exp(-torch.cdist(Ai, B) ** 2 / sigma2).sum().item()
        return total / (len(A) * len(B))

    k_xx = kernel_XX_chunked(X)
    k_yy = kernel_XX_chunked(Y)
    k_xy = kernel_XY_chunked(X, Y)

    mmd2 = k_xx + k_yy - 2.0 * k_xy
    mmd = np.sqrt(max(mmd2, 0.0))
    return mmd2, sigma, mmd


# ---------------------------------------------------------------------------
# 4. JS divergence (random projection + histogram)
# ---------------------------------------------------------------------------
def compute_js_divergence(X, Y, n_projections=NUM_JS_PROJECTIONS, n_bins=NUM_JS_BINS):
    """
    Compute JS divergence between two feature distributions.
    Uses random projections to 1D + histogram binning.
    Returns: average JS divergence over projections.
    """
    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))
    D = X.shape[1]

    js_values = []
    for _ in range(n_projections):
        # Random projection direction
        v = torch.randn(D)
        v = v / v.norm()

        x_proj = X @ v
        y_proj = Y @ v

        all_vals = torch.cat([x_proj, y_proj])
        min_val, max_val = all_vals.min().item(), all_vals.max().item()
        eps_range = (max_val - min_val) * 1e-4
        min_val -= eps_range
        max_val += eps_range

        x_hist = torch.histc(x_proj, bins=n_bins, min=min_val, max=max_val)
        y_hist = torch.histc(y_proj, bins=n_bins, min=min_val, max=max_val)

        # Laplace smoothing
        x_hist = x_hist + 1e-8
        y_hist = y_hist + 1e-8

        x_p = x_hist / x_hist.sum()
        y_p = y_hist / y_hist.sum()
        m = 0.5 * (x_p + y_p)

        # KL divergence (handle zeros)
        kl_xm = (x_p * torch.log(x_p / m)).sum().item()
        kl_ym = (y_p * torch.log(y_p / m)).sum().item()

        js = 0.5 * kl_xm + 0.5 * kl_ym
        js_values.append(js)

    return np.mean(js_values), np.std(js_values)


# ---------------------------------------------------------------------------
# 5. Pairwise computation & output
# ---------------------------------------------------------------------------
def compute_pairwise_matrix(feats_by_signer, signers, method_name, feat_type):
    """
    Compute pairwise MMD and JS matrices for all signer pairs.
    """
    active_signers = [s for s in signers if s in feats_by_signer]
    n = len(active_signers)

    mmd_matrix = np.zeros((n, n))
    js_matrix = np.zeros((n, n))
    sigma_values = np.zeros((n, n))

    pairs = list(combinations(range(n), 2))
    print(f'\n  Computing {len(pairs)} signer pairs for {method_name} ({feat_type})...')

    for idx, (i, j) in enumerate(pairs):
        s_i, s_j = active_signers[i], active_signers[j]
        X, Y = feats_by_signer[s_i], feats_by_signer[s_j]

        mmd2, sigma, mmd_val = compute_mmd(X, Y)
        js_val, _ = compute_js_divergence(X, Y)

        mmd_matrix[i, j] = mmd_matrix[j, i] = mmd_val
        js_matrix[i, j] = js_matrix[j, i] = js_val
        sigma_values[i, j] = sigma_values[j, i] = sigma

        if (idx + 1) % 5 == 0 or idx == len(pairs) - 1:
            print(f'    [{idx+1}/{len(pairs)}] {s_i} vs {s_j}: '
                  f'MMD={mmd_val:.6f}, JS={js_val:.6f}')

    return active_signers, mmd_matrix, js_matrix, sigma_values


def print_matrix(matrix, signers, title):
    """Pretty-print a matrix with signer labels."""
    print(f'\n  {title}:')
    header = '          ' + ''.join(f'{s:>10}' for s in signers)
    print(header)
    for i, s in enumerate(signers):
        row = f'{s:>8}  ' + ''.join(f'{matrix[i,j]:10.6f}' for j in range(len(signers)))
        print(row)
    print(f'  Mean pairwise: {matrix[matrix > 0].mean():.6f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Compute MMD and JS divergence')
    parser.add_argument('--max_frames', type=int, default=5000,
                        help='Max frames per signer (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for model inference (default: cuda:0)')
    parser.add_argument('--skip_model', action='store_true',
                        help='Skip model-based methods (3 & 4), only compute raw features')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 70)
    print('MMD & JS Divergence Analysis: Cross-Performer Feature Distribution')
    print(f'Signers: {SIGNERS}')
    print(f'Max frames per signer: {args.max_frames}')
    print(f'Device: {args.device}')
    print('=' * 70)

    # ======================================================================
    # Method 1: 微调 CLIP (Conformer fine-tuned lip features, 768-dim)
    # ======================================================================
    print('\n' + '=' * 50)
    print('[Method 1] 微调 CLIP (Conformer fine-tuned lip, 768-dim)')
    print('=' * 50)
    ft_feats, ft_seq = load_features_by_signer(CONFORMER_FT_ROOT, SIGNERS, args.max_frames)
    ft_signers, ft_mmd, ft_js, _ = compute_pairwise_matrix(
        ft_feats, SIGNERS, 'Fine-tuned CLIP', 'lip')

    # ======================================================================
    # Method 2: 原生冻结 CLIP (raw CLIP features, 1024-dim)
    # ======================================================================
    print('\n' + '=' * 50)
    print('[Method 2] 原生冻结 CLIP (raw CLIP ViT-L/14, 1024-dim)')
    print('=' * 50)
    frozen_lip, frozen_lip_seq = load_features_by_signer(FROZEN_LIP_ROOT, SIGNERS, args.max_frames)
    fz_signers, fz_mmd, fz_js, _ = compute_pairwise_matrix(
        frozen_lip, SIGNERS, 'Frozen CLIP', 'lip')

    # Hand features (frozen CLIP)
    print('\n  [Hand features] 原生冻结 CLIP hand:')
    frozen_hand, frozen_hand_seq = load_features_by_signer(FROZEN_HAND_ROOT, SIGNERS, args.max_frames)
    fh_signers, fh_mmd, fh_js, _ = compute_pairwise_matrix(
        frozen_hand, SIGNERS, 'Frozen CLIP', 'hand')

    # ======================================================================
    # Methods 3 & 4: Load model and extract CSSP / CSSP+VP-Align features
    # ======================================================================
    cssp_lip = None
    vpa_lip = None
    cssp_hand = None
    vpa_hand = None

    if not args.skip_model:
        spatio_proj, spatiotemp_proj, sn_spatial, sn_spatiotem, fusion_proj = \
            load_model_components(args.device)
        device = args.device

        # ---- Method 3: Frozen + CSSP (lip) ----
        print('\n' + '=' * 50)
        print('[Method 3] 冻结 + CSSP (projection + SignerNorm, 768-dim)')
        print('=' * 50)
        print('  [Lip] Extracting CSSP features...')
        cssp_lip = extract_cssp_features(frozen_lip_seq, spatiotemp_proj,
                                          sn_spatiotem, max_frames=args.max_frames, device=device)
        cs_signers, cs_mmd, cs_js, _ = compute_pairwise_matrix(
            cssp_lip, SIGNERS, 'Frozen+CSSP', 'lip')

        print('  [Hand] Extracting CSSP features...')
        cssp_hand = extract_cssp_features(frozen_hand_seq, spatio_proj,
                                           sn_spatial, max_frames=args.max_frames, device=device)
        csh_signers, csh_mmd, csh_js, _ = compute_pairwise_matrix(
            cssp_hand, SIGNERS, 'Frozen+CSSP', 'hand')

        # ---- Method 4: Frozen + CSSP + VP-Align (lip) ----
        print('\n' + '=' * 50)
        print('[Method 4] 冻结 + CSSP + VP-Align (fusion_proj, 2048-dim)')
        print('=' * 50)
        print('  [Lip] Extracting VP-Align features...')
        vpa_lip = extract_vpalign_features(frozen_lip_seq, spatiotemp_proj,
                                            sn_spatiotem, fusion_proj, max_frames=args.max_frames, device=device)
        vp_signers, vp_mmd, vp_js, _ = compute_pairwise_matrix(
            vpa_lip, SIGNERS, 'Frozen+CSSP+VP-Align', 'lip')

        print('  [Hand] Extracting VP-Align features...')
        vpa_hand = extract_vpalign_features(frozen_hand_seq, spatio_proj,
                                             sn_spatial, fusion_proj, max_frames=args.max_frames, device=device)
        vph_signers, vph_mmd, vph_js, _ = compute_pairwise_matrix(
            vpa_hand, SIGNERS, 'Frozen+CSSP+VP-Align', 'hand')

    # ======================================================================
    # Summary output
    # ======================================================================
    print('\n\n' + '=' * 70)
    print('SUMMARY: LIP Feature MMD (RBF kernel)')
    print('=' * 70)

    for name, mmd, signers in [
        ('1. 微调 CLIP', ft_mmd, ft_signers),
        ('2. 原生冻结 CLIP', fz_mmd, fz_signers),
    ]:
        print_matrix(mmd, signers, name)

    if not args.skip_model:
        for name, mmd, signers in [
            ('3. 冻结+CSSP', cs_mmd, cs_signers),
            ('4. 冻结+CSSP+VP-Align', vp_mmd, vp_signers),
        ]:
            print_matrix(mmd, signers, name)

    print('\n' + '=' * 70)
    print('SUMMARY: LIP Feature JS Divergence')
    print('=' * 70)

    for name, js, signers in [
        ('1. 微调 CLIP', ft_js, ft_signers),
        ('2. 原生冻结 CLIP', fz_js, fz_signers),
    ]:
        print_matrix(js, signers, name)

    if not args.skip_model:
        for name, js, signers in [
            ('3. 冻结+CSSP', cs_js, cs_signers),
            ('4. 冻结+CSSP+VP-Align', vp_js, vp_signers),
        ]:
            print_matrix(js, signers, name)

    print('\n' + '=' * 70)
    print('SUMMARY: HAND Feature MMD (RBF kernel)')
    print('=' * 70)
    print_matrix(fh_mmd, fh_signers, '2. 原生冻结 CLIP (hand)')

    if not args.skip_model:
        print_matrix(csh_mmd, csh_signers, '3. 冻结+CSSP (hand)')
        print_matrix(vph_mmd, vph_signers, '4. 冻结+CSSP+VP-Align (hand)')

    print('\n' + '=' * 70)
    print('SUMMARY: HAND Feature JS Divergence')
    print('=' * 70)
    print_matrix(fh_js, fh_signers, '2. 原生冻结 CLIP (hand)')

    if not args.skip_model:
        print_matrix(csh_js, csh_signers, '3. 冻结+CSSP (hand)')
        print_matrix(vph_js, vph_signers, '4. 冻结+CSSP+VP-Align (hand)')

    # ======================================================================
    # Comparative table (mean pairwise values)
    # ======================================================================
    print('\n\n' + '=' * 70)
    print('COMPARATIVE SUMMARY: Mean Pairwise Metrics')
    print('=' * 70)
    print(f'{"Method":<30} {"Lip MMD":>10} {"Lip JS":>10} {"Hand MMD":>10} {"Hand JS":>10}')
    print('-' * 70)

    # Compute mean of upper-triangular (non-zero) entries
    def mean_pairwise(mat):
        mask = np.triu(np.ones_like(mat), k=1)
        return mat[mask > 0].mean()

    rows = [
        ('1. 微调 CLIP', mean_pairwise(ft_mmd), mean_pairwise(ft_js), None, None),
        ('2. 原生冻结 CLIP', mean_pairwise(fz_mmd), mean_pairwise(fz_js),
         mean_pairwise(fh_mmd), mean_pairwise(fh_js)),
    ]
    if not args.skip_model:
        rows += [
            ('3. 冻结+CSSP', mean_pairwise(cs_mmd), mean_pairwise(cs_js),
             mean_pairwise(csh_mmd), mean_pairwise(csh_js)),
            ('4. 冻结+CSSP+VP-Align', mean_pairwise(vp_mmd), mean_pairwise(vp_js),
             mean_pairwise(vph_mmd), mean_pairwise(vph_js)),
        ]

    for name, l_mmd, l_js, h_mmd, h_js in rows:
        lip_mmd_str = f'{l_mmd:.6f}' if l_mmd is not None else 'N/A'
        lip_js_str  = f'{l_js:.6f}'  if l_js  is not None else 'N/A'
        hand_mmd_str = f'{h_mmd:.6f}' if h_mmd is not None else 'N/A'
        hand_js_str  = f'{h_js:.6f}'  if h_js  is not None else 'N/A'
        print(f'{name:<30} {lip_mmd_str:>10} {lip_js_str:>10} {hand_mmd_str:>10} {hand_js_str:>10}')

    print('=' * 70)
    print('\nDone. Results saved in memory (modify script to save to file if needed).')


if __name__ == '__main__':
    main()
