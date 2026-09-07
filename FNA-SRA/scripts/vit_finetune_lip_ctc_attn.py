# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Fine-tuning the CLIP ViT-L/14 encoder with a CTC + Attention hybrid decoder
# for lip-reading phoneme recognition on the MCCSD dataset.
#
# Architecture (CTC/Attention hybrid):
#   Lip ROI frames (T, 3, 224, 224)
#       → ViT Encoder (CLIP ViT-L/14, fine-tuned)
#       → [encoder_outputs] (T, 1024)
#              ↓                          ↓
#       CTC Linear (1024→V)       Transformer Decoder
#       → CTC Loss                → Cross-Entropy Loss
#
#   L = (1 - ctc_weight) * L_att + ctc_weight * L_ctc
#
# Dataset split: 4:1 train/test per signer group (or per dataset), at the
# utterance level with a fixed random seed for reproducibility.
#
# Usage:
#   # Step 1 – pre-extract lip crops (run once)
#   python scripts/vit_finetune_lip_ctc_attn.py \
#       --video_root /path/to/RawVideo \
#       --save_dir   /path/to/crops \
#       --signers    LF HS WT XP \
#       --extract_crops
#
#   # Step 2 – train
#   python scripts/vit_finetune_lip_ctc_attn.py \
#       --video_root     /path/to/RawVideo \
#       --save_dir       /path/to/output \
#       --anno_root      preprocess/MCCSD \
#       --crop_cache_dir /path/to/crops/lip_crops \
#       --signers        LF HS WT XP \
#       --batch_size     4 \
#       --epochs         50 \
#       --lr             1e-4 \
#       --device         cuda:0

import argparse
import os
import os.path as osp
import glob
import re
import json
import math
import random
import copy
from collections import Counter
from typing import Optional

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward
from utils.evaluate import compute_per, compute_syllable_wer, compute_cer

_GLOBAL_SEED = 42
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


# ==========================================================================
# 1. MediaPipe lip detection  (identical to vit_extract_lip_feature.py)
# ==========================================================================

_OUTER_LIP = [
    61, 185, 40, 39, 37,  0, 267, 269, 270, 409, 291,
   375, 321, 405, 314, 17,  84, 181,  91, 146,
]

_INNER_LIP = [
    78, 191,  80,  81,  82, 13, 312, 311, 310, 415, 308,
   324, 318, 402, 317, 14,  87, 178,  88,  95,
]

_ALL_LIP = _OUTER_LIP + _INNER_LIP

_LEFT_EYE_OUTER  = 33
_RIGHT_EYE_OUTER = 263


class LipDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        padding: float = 0.20,
    ):
        self.padding = padding
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Image.Image | None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark

        lx = lms[_LEFT_EYE_OUTER].x  * w
        ly = lms[_LEFT_EYE_OUTER].y  * h
        rx = lms[_RIGHT_EYE_OUTER].x * w
        ry = lms[_RIGHT_EYE_OUTER].y * h

        angle = np.degrees(np.arctan2(ry - ly, rx - lx))

        eye_cx = (lx + rx) / 2
        eye_cy = (ly + ry) / 2
        M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        raw_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in _ALL_LIP])
        ones    = np.ones((len(raw_pts), 1))
        rot_pts = (M @ np.hstack([raw_pts, ones]).T).T

        x_min, y_min = rot_pts.min(axis=0)
        x_max, y_max = rot_pts.max(axis=0)
        lip_w, lip_h = x_max - x_min, y_max - y_min

        pad_x = lip_w * self.padding
        pad_y = lip_h * self.padding
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        crop = rotated[y1:y2, x1:x2]
        return Image.fromarray(crop) if crop.size > 0 else None

    def close(self):
        self._mesh.close()


# ==========================================================================
# 2. Video I/O
# ==========================================================================

def read_video_frames_bgr(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ==========================================================================
# 3. Annotation loading and phoneme vocabulary
# ==========================================================================

def load_annotations_from_anno_root(anno_root: str, signers: list[str]):
    """
    Load all annotation entries from {anno_root}/{signer}/ for holdout-style
    annotation directories that contain per-signer TextGrid files and per-signer
    *_info_ml.npy files.

    Priority: load from holdout_train/test_info_ml.npy if they exist (from
    make_info.py holdout split), otherwise scan TextGrid files directly.
    """
    entries = []

    holdout_train = osp.join(anno_root, 'holdout_train_info_ml.npy')
    holdout_test  = osp.join(anno_root, 'holdout_test_info_ml.npy')

    if osp.exists(holdout_train) and osp.exists(holdout_test):
        train_data = np.load(holdout_train, allow_pickle=True).item()
        test_data  = np.load(holdout_test, allow_pickle=True).item()
        for k, v in train_data.items():
            if isinstance(k, (int, np.integer)):
                v['_split'] = 'train'
                entries.append(v)
        for k, v in test_data.items():
            if isinstance(k, (int, np.integer)):
                v['_split'] = 'test'
                entries.append(v)
        return entries

    train_info = osp.join(anno_root, 'train_info_ml.npy')
    test_info  = osp.join(anno_root, 'test_info_ml.npy')
    if osp.exists(train_info) and osp.exists(test_info):
        train_data = np.load(train_info, allow_pickle=True).item()
        test_data  = np.load(test_info, allow_pickle=True).item()
        for k, v in train_data.items():
            if isinstance(k, (int, np.integer)):
                v['_split'] = 'train'
                entries.append(v)
        for k, v in test_data.items():
            if isinstance(k, (int, np.integer)):
                v['_split'] = 'test'
                entries.append(v)
        return entries

    for signer in signers:
        tg_dir = osp.join(anno_root, signer)
        if not osp.isdir(tg_dir):
            continue
        tg_files = sorted(glob.glob(osp.join(tg_dir, f'{signer}-*-V.TextGrid')))
        for tg_path in tg_files:
            m = re.match(r'([A-Z]+)-(\d+)-V\.TextGrid', osp.basename(tg_path))
            if not m:
                continue
            fileid = f'{m.group(1)}-{m.group(2)}'
            intervals = _parse_textgrid(tg_path)
            entries.append({
                'fileid': fileid,
                'signer': signer,
                'phoneme_intervals': intervals,
                '_split': 'unknown',
            })

    return entries


def _parse_textgrid(tg_path: str) -> list:
    intervals = []
    with open(tg_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_intervals = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'intervals:' in line and 'size' in line:
            in_intervals = True
            i += 1
            continue
        if in_intervals and line.startswith('intervals'):
            xmin_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            xmax_line = lines[i + 2].strip() if i + 2 < len(lines) else ''
            text_line = lines[i + 3].strip() if i + 3 < len(lines) else ''

            xmin_m = re.search(r'[\d.]+', xmin_line)
            xmax_m = re.search(r'[\d.]+', xmax_line)
            text_m  = re.search(r'text\s*=\s*"([^"]*)"', text_line)

            if xmin_m and xmax_m and text_m:
                t_start = float(xmin_m.group())
                t_end   = float(xmax_m.group())
                label   = text_m.group(1).strip()
                if label:
                    intervals.append([t_start, t_end, label])
            i += 4
        else:
            i += 1

    return intervals


def get_phoneme_sequence(entry: dict) -> str:
    intervals = entry.get('phoneme_intervals', [])
    if intervals:
        return ' '.join(iv[2] for iv in intervals if iv[2])
    gloss = entry.get('gloss', '')
    if gloss:
        return gloss.replace(' - ', ' ').replace('-', ' ')
    return ''


def build_phoneme_vocab(entries: list[dict]) -> dict[str, int]:
    counter = Counter()
    for entry in entries:
        seq = get_phoneme_sequence(entry)
        for token in seq.split():
            counter[token] += 1

    specials = ['<blank>', '<sos>', '<eos>', '<pad>']
    vocab = {s: i for i, s in enumerate(specials)}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


# ==========================================================================
# 4. Video root resolution
# ==========================================================================

def build_signer_root_map(args) -> dict[str, str]:
    root_map = {}
    for signer in args.signers:
        root_map[signer] = args.video_root
    if args.sub_video_root:
        for signer in args.sub_signers:
            root_map[signer] = args.sub_video_root
    for signer in args.signers:
        if signer not in root_map:
            root_map[signer] = args.video_root
    return root_map


def resolve_video_path(signer: str, fileid: str, root_map: dict[str, str]) -> str | None:
    root = root_map.get(signer)
    if root is None:
        return None
    path = osp.join(root, signer, f'{fileid}.mp4')
    if osp.exists(path):
        return path
    for alt_root in set(root_map.values()):
        path = osp.join(alt_root, signer, f'{fileid}.mp4')
        if osp.exists(path):
            return path
    return None


# ==========================================================================
# 5. Lip ROI crop extraction and caching
# ==========================================================================

def extract_and_cache_crops(args, entries: list[dict], root_map: dict[str, str]) -> dict[str, str]:
    """
    For every entry, detect lip crops from the video and save them as an
    .npy file of shape (T, 3, H, W).  Returns a mapping fileid → crop_path.
    """
    cache_dir = osp.join(args.save_dir, 'lip_crops')
    os.makedirs(cache_dir, exist_ok=True)

    detector = LipDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.padding,
    )

    crop_map = {}
    try:
        unique_videos = sorted({(e['signer'], e['fileid']) for e in entries})
        for signer, fileid in tqdm.tqdm(unique_videos, desc='Extracting lip crops'):
            save_path = osp.join(cache_dir, signer, f'{fileid}.npy')
            os.makedirs(osp.dirname(save_path), exist_ok=True)

            if osp.exists(save_path):
                crop_map[fileid] = save_path
                continue

            video_path = resolve_video_path(signer, fileid, root_map)
            if video_path is None:
                print(f'[WARN] Video not found: {signer}/{fileid}.mp4')
                continue

            frames_bgr = read_video_frames_bgr(video_path)
            if not frames_bgr:
                print(f'[WARN] No frames: {video_path}')
                continue

            crops = []
            for i, frame in enumerate(frames_bgr):
                if i % args.frame_step != 0:
                    continue
                h, w = frame.shape[:2]
                scale = args.max_frame_size / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_LINEAR)
                roi = detector.detect(frame)
                if roi is not None:
                    arr = np.array(roi.resize((224, 224), Image.BILINEAR))
                    crops.append(arr)

            if not crops:
                print(f'[WARN] No lip crops detected in {video_path}')
                continue

            crops_np = np.stack(crops, axis=0).astype(np.uint8)
            np.save(save_path, crops_np)
            crop_map[fileid] = save_path

    finally:
        detector.close()

    return crop_map


# ==========================================================================
# 6. Dataset
# ==========================================================================

class LipROIDataset(Dataset):
    """
    Loads pre-extracted lip ROI crops (T, H, W, 3) and phoneme labels.

    Each item returns:
        frames:    Tensor (T, 3, 224, 224) – resized, normalised lip crops
        label_ids: Tensor (L,) – phoneme token IDs (without <sos>/<eos>)
        label_str: str – space-separated phoneme sequence
        fileid:    str
    """

    def __init__(
        self,
        entries: list[dict],
        vocab: dict[str, int],
        crop_map: dict[str, str],
        image_processor: AutoImageProcessor,
        split: str = 'train',
    ):
        self.vocab = vocab
        self.crop_map = crop_map
        self.image_processor = image_processor
        self.split = split

        if split in ('train', 'test'):
            self.entries = [e for e in entries if e.get('_split', 'train') == split]
        else:
            self.entries = list(entries)

        self.entries = [e for e in self.entries if e['fileid'] in crop_map]
        if not self.entries:
            raise RuntimeError(
                f'No valid entries for split="{split}". '
                f'Run with --extract_crops first.'
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        fileid = entry['fileid']
        crop_path = self.crop_map[fileid]

        crops_np = np.load(crop_path)  # (T, H, W, 3) uint8 RGB
        T = crops_np.shape[0]

        images = [Image.fromarray(crops_np[t]) for t in range(T)]
        pv = self.image_processor(list(images), return_tensors='pt').pixel_values  # (T, 3, 224, 224)

        phoneme_str = get_phoneme_sequence(entry)
        label_ids = torch.tensor(
            [self.vocab.get(t, self.vocab['<blank>']) for t in phoneme_str.split()],
            dtype=torch.long
        )

        return {
            'frames':     pv,
            'label_ids':  label_ids,
            'label_str':  phoneme_str,
            'fileid':     fileid,
            'num_frames': T,
        }


def collate_lip_batch(batch: list[dict]) -> dict:
    frames_list     = [item['frames'] for item in batch]
    label_ids_list  = [item['label_ids'] for item in batch]
    label_strs      = [item['label_str'] for item in batch]
    fileids         = [item['fileid'] for item in batch]
    frame_lens      = torch.tensor([item['num_frames'] for item in batch], dtype=torch.long)
    label_lens      = torch.tensor([len(ids) for ids in label_ids_list], dtype=torch.long)

    frames_padded = pad_sequence(frames_list, batch_first=True)       # (B, max_T, 3, 224, 224)
    labels_padded = pad_sequence(label_ids_list, batch_first=True,
                                  padding_value=0)                     # (B, max_L)

    return {
        'frames':      frames_padded,
        'frame_lens':  frame_lens,
        'labels':      labels_padded,
        'label_lens':  label_lens,
        'label_strs':  label_strs,
        'fileids':     fileids,
    }


# ==========================================================================
# 7. Model components
# ==========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerAttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.embed_scale = math.sqrt(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        tgt_emb = self.embedding(tgt) * self.embed_scale
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)


class LipCTCAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_name: str = 'openai/clip-vit-large-patch14',
        cache_dir: Optional[str] = None,
        encoder_dim: int = 1024,
        d_model: int = 512,
        num_dec_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 3,
        blank_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_vit_batch: int = 8,
    ):
        super().__init__()
        self.vocab_size   = vocab_size
        self.blank_idx    = blank_idx
        self.sos_idx      = sos_idx
        self.eos_idx      = eos_idx
        self.pad_idx      = pad_idx
        self.d_model      = d_model
        self.max_vit_batch = max_vit_batch

        self.vit = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=False, cache_dir=cache_dir
        )

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

        self.decoder = TransformerAttentionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def encode(self, frames: torch.Tensor, frame_lens: torch.Tensor):
        """
        Args:
            frames:     (B, max_T, 3, 224, 224)
            frame_lens: (B,) actual frame counts
        Returns:
            encoder_out: (B, max_T, d_model)
            mask:        (B, max_T) bool, True = pad position
        """
        B, max_T = frames.shape[:2]

        flat_frames = frames.view(B * max_T, 3, 224, 224)
        cls_list = []
        for i in range(0, flat_frames.size(0), self.max_vit_batch):
            chunk = flat_frames[i:i + self.max_vit_batch]
            vit_out = self.vit(chunk, output_hidden_states=True).hidden_states[-1]
            cls_list.append(vit_out[:, 0, :])
        cls_tokens = torch.cat(cls_list, dim=0).view(B, max_T, -1)

        encoder_out = self.encoder_proj(cls_tokens)                # (B, max_T, d_model)
        encoder_out = self.pos_encoder(encoder_out)

        mask = torch.arange(max_T, device=frames.device).unsqueeze(0) >= frame_lens.unsqueeze(1)

        return encoder_out, mask

    def forward(
        self,
        frames: torch.Tensor,
        frame_lens: torch.Tensor,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        ctc_weight: float = 0.3,
    ):
        encoder_out, encoder_mask = self.encode(frames, frame_lens)

        ctc_loss = self._compute_ctc_loss(encoder_out, encoder_mask, labels, label_lens)

        att_loss = self._compute_att_loss(encoder_out, encoder_mask, labels, label_lens)

        loss = (1.0 - ctc_weight) * att_loss + ctc_weight * ctc_loss

        return {
            'loss':     loss,
            'ctc_loss': ctc_loss,
            'att_loss': att_loss,
        }

    def _compute_ctc_loss(self, encoder_out, encoder_mask, labels, label_lens):
        B, max_T, D = encoder_out.shape

        ctc_logits = self.ctc_head(encoder_out)                   # (B, max_T, V)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)              # (max_T, B, V)

        input_lengths = (~encoder_mask).sum(dim=1).long()          # (B,)
        input_lengths = input_lengths.clamp(min=1)

        target_lengths = label_lens.clamp(min=1)

        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            labels,
            input_lengths,
            target_lengths,
            blank=self.blank_idx,
            reduction='mean',
            zero_infinity=True,
        )
        return ctc_loss

    def _compute_att_loss(self, encoder_out, encoder_mask, labels, label_lens):
        B, max_T, D = encoder_out.shape
        device = encoder_out.device
        max_L = labels.size(1)

        sos = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        eos = torch.full((B, 1), self.eos_idx, dtype=torch.long, device=device)
        tgt_in  = torch.cat([sos, labels], dim=1)                  # (B, 1+L)
        tgt_out = torch.cat([labels, eos], dim=1)                  # (B, L+1)
        tgt_len = max_L + 1

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

        tgt_pad_mask = torch.arange(tgt_len, device=device).unsqueeze(0) >= (label_lens + 1).unsqueeze(1)

        logits = self.decoder(
            tgt_in, encoder_out,
            memory_key_padding_mask=encoder_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )                                                           # (B, tgt_len, V)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tgt_out.reshape(-1),
            ignore_index=self.pad_idx,
        )
        return loss

    @torch.no_grad()
    def decode(
        self,
        frames: torch.Tensor,
        frame_lens: torch.Tensor,
        max_len: int = 200,
        ctc_weight: float = 0.3,
        beam_width: int = 1,
    ) -> list[str]:
        """
        Greedy or beam decoding using the attention decoder (teacher-forcing
        during inference; CTC is used for rescoring when beam_width > 1).
        """
        self.eval()
        encoder_out, encoder_mask = self.encode(frames, frame_lens)
        B, max_T, D = encoder_out.shape
        device = encoder_out.device

        if beam_width <= 1:
            return self._greedy_decode(encoder_out, encoder_mask, max_len, device, B)
        else:
            return self._ctc_prefix_beam_decode(
                encoder_out, encoder_mask, max_len, device, B, beam_width, ctc_weight
            )

    def _greedy_decode(self, encoder_out, encoder_mask, max_len, device, B):
        generated = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(B)]

        for _ in range(max_len):
            if finished.all():
                break
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=device
            )
            logits = self.decoder(
                generated, encoder_out,
                memory_key_padding_mask=encoder_mask,
                tgt_mask=tgt_mask,
            )
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)

            for b in range(B):
                if finished[b]:
                    continue
                tok = next_token[b].item()
                if tok == self.eos_idx or tok == self.pad_idx:
                    finished[b] = True
                else:
                    outputs[b].append(tok)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        id2token = {v: k for k, v in self._id2token_map.items()} if hasattr(self, '_id2token_map') else {}
        results = []
        for b in range(B):
            if id2token:
                results.append(' '.join(id2token.get(t, '<unk>') for t in outputs[b]))
            else:
                results.append(' '.join(str(t) for t in outputs[b]))
        return results

    def _ctc_prefix_beam_decode(self, encoder_out, encoder_mask, max_len, device, B,
                                 beam_width, ctc_weight):
        ctc_logits = self.ctc_head(encoder_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        results = []
        id2token = {}
        if hasattr(self, '_id2token_map'):
            id2token = self._id2token_map

        for b in range(B):
            enc_b = encoder_out[b:b+1, :, :]
            mask_b = encoder_mask[b:b+1, :] if encoder_mask is not None else None
            enc_len = (~mask_b).sum().item() if mask_b is not None else enc_b.size(1)

            ctc_lp = ctc_log_probs[b:b+1, :int(enc_len), :]

            hyp = [(0.0, [self.sos_idx], torch.tensor(0.0, device=device))]

            for step in range(max_len):
                new_hyps = []
                for total_score, tokens, ctc_score in hyp:
                    tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt.size(1), device=device
                    )
                    logits = self.decoder(tgt, enc_b, memory_key_padding_mask=mask_b, tgt_mask=tgt_mask)
                    att_log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                    topk_scores, topk_ids = att_log_probs.topk(beam_width)
                    for score, tok_id in zip(topk_scores.tolist(), topk_ids.tolist()):
                        new_tokens = tokens + [tok_id]
                        new_total = total_score + score
                        new_hyps.append((new_total, new_tokens, ctc_score))

                new_hyps.sort(key=lambda x: x[0], reverse=True)
                hyp = new_hyps[:beam_width]

                if all(t[-1] == self.eos_idx for _, t, _ in hyp):
                    break

            best = hyp[0][1]
            if id2token:
                tokens = [id2token.get(t, '<unk>') for t in best
                          if t not in (self.sos_idx, self.eos_idx, self.pad_idx)]
            else:
                tokens = [str(t) for t in best
                          if t not in (self.sos_idx, self.eos_idx, self.pad_idx)]
            results.append(' '.join(tokens))

        return results

    def set_id2token(self, vocab: dict[str, int]):
        self._id2token_map = {v: k for k, v in vocab.items()}


# ==========================================================================
# 8. Training utilities
# ==========================================================================

def split_entries(entries: list[dict], train_ratio: float = 0.8, seed: int = 42):
    fileids = sorted({e['fileid'] for e in entries})
    rng = random.Random(seed)
    rng.shuffle(fileids)
    n_train = int(len(fileids) * train_ratio)
    train_ids = set(fileids[:n_train])
    test_ids  = set(fileids[n_train:])

    for e in entries:
        e['_split'] = 'train' if e['fileid'] in train_ids else 'test'

    n_train_e = sum(1 for e in entries if e['_split'] == 'train')
    n_test_e  = sum(1 for e in entries if e['_split'] == 'test')
    print(f'Train samples: {n_train_e}  ({n_train} unique utterances)')
    print(f'Test  samples: {n_test_e}  ({len(test_ids)} unique utterances)')

    return entries


def train_one_epoch(model, dataloader, optimizer, device, ctc_weight, epoch, args, scaler=None):
    model.train()
    total_loss = 0.0
    total_ctc  = 0.0
    total_att  = 0.0
    n_batches  = 0
    use_amp = scaler is not None

    pbar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch} [train]')
    for batch in pbar:
        frames     = batch['frames'].to(device)
        frame_lens = batch['frame_lens'].to(device)
        labels     = batch['labels'].to(device)
        label_lens = batch['label_lens'].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(frames, frame_lens, labels, label_lens, ctc_weight=ctc_weight)
        loss = outputs['loss']

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_ctc  += outputs['ctc_loss'].item()
        total_att  += outputs['att_loss'].item()
        n_batches  += 1

        pbar.set_postfix({
            'loss': f'{total_loss / n_batches:.4f}',
            'ctc':  f'{total_ctc / n_batches:.4f}',
            'att':  f'{total_att / n_batches:.4f}',
        })

    return {
        'loss':     total_loss / n_batches,
        'ctc_loss': total_ctc / n_batches,
        'att_loss': total_att / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, vocab, split='test'):
    model.eval()
    all_preds  = []
    all_refs   = []
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm.tqdm(dataloader, desc=f'[{split}]')
    for batch in pbar:
        frames     = batch['frames'].to(device)
        frame_lens = batch['frame_lens'].to(device)
        labels     = batch['labels'].to(device)
        label_lens = batch['label_lens'].to(device)
        ref_strs   = batch['label_strs']

        outputs = model(frames, frame_lens, labels, label_lens)
        total_loss += outputs['loss'].item()
        n_batches  += 1

        preds = model.decode(frames, frame_lens, beam_width=1)
        all_preds.extend(preds)
        all_refs.extend(ref_strs)

    per = compute_per(all_preds, all_refs)
    syllable_wer = compute_syllable_wer(all_preds, all_refs)
    cer = compute_cer(all_preds, all_refs)

    return {
        'loss':         total_loss / max(n_batches, 1),
        'per':          per,
        'syllable_wer': syllable_wer,
        'cer':          cer,
        'preds':        all_preds,
        'refs':         all_refs,
    }


# ==========================================================================
# 8. Main
# ==========================================================================

def get_parser():
    p = argparse.ArgumentParser(
        description='Fine-tune CLIP ViT-L/14 with CTC+Attention hybrid decoder '
                    'for lip-reading phoneme recognition on MCCSD.'
    )
    p.add_argument('--video_root', required=True,
                   help='Root dir of raw .mp4 videos, e.g. /path/to/RawVideo')
    p.add_argument('--sub_video_root', default=None,
                   help='Secondary video root for mccsd_sub signers (YX, YZ). '
                        'Each signer dir (e.g. YX/) is expected directly under this root. '
                        'If not set, --video_root is used for all signers.')
    p.add_argument('--save_dir', required=True,
                   help='Output directory for model checkpoints and logs')
    p.add_argument('--anno_root', default='preprocess/MCCSD',
                   help='Annotation root directory. Supports: '
                        '(a) holdout_*_info_ml.npy for holdout split, '
                        '(b) train/test_info_ml.npy for 6H pre-defined split, '
                        '(c) per-signer TextGrid files for auto 4:1 split.')
    p.add_argument('--crop_cache_dir', default=None,
                   help='Directory of pre-extracted lip crops. If not set, '
                        'defaults to {save_dir}/lip_crops.')
    p.add_argument('--signers', nargs='+', default=['LF', 'HS', 'WT', 'XP'],
                   help='Signer IDs to include')
    p.add_argument('--sub_signers', nargs='+', default=['YX', 'YZ'],
                   help='Signer IDs located under --sub_video_root')

    p.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    p.add_argument('--cache_dir', default=None,
                   help='HuggingFace model cache directory')

    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--ctc_weight', type=float, default=0.3,
                   help='Weight of CTC loss; attention loss weight = 1 - ctc_weight')
    p.add_argument('--warmup_epochs', type=int, default=5,
                   help='Number of warmup epochs where ctc_weight linearly increases')

    p.add_argument('--d_model', type=int, default=512,
                   help='Decoder hidden dimension')
    p.add_argument('--num_dec_layers', type=int, default=4,
                   help='Number of Transformer decoder layers')
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--dim_feedforward', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--train_ratio', type=float, default=0.8,
                   help='Train split ratio (default 0.8 = 4:1)')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--eval_only', action='store_true',
                   help='Skip training, only evaluate a saved checkpoint')
    p.add_argument('--checkpoint', default=None,
                   help='Path to checkpoint .pt file for eval or resume')

    p.add_argument('--extract_crops', action='store_true',
                   help='Only extract and cache lip crops, then exit')

    p.add_argument('--padding', type=float, default=0.20)
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    p.add_argument('--frame_step', type=int, default=2,
                   help='Process every Nth frame (default 2 → ~2x speed). '
                        'Set to 1 for all frames.')
    p.add_argument('--max_frame_size', type=int, default=480,
                   help='Resize video frame so its longer side ≤ this before '
                        'MediaPipe detection (default 480). Speeds up face mesh.')

    p.add_argument('--save_every', type=int, default=10,
                   help='Save checkpoint every N epochs')
    p.add_argument('--eval_every', type=int, default=5,
                   help='Evaluate every N epochs')

    p.add_argument('--extract_features_after_train', action='store_true',
                   help='After training, use the best checkpoint to extract ViT '
                        'features from ALL videos (train+test) and save as .npy '
                        'files for downstream SLT training.')
    p.add_argument('--feat_save_dir', default=None,
                   help='Output directory for extracted features. '
                        'Defaults to {save_dir}/features.')
    p.add_argument('--feat_batch_size', type=int, default=32,
                   help='Batch size for feature extraction')
    p.add_argument('--feat_dir_suffix', default='_ft',
                   help='Feature directory suffix (default "_ft")')
    p.add_argument('--max_vit_batch', type=int, default=8,
                   help='Max number of frames per ViT forward chunk '
                        '(reduces GPU memory, default 8)')
    p.add_argument('--amp', action='store_true', default=True,
                   help='Use Automatic Mixed Precision (AMP) for faster training')
    p.add_argument('--no_amp', action='store_false', dest='amp',
                   help='Disable AMP')

    return p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = get_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    crop_cache_dir = args.crop_cache_dir or osp.join(args.save_dir, 'lip_crops')

    root_map = build_signer_root_map(args)
    all_signers = sorted(set(args.signers) | set(args.sub_signers))
    print(f'Video root map:')
    for s in all_signers:
        print(f'  {s} → {root_map.get(s, "N/A")}')

    print('=' * 60)
    print('Loading annotations ...')
    entries = load_annotations_from_anno_root(args.anno_root, all_signers)
    print(f'Loaded {len(entries)} annotation entries.')

    if args.extract_crops:
        print('Extracting and caching lip crops ...')
        crop_map = extract_and_cache_crops(args, entries, root_map)
        print(f'Cached {len(crop_map)} videos to {crop_cache_dir}')
        return

    crop_map = {}
    for signer in all_signers:
        signer_dir = osp.join(crop_cache_dir, signer)
        if osp.isdir(signer_dir):
            for fname in os.listdir(signer_dir):
                if fname.endswith('.npy'):
                    fileid = fname[:-4]
                    crop_map[fileid] = osp.join(signer_dir, fname)
    print(f'Found {len(crop_map)} cached lip crop files.')

    if not crop_map:
        print('No cached lip crops found. Run with --extract_crops first.')
        return

    if not any(e.get('_split') in ('train', 'test') for e in entries):
        print(f'Splitting data {args.train_ratio:.0%}:{1 - args.train_ratio:.0%} (seed={args.seed}) ...')
        entries = split_entries(entries, args.train_ratio, args.seed)

    print('Building phoneme vocabulary ...')
    train_entries = [e for e in entries if e.get('_split') == 'train']
    vocab = build_phoneme_vocab(train_entries)
    print(f'Vocabulary size: {len(vocab)}')

    id2token = {v: k for k, v in vocab.items()}

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    train_dataset = LipROIDataset(
        entries, vocab, crop_map, image_processor, split='train'
    )
    test_dataset = LipROIDataset(
        entries, vocab, crop_map, image_processor, split='test'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_lip_batch,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_lip_batch,
        pin_memory=True,
    )

    print(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = LipCTCAttentionModel(
        vocab_size=len(vocab),
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        d_model=args.d_model,
        num_dec_layers=args.num_dec_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_idx=vocab['<pad>'],
        blank_idx=vocab['<blank>'],
        sos_idx=vocab['<sos>'],
        eos_idx=vocab['<eos>'],
        max_vit_batch=args.max_vit_batch,
    )
    model.set_id2token(vocab)
    model.to(device)

    model.vit.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total_params / 1e6:.2f}M')
    print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    start_epoch = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    if args.checkpoint and osp.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from checkpoint: {args.checkpoint} (epoch {start_epoch})')

    if args.eval_only:
        print('\nEvaluating ...')
        metrics = evaluate(model, test_loader, device, vocab, split='test')
        print(f'PER: {metrics["per"]:.2f}%')
        print(f'Syllable WER: {metrics["syllable_wer"]:.2f}%')
        print(f'CER: {metrics["cer"]:.2f}%')
        print(f'Loss: {metrics["loss"]:.4f}')
        return

    os.makedirs(args.save_dir, exist_ok=True)
    best_per = float('inf')
    best_epoch = 0

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            current_ctc_weight = args.ctc_weight * (epoch / args.warmup_epochs)
        else:
            current_ctc_weight = args.ctc_weight

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, current_ctc_weight, epoch, args, scaler
        )
        scheduler.step()

        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {train_metrics["loss"]:.4f} '
              f'(CTC: {train_metrics["ctc_loss"]:.4f}, '
              f'ATT: {train_metrics["att_loss"]:.4f}) '
              f'| LR: {scheduler.get_last_lr()[0]:.2e} '
              f'| CTC weight: {current_ctc_weight:.2f}')

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, test_loader, device, vocab, split='test')
            per = val_metrics['per']
            print(f'  → Test  PER: {per:.2f}%  '
                  f'Syllable WER: {val_metrics["syllable_wer"]:.2f}%  '
                  f'CER: {val_metrics["cer"]:.2f}%  '
                  f'Loss: {val_metrics["loss"]:.4f}')

            if per < best_per:
                best_per = per
                best_epoch = epoch
                best_path = osp.join(args.save_dir, 'best_model.pt')
                ckpt_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': vocab,
                    'per': per,
                    'args': vars(args),
                }
                if scaler is not None:
                    ckpt_dict['scaler_state_dict'] = scaler.state_dict()
                torch.save(ckpt_dict, best_path)
                print(f'  → Saved best model (PER={per:.2f}%) to {best_path}')

        if epoch % args.save_every == 0:
            ckpt_path = osp.join(args.save_dir, f'checkpoint_epoch{epoch}.pt')
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'args': vars(args),
            }
            if scaler is not None:
                ckpt_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(ckpt_dict, ckpt_path)

    print(f'\nTraining finished. Best PER: {best_per:.2f}% at epoch {best_epoch}')

    final_ckpt = osp.join(args.save_dir, 'final_model.pt')
    final_dict = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'args': vars(args),
    }
    if scaler is not None:
        final_dict['scaler_state_dict'] = scaler.state_dict()
    torch.save(final_dict, final_ckpt)
    print(f'Final model saved to {final_ckpt}')

    if args.extract_features_after_train:
        best_ckpt = osp.join(args.save_dir, 'best_model.pt')
        if not osp.exists(best_ckpt):
            print('[WARN] best_model.pt not found, skipping feature extraction.')
            return

        print('\n' + '=' * 60)
        print('Extracting ViT features from ALL videos using fine-tuned encoder ...')
        _extract_features_post_train(args, best_ckpt)


def _extract_features_post_train(args, checkpoint_path: str):
    feat_save_dir = args.feat_save_dir or osp.join(args.save_dir, 'features')
    _model_name = os.path.split(args.model_name)[-1]
    feat_dir_name = f'{_model_name}_lip_feat_mccsd{args.feat_dir_suffix}'

    from transformers import AutoImageProcessor, CLIPVisionModel

    class _ViTReader:
        def __init__(self):
            self.model = CLIPVisionModel.from_pretrained(
                args.model_name, output_hidden_states=True, cache_dir=args.cache_dir
            )
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            vit_state = {k[4:]: v for k, v in state_dict.items() if k.startswith('vit.')}
            self.model.load_state_dict(vit_state, strict=False)
            self.model.to(args.device).eval()
            self.image_processor = AutoImageProcessor.from_pretrained(args.model_name)

        @torch.no_grad()
        def get_feats(self, images):
            pv = self.image_processor(
                list(images), return_tensors='pt'
            ).to(args.device).pixel_values
            out = self.model(pv).hidden_states[-1]
            return out[:, 0].cpu().numpy()

    detector = LipDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.padding,
    )
    reader = _ViTReader()

    root_map = build_signer_root_map(args)
    all_signers = sorted(set(args.signers) | set(args.sub_signers))

    try:
        for signer in all_signers:
            root = root_map.get(signer, args.video_root)
            signer_dir = osp.join(root, signer)
            if not osp.isdir(signer_dir):
                print(f'[WARN] Not found: {signer_dir}, skipping.')
                continue

            video_files = sorted(glob.glob(osp.join(signer_dir, '*.mp4')))
            if not video_files:
                print(f'[WARN] No .mp4 in {signer_dir}, skipping.')
                continue

            save_dir = osp.join(feat_save_dir, feat_dir_name, signer)
            os.makedirs(save_dir, exist_ok=True)

            for video_path in tqdm.tqdm(video_files, desc=f'[lip_ft][{signer}]'):
                vid = osp.splitext(osp.basename(video_path))[0]
                save_path = osp.join(save_dir, f'{vid}.npy')
                if osp.exists(save_path):
                    continue

                frames_bgr = read_video_frames_bgr(video_path)
                if not frames_bgr:
                    continue

                lip_crops = []
                for i, frame in enumerate(frames_bgr):
                    if i % args.frame_step != 0:
                        continue
                    h, w = frame.shape[:2]
                    scale = args.max_frame_size / max(h, w)
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_LINEAR)
                    roi = detector.detect(frame)
                    if roi is not None:
                        lip_crops.append(roi)

                if not lip_crops:
                    continue

                feats = []
                for j in range(0, len(lip_crops), args.feat_batch_size):
                    feats.append(reader.get_feats(lip_crops[j: j + args.feat_batch_size]))
                feats = np.concatenate(feats, axis=0)

                np.save(save_path, feats)

        print(f'Features saved to {osp.join(feat_save_dir, feat_dir_name)}')
    finally:
        detector.close()


if __name__ == '__main__':
    main()
