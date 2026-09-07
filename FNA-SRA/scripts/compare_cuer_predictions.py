"""
compare_cuer_predictions.py

对多个 fold 模型（每个 fold 的测试集来自不同的 cuer）在共同 target 上进行推理，
输出相同 target 下各模型（cuer）的预测结果对比，从而说明不同 cuer 之间的差异性。

用法示例:
    python scripts/compare_cuer_predictions.py \
        --ckpts \
            logs/2026-03-11T13-55-01_mccsd_cv_WT/checkpoints/epoch=00353-step=0059826-wer=7.61.ckpt \
            logs/2026-03-10T16-53-48_mccsd_cv_LF/checkpoints/epoch=00201-step=0034138-wer=12.74.ckpt \
        --configs \
            logs/2026-03-11T13-55-01_mccsd_cv_WT/configs/2026-03-11T13-55-01-project.yaml \
            logs/2026-03-10T16-53-48_mccsd_cv_LF/configs/2026-03-10T16-53-48-project.yaml \
        --labels WT LF \
        --split test \
        --top_n 30 \
        --output_file results/cuer_comparison.txt
"""

import argparse
import os
import sys
import torch
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from fna_sra.t5_sra import FlanT5SLT
from dataset.mccsd import MCCSD


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str, device: torch.device) -> FlanT5SLT:
    """Load a FlanT5SLT model from config + checkpoint."""
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.params

    model = FlanT5SLT(**OmegaConf.to_container(model_cfg, resolve=True))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def build_dataset(config_path: str, split: str) -> MCCSD:
    """Build the MCCSD dataset for a given split from the saved project config."""
    cfg = OmegaConf.load(config_path)
    data_params = cfg.data.params[split].params
    data_params = OmegaConf.to_container(data_params, resolve=True)
    return MCCSD(**data_params)


@torch.no_grad()
def run_inference(
    model: FlanT5SLT,
    dataset: MCCSD,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> List[Dict]:
    """
    Run model inference over the dataset.

    Returns a list of dicts, one per sample:
        {
            'id':        fileid,
            'signer':    signer id,
            'target':    ground-truth phoneme/gloss string,
            'predicted': model prediction string,
        }
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=MCCSD.collate_fn,
    )

    results = []
    for batch in loader:
        inputs = model.get_inputs(batch)
        if not inputs['text']:          # skip empty batches
            continue

        visual_outputs, visual_masks = model.prepare_visual_inputs(inputs)
        visual_outputs = model.fusion_proj(visual_outputs)

        input_embeds, input_masks, output_tokens, _ = model.prepare_inputs(
            visual_outputs, visual_masks, inputs, split='test', batch_idx=0
        )

        generated = model.t5_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            num_beams=5,
            max_length=model.max_txt_len,
            top_p=0.9,
            do_sample=False,     # greedy beam search for reproducibility
        )

        pred_strs = model.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
        ref_strs  = model.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)

        for i in range(len(inputs['ids'])):
            results.append({
                'id':        inputs['ids'][i],
                'signer':    inputs['signers'][i],
                'target':    ref_strs[i].lower().strip(),
                'predicted': pred_strs[i].lower().strip(),
            })

    return results


def phoneme_wer(ref: str, hyp: str) -> float:
    """Token-level WER (each space-separated token = one phoneme)."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # Standard edit distance
    dp = list(range(len(h) + 1))
    for i, ri in enumerate(r):
        new_dp = [i + 1] + [0] * len(h)
        for j, hj in enumerate(h):
            if ri == hj:
                new_dp[j + 1] = dp[j]
            else:
                new_dp[j + 1] = 1 + min(dp[j], dp[j + 1], new_dp[j])
        dp = new_dp
    return dp[len(h)] / len(r)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare cuer predictions across fold models")
    parser.add_argument('--ckpts',   nargs='+', required=True,
                        help='Checkpoint paths, one per fold model')
    parser.add_argument('--configs', nargs='+', required=True,
                        help='Project YAML config paths, one per fold model')
    parser.add_argument('--labels',  nargs='+', default=None,
                        help='Human-readable labels for each fold (e.g. WT LF HS XP)')
    parser.add_argument('--split',   default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use for each model (default: test)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--top_n',   type=int, default=30,
                        help='Number of shared-target examples to show per model')
    parser.add_argument('--output_file', default=None,
                        help='Optional path to save the comparison table as text')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.configs), \
        "--ckpts and --configs must have the same number of entries"

    n_models = len(args.ckpts)
    labels = args.labels if args.labels else [f"Model_{i}" for i in range(n_models)]
    assert len(labels) == n_models, "--labels must match --ckpts count"

    device = torch.device(args.device)

    # -----------------------------------------------------------------------
    # 1. Load models and run inference
    # -----------------------------------------------------------------------
    all_results: Dict[str, List[Dict]] = {}   # label -> list of result dicts

    for label, cfg_path, ckpt_path in zip(labels, args.configs, args.ckpts):
        print(f"\n{'='*60}")
        print(f"Processing model [{label}]")
        print(f"  config : {cfg_path}")
        print(f"  ckpt   : {ckpt_path}")

        model   = load_model(cfg_path, ckpt_path, device)
        dataset = build_dataset(cfg_path, args.split)
        print(f"  dataset size ({args.split}): {len(dataset)}")

        results = run_inference(model, dataset, args.batch_size, device, args.num_workers)
        all_results[label] = results

        # Free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. Build per-target index for each model
    # -----------------------------------------------------------------------
    # target -> { label -> [ {id, signer, predicted} ] }
    target_index: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for label, results in all_results.items():
        for r in results:
            target_index[r['target']][label].append({
                'id':        r['id'],
                'signer':    r['signer'],
                'predicted': r['predicted'],
            })

    # Keep only targets that appear in ALL models
    shared_targets = [
        t for t, model_dict in target_index.items()
        if all(lbl in model_dict and len(model_dict[lbl]) > 0 for lbl in labels)
    ]
    print(f"\nShared targets across all {n_models} models: {len(shared_targets)}")

    # -----------------------------------------------------------------------
    # 3. Score and rank targets by prediction diversity
    # -----------------------------------------------------------------------
    def diversity_score(target: str) -> float:
        """
        Average pairwise WER among the first prediction of each model.
        Higher = more diverse (more interesting to show).
        """
        preds = [target_index[target][lbl][0]['predicted'] for lbl in labels]
        pairs = 0
        total_wer = 0.0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                total_wer += phoneme_wer(preds[i], preds[j])
                pairs += 1
        return total_wer / max(pairs, 1)

    ranked = sorted(shared_targets, key=diversity_score, reverse=True)

    # -----------------------------------------------------------------------
    # 4. Format and print comparison table
    # -----------------------------------------------------------------------
    sep  = "=" * 80
    sep2 = "-" * 80

    lines = []
    lines.append(sep)
    lines.append("CROSS-CUER PREDICTION COMPARISON")
    lines.append(f"Models: {', '.join(labels)}")
    lines.append(f"Split : {args.split}")
    lines.append(f"Showing top {min(args.top_n, len(ranked))} most-diverse shared targets")
    lines.append(sep)

    for rank, target in enumerate(ranked[:args.top_n], 1):
        lines.append(f"\n[{rank:03d}]  TARGET : {target}")
        lines.append(sep2)
        for lbl in labels:
            entries = target_index[target][lbl]
            for entry in entries:          # may be multiple samples per target per model
                wer = phoneme_wer(target, entry['predicted'])
                lines.append(
                    f"  [{lbl:<6}]  signer={entry['signer']:<8}  "
                    f"id={entry['id']:<20}  WER={wer*100:5.1f}%"
                )
                lines.append(f"           pred : {entry['predicted']}")
        lines.append("")

    # Summary statistics
    lines.append(sep)
    lines.append("SUMMARY STATISTICS (over shared targets, first sample per model per target)")
    lines.append(sep2)
    for lbl in labels:
        wers = []
        for target in shared_targets:
            if lbl in target_index[target] and target_index[target][lbl]:
                pred = target_index[target][lbl][0]['predicted']
                wers.append(phoneme_wer(target, pred) * 100)
        if wers:
            avg = sum(wers) / len(wers)
            lines.append(f"  [{lbl:<6}]  mean WER on shared targets = {avg:.2f}%  (n={len(wers)})")
    lines.append(sep)

    output = "\n".join(lines)
    print("\n" + output)

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output + "\n")
        print(f"\nSaved to: {args.output_file}")


if __name__ == '__main__':
    main()
