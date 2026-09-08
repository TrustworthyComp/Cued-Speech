# FNA-CSR: Beyond Visual Encoder Fine-tuning for Cued Speech Recognition

<!-- badges -->
[![Paper](https://img.shields.io/badge/Paper-IEEE%20ICDM%202026-2E6B8B)]()
[![Code](https://img.shields.io/badge/Code-%20Repository-9C27B0)](https://github.com/TrustworthyComp/Cued-Speech)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**FNA-CSR** (Freeze, Normalize, and Align for Cued Speech Recognition) is a fine-tuning-free multimodal pattern mining framework for **Mandarin Chinese Cued Speech Recognition (ACSR)**. It keeps the visual encoders fully frozen, calibrates cross-cuer feature distributions in feature space, globally aligns the visual and phoneme modalities, and lets a lightweight LoRA-adapted LLM (Flan-T5-XL) decode phoneme sequences.

> **FNA-CSR: Beyond Visual Encoder Fine-tuning: A Generalizable Data Mining Paradigm for Multimodal Chinese Cued Speech Recognition**
> *Under review at IEEE ICDM 2026*

<p align="center">
  <img src="images/framework-1.png" width="100%" alt="FNA-CSR framework overview"/>
</p>

## Why FNA-CSR?

Automatic Cued Speech Recognition suffers from two intertwined data challenges:

1. **Cross-subject distribution shift** — different cuers exhibit distinct physiological traits and gestural styles; fine-tuning memorizes cuer-specific noise instead of shared semantics, and fails catastrophically on *unseen* cuers.
2. **Cross-modal temporal asynchrony** — hand gestures precede lip articulation by a few milliseconds, violating the frame-wise alignment assumption of most fusion architectures.

FNA-CSR reformulates ACSR as a **feature distribution optimization problem** instead of an encoder adaptation problem. With fully frozen visual encoders, it:

- **CSSP (Cross-Subject Semantic Processor)** unifies feature distributions across cuers via *instance distribution normalization* and *hard N-tuplet contrastive clustering*;
- **VP-Align (Visual–Phoneme Embedding Alignment)** eliminates cross-modal timing mismatch via *global contrastive alignment* in the shared LLM embedding space, bypassing frame-wise alignment;
- a **frozen Flan-T5-XL adapted with LoRA** (r=16, α=32) generates phoneme sequences from aligned visual features.

The paradigm reduces trainable parameters by **99.2%** (~25.6M vs. ~3,154M for full fine-tuning) while achieving SOTA generalization.

## Key Results

### Main Comparison (CER ↓ / WER ↓, %)

| Method                    | MCCSD (1-H) | | MCCSD (1-HI) | | MCCSD (6-H) | | MHI-MCCSD (8-HI) | |
| ------------------------- | :---------: | :---: | :----------: | :---: | :---------: | :---: | :--------------: | :---: |
|                           | CER  | WER | CER   | WER | CER  | WER | CER   | WER |
| ResNet18 + MHSA           | 26.19 | 61.87 | 66.70  | 98.67 | 61.83 | 94.34 | 82.03  | 99.91 |
| CMML                      | 9.81  | 25.54 | 32.23  | 69.45 | 30.01 | 68.12 | 51.80  | 91.26 |
| EcoCued                   | 9.54  | 25.03 | 29.56  | 61.59 | 29.75 | 67.83 | 50.52  | 90.13 |
| STF-ACSR                  | **1.82** | **5.19** | **4.62** | **12.21** | 8.35 | 21.06 | 10.96  | 25.67 |
| Cued-Agent                | 2.61  | 6.56  | 6.72   | 16.23 | 9.05  | 20.54 | 12.67  | 29.86 |
| **FNA-CSR (ours)**        | 4.27  | 8.51  | 9.46   | 16.11 | **2.72** | **4.74** | **10.81** | **19.26** |

### Leave-One-Cuer-Out (LOCO) Generalization (CER ↓ / WER ↓, %)

| Method       | MCCSD (4-H) CER | WER | MCCSD (6-H) CER | WER | MHI-MCCSD (8-HI) CER | WER |
| ------------ | :-------------: | :---: | :-------------: | :---: | :-----------------: | :---: |
| CMML         | 63.4 | 92.5 | — | — | — | — |
| EcoCued      | 57.8 | 84.1 | — | — | — | — |
| ISCL-ISCA    | 34.7 | 59.4 | — | — | — | — |
| FedCSR       | 52.3 | 76.8 | — | — | — | — |
| **FNA-CSR**  | **21.2** | **33.3** | **5.3** | **11.1** | **17.29±5.56** | **25.24±6.27** |

On MCCSD (6-H), the LOCO result (11.1% WER) closely approaches the in-distribution result (4.74% WER), demonstrating that CSSP effectively closes the OOD gap. MMD-based distribution analysis further shows a strong correlation between feature-space convergence and LOCO generalization. On MHI-MCCSD (8-HI), the LOCO result is the mean±std over eight leave-one-cuer-out folds; the frozen-backbone baseline attains 14.14±8.93% CER / 20.36±12.29% WER under identical splits (per-fold breakdown in the paper).

### Cued Speech and Feature Distribution Visualizations

<p align="center">
  <img src="images/MCCSD_3.png" width="38%" alt="Mandarin Chinese CS system"/>
  <img src="images/hebing4-2.png" width="52%" alt="UMAP visualization of sentence embeddings"/>
</p>

*Left:* the Mandarin Chinese CS system — five hand positions encode vowel groups and eight hand shapes distinguish consonant phonemes, augmenting lip movements for hearing-impaired communication. *Right:* UMAP visualization of sentence embeddings on MCCSD (6-H). Raw frozen CLIP features cluster by **cuer identity** (a), while CSSP+VP-Align transforms them into clusters organized by **phoneme content** (b) — visual evidence that cuer-clustered features become semantic-driven after distribution-level optimization.

## Paper ↔ Code Mapping

| Paper module                              | Code implementation                                                                                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Feature Encoder (frozen CLIP ViT-L/14)    | `scripts/vit_extract_lip_feature.py`, `scripts/vit_extract_hand_feature.py`, `scripts/vit_extract_global_feature.py`                                 |
| S² multi-scale wrapping (224² + 448²)     | `utils/s2wrapper.py`                                                                                                                                 |
| CSSP: Instance Distribution Normalization | `SignerNorm` in `fna_sra/t5_sra.py`                                                                                                                  |
| CSSP: Temporal Convolution Encoder        | `fna_sra/tconv.py` (`TemporalConv`)                                                                                                                  |
| CSSP: Hard N-tuplet Contrastive Clustering| `compute_triplet_loss` in `fna_sra/t5_sra.py` (multi-positive hard triplet)                                                                          |
| Semantic–Subject Batch Sampler            | `GKSampler` in `dataset/sampler.py` (G glosses × K signers per batch, DDP-aware)                                                                     |
| VP-Align: Global Contrastive Alignment    | `visual_textual_align` in `fna_sra/t5_sra.py` + `fna_sra/clip_loss.py`                                                                               |
| Vision→Text Projector                     | `fna_sra/mm_projector.py`                                                                                                                            |
| LLM Decoder (Flan-T5-XL + LoRA)           | `FlanT5SLT` in `fna_sra/t5_sra.py`; CTC and Transformer decoders in `fna_sra/ctc_align.py`, `fna_sra/transformer_align.py`                           |
| Distribution analysis (MMD / JS)          | `scripts/compute_mmd_js.py`, `scripts/compute_mmd_cer_correlation.py`                                                                                |
| Phoneme → Pinyin conversion               | `scripts/phoneme_to_pinyin.py`, `scripts/phoneme_to_pinyin_robust.py`, `scripts/phoneme_to_pinyin_v2.py`, `scripts/phoneme_to_pinyin_v3.py`          |

## Two-Stage Training

FNA-CSR follows the two-stage schedule described in the paper:

- **Stage 1 — Warm-up (4K steps):** only CSSP and VP-Align are trained with the N-tuplet and CLIP contrastive losses while the T5 decoder stays frozen. This organizes the frozen feature space into a cuer-invariant, semantically clustered structure before generation begins.
- **Stage 2 — Joint training:** CSSP, the fusion projector, VP-Align, and T5 LoRA adapters are jointly optimized with the combined objective `L_total = L_ce + α·L_clip + λ·L_N-tuplet` (α = 1.0, λ = 0.1).

In the codebase, the warm-up length is controlled by `warm_up_steps` in the config; during `global_step <= warm_up_steps`, `FlanT5SLT.shared_step` optimizes the contrastive alignment loss only. Inference uses beam search with all encoders frozen.

## Directory Structure

```
FNA-SRA/
├── fna_sra/                  # Core framework package
│   ├── t5_sra.py             # FlanT5SLT: LoRA-adapted Flan-T5 + SignerNorm + VP-Align + N-tuplet
│   ├── asb.py                # AbstractSLT base class
│   ├── tconv.py              # Temporal convolution encoder
│   ├── mm_projector.py       # Vision-to-text projector (768 → 2048)
│   ├── clip_loss.py          # CLIP-style global contrastive loss (VP-Align)
│   ├── ctc_align.py          # CTC decoder / alignment
│   ├── transformer_align.py  # Transformer decoder / alignment
│   ├── decoder_utils.py      # Phoneme tokenizer & vocabulary
│   ├── callbacks.py          # Training callbacks
│   ├── lr_scheduler.py       # Learning rate scheduler
│   └── constants.py          # Prompt constants
├── dataset/                  # Dataset modules
│   ├── datamodule.py         # Lightning DataModule
│   ├── mccsd.py              # MCCSD / MHI-MCCSD dataset loader
│   ├── p14t.py               # Phoenix14T dataset loader
│   └── sampler.py            # GKSampler (Semantic-Subject batch sampler, DDP-aware)
├── utils/                    # Utility functions
│   ├── helpers.py            # Config instantiation, mask utilities
│   ├── evaluate.py           # Evaluation metrics (CER/WER)
│   ├── evaluate_dict.py      # Dictionary-based WER
│   └── s2wrapper.py          # S² multi-scale wrapping
├── scripts/                  # Feature extraction & analysis scripts
│   ├── vit_extract_lip_feature.py          # CLIP ViT lip feature extraction
│   ├── vit_extract_hand_feature.py         # CLIP ViT hand feature extraction
│   ├── vit_extract_global_feature.py       # Global body-pose feature extraction
│   ├── vit_extract_lip_feature_finetuned.py# Finetuned ViT lip extraction
│   ├── conformer_extract_lip_feature.py    # Conformer lip feature extraction
│   ├── mae_extract_feature.py              # MAE visual feature extraction
│   ├── vit_finetune_lip_ctc_attn.py        # ViT lip finetuning with CTC+Attn
│   ├── compute_mmd_js.py                   # Cross-cuer MMD / JS divergence
│   ├── compute_mmd_cer_correlation.py      # MMD–CER correlation analysis
│   ├── compare_cuer_predictions.py         # Per-cuer prediction comparison
│   ├── visualize_crops.py                  # ROI crop visualization
│   └── phoneme_to_pinyin*.py               # Phoneme → Pinyin conversion
├── preprocess/               # Data preprocessing
│   ├── MCCSD/                # MCCSD annotation builders (make_info*.py)
│   └── MHI_MCCSD/            # MHI-MCCSD annotation builders
├── configs/                  # Training configuration
│   └── finetune_mccsd_6H_all.yaml
└── main.py                   # Training / evaluation entry point
```

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n fna-csr python=3.11 -y
conda activate fna-csr
```

### 2. Install Dependencies

```bash
conda activate fna-csr

# PyTorch (choose the command matching your CUDA version)
pip install torch torchvision torchaudio

# All other dependencies
pip install -r requirements.txt

# Optional: logging and monitoring
pip install wandb tensorboard

# HuggingFace mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. Download Pretrained Models

The framework uses `google/flan-t5-xl` as the LLM decoder and CLIP ViT-L/14 as the visual encoder (via HuggingFace), both downloaded automatically on first run. LoRA keeps the trainable footprint at ~25.6M parameters.

## Data Preparation

### MCCSD 6-Speaker Dataset (6H)

The dataset contains 6 speakers (LF, HS, WT, XP, YX, YZ) with ~1000 video samples each. Videos should be organized as:

```
mccsd_datasets/
└── RawVideo/
    ├── LF/    # Speaker folders with .mp4 videos
    ├── HS/
    ├── WT/
    ├── XP/
    └── ...
mccsd_sub/
├── YX/
└── YZ/
```

### Step 1: Build Annotation Files

```bash
conda activate fna-csr
cd preprocess/MCCSD

# Generate train/test info files
python make_info_6H.py \
    --data_root /path/to/mccsd_sub \
    --alt_roots /path/to/mccsd_datasets/RawVideo \
    --train_txt /path/to/multi_speaker_train_labels_6H.txt \
    --test_txt /path/to/multi_speaker_test_labels_6H.txt \
    --save_dir /path/to/save_dir
```

This generates `train_info_ml.npy` and `test_info_ml.npy` under the save directory.

> **MHI-MCCSD (8-HI)** — the hearing-impaired benchmark with 8 cuers is also supported: annotation builders live in `preprocess/MHI_MCCSD/` and share the same `dataset/mccsd.py` loader. Follow the same pipeline with the corresponding `make_info*.py` scripts.

### Step 2: Extract Visual Features

Extract CLIP ViT-L/14 features for lip and hand regions (and optionally global body-pose):

```bash
conda activate fna-csr
cd FNA-SRA

# Lip features (per-frame mouth crops)
python scripts/vit_extract_lip_feature.py \
    --video_root /path/to/mccsd_datasets/RawVideo \
    --save_dir /path/to/mccsd_datasets/Lip_Features \
    --device cuda:0 \
    --signers LF HS WT XP \
    --force

python scripts/vit_extract_lip_feature.py \
    --video_root /path/to/mccsd_sub \
    --save_dir /path/to/mccsd_datasets/Lip_Features \
    --device cuda:0 \
    --signers YX YZ \
    --force

# Hand features (left hand per-frame crops)
python scripts/vit_extract_hand_feature.py \
    --video_root /path/to/mccsd_datasets/RawVideo \
    --save_dir /path/to/mccsd_datasets/Hand_Features \
    --device cuda:0 \
    --signers LF HS WT XP \
    --left_only \
    --force

python scripts/vit_extract_hand_feature.py \
    --video_root /path/to/mccsd_sub \
    --save_dir /path/to/mccsd_datasets/Hand_Features \
    --device cuda:0 \
    --signers YX YZ \
    --left_only \
    --force
```

Expected output structure:

```
mccsd_datasets/
├── Lip_Features/clip-vit-large-patch14_lip_feat_mccsd/{speaker}/{video_id}.npy
└── Hand_Features/clip-vit-large-patch14_hand_feat_mccsd/{speaker}/{video_id}_left.npy
```

## Training

### Update Configuration

Edit [configs/finetune_mccsd_6H_all.yaml](FNA-SRA/configs/finetune_mccsd_6H_all.yaml) to set your local paths:

```yaml
data:
  params:
    train:
      params:
        anno_root: ./preprocess/MCCSD_6H
        hand_feat_root: /path/to/mccsd_datasets/Hand_Features/clip-vit-large-patch14_hand_feat_mccsd
        lip_feat_root: /path/to/mccsd_datasets/Lip_Features/clip-vit-large-patch14_lip_feat_mccsd
```

### Start Training

```bash
conda activate fna-csr
cd FNA-SRA

# Training with Character Error Rate (CER) as evaluation metric
CUDA_VISIBLE_DEVICES=0 python main.py \
    -c configs/finetune_mccsd_6H_all.yaml \
    -e cer \
    -n mccsd_6H_all
```

### Key Training Arguments

| Argument             | Description                                | Default    |
| -------------------- | ------------------------------------------ | ---------- |
| `-c, --config`       | Configuration file(s) to load              | (required) |
| `-e, --evaluation`   | Metric: `cer`, `wer`, `per`, `bleu`, `mse` | `mse`      |
| `-n, --name`         | Run name postfix for log directory         | (auto)     |
| `-s, --seed`         | Random seed                                | `0`        |
| `-l, --logdir`       | Base log directory                         | `logs`     |
| `-f, --fast_dev_run` | Debug mode (1 batch only)                  | `false`    |
| `-r, --resume`       | Resume from checkpoint directory           | `None`     |
| `--ckpt`             | Specific checkpoint file                   | `None`     |
| `--test`             | Run in test-only mode                      | `false`    |
| `--no_test`          | Skip test phase after training             | `true`     |

### Monitoring Training

```bash
# View live training log
tail -f logs/202X-XX-XX_mccsd_6H_all/train.log

# TensorBoard
tensorboard --logdir logs/
```

## Pretrained Model

| Dataset     | CER  | WER  | Download                                                                                              |
| ----------- | ---- | ---- | ----------------------------------------------------------------------------------------------------- |
| MCCSD (6-H) | 2.72 | 4.74 | [Google Drive](https://drive.google.com/file/d/18dviSs0XFIxYwppNO1nnfz-IJT47a-VC/view?usp=drive_link) |

## Evaluation

After training completes, the best checkpoint is automatically used for testing. Results (CER/WER) are printed at the end.

Manual evaluation:

```bash
conda activate fna-csr
cd FNA-SRA
python main.py \
    -c configs/finetune_mccsd_6H_all.yaml \
    --test \
    --ckpt logs/<run_name>/checkpoints/<best>.ckpt
```

## Citation

```bibtex
@article{feng2026fnacsr,
  title   = {Beyond Visual Encoder Fine-tuning: A Generalizable Data Mining
             Paradigm for Multimodal Chinese Cued Speech Recognition},
  author  = {Ling Feng and Bu Zhong and Jianglun Wu and Liu Li and
             Francis Chi Moon Lau and Donglong Chen and Yupeng Li},
  journal = {Under review at IEEE International Conference on Data Mining (ICDM)},
  year    = {2026}
}
```
