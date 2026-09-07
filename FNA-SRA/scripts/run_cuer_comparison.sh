#!/bin/bash
# 对比 WT 和 LF 两个 cuer 的模型预测差异

cd /home/uic2/fengling/mccsd/FNA-SRA-main

# 指定使用 GPU 0（避免 device 编号冲突）
export CUDA_VISIBLE_DEVICES=0

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
    --batch_size 4 \
    --num_workers 4 \
    --device cuda \
    --output_file results/cuer_comparison_WT_LF.txt
