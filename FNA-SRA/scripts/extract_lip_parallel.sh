#!/bin/bash
# Parallel lip feature extraction across 4 GPUs
# cuda:0 → F002, F003   (F002 899/1010, F003 187/1010 — resume)
# cuda:1 → M001, M002   (M001 775/1010 — resume)
# cuda:2 → M003, M004
# cuda:3 → M005
#
# Uses nohup + setsid so each worker survives parent-shell termination.

VIDEO_ROOT=/home/uic2/mhi-mccsd
SAVE_DIR=/home/uic2/mhi-mccsd/Features
LOG_DIR=/home/uic2/mhi-mccsd/logs
SCRIPT=$(cd "$(dirname "$0")" && pwd)/vit_extract_lip_feature.py
CONDA_PYTHON=/home/uic2/miniconda3/envs/mccsd/bin/python

mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting parallel lip feature extraction"

nohup setsid "$CONDA_PYTHON" "$SCRIPT" \
    --video_root "$VIDEO_ROOT" \
    --save_dir   "$SAVE_DIR" \
    --signers F002 F003 \
    --device cuda:0 \
    > "$LOG_DIR/lip_cuda0_F002_F003.log" 2>&1 &
PID0=$!

nohup setsid "$CONDA_PYTHON" "$SCRIPT" \
    --video_root "$VIDEO_ROOT" \
    --save_dir   "$SAVE_DIR" \
    --signers M001 M002 \
    --device cuda:1 \
    > "$LOG_DIR/lip_cuda1_M001_M002.log" 2>&1 &
PID1=$!

nohup setsid "$CONDA_PYTHON" "$SCRIPT" \
    --video_root "$VIDEO_ROOT" \
    --save_dir   "$SAVE_DIR" \
    --signers M003 M004 \
    --device cuda:2 \
    > "$LOG_DIR/lip_cuda2_M003_M004.log" 2>&1 &
PID2=$!

nohup setsid "$CONDA_PYTHON" "$SCRIPT" \
    --video_root "$VIDEO_ROOT" \
    --save_dir   "$SAVE_DIR" \
    --signers M005 \
    --device cuda:3 \
    > "$LOG_DIR/lip_cuda3_M005.log" 2>&1 &
PID3=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched (nohup+setsid):"
echo "  cuda:0  F002 F003  PID=$PID0  log=lip_cuda0_F002_F003.log"
echo "  cuda:1  M001 M002  PID=$PID1  log=lip_cuda1_M001_M002.log"
echo "  cuda:2  M003 M004  PID=$PID2  log=lip_cuda2_M003_M004.log"
echo "  cuda:3  M005       PID=$PID3  log=lip_cuda3_M005.log"
echo "Log dir: $LOG_DIR"

echo "$PID0 $PID1 $PID2 $PID3" > "$LOG_DIR/lip_pids.txt"
echo "PIDs saved to $LOG_DIR/lip_pids.txt"
