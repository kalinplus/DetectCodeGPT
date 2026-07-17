#!/usr/bin/env bash
# codeparrot-small demo: generate samples (if missing) + zero-shot detection.
# Usage: bash run_codeparrot_detect.sh [GPU_ID]   (default 0)
set -euo pipefail

GPU="${1:-0}"
PYTHON=/mnt/workspace/hkl/miniconda3/envs/de/bin/python
REPO=/mnt/cpfs/hkl/repeat/DetectCodeGPT
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="$GPU"

DATASET=CodeSearchNet
MODEL=codeparrot/codeparrot-small
MAX_NUM=1000
TEMPERATURE=0.2
DATASET_KEY="${MODEL##*/}-${MAX_NUM}-tp${TEMPERATURE}"   # must match generate.py output folder
OUT="$REPO/code-generation/output/$DATASET/$DATASET_KEY/outputs.txt"

# stage 1: generate (skip if already done)
if [ ! -s "$OUT" ]; then
  cd "$REPO/code-generation"
  "$PYTHON" generate.py --path "data/$DATASET" --model_name "$MODEL" \
      --max_num "$MAX_NUM" --temperature "$TEMPERATURE"
fi

# stage 2: detection -> ROC AUCs + results.pdf
cd "$REPO/code-detection"
"$PYTHON" main.py
