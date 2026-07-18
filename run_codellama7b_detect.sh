#!/usr/bin/env bash
# =====================================================================
# DetectCodeGPT paper reproduction: CodeLlama-7b-hf on CodeSearchNet
#
#   Stage 1  code-generation/generate.py  -> 10000 machine samples, tp0.2
#   Stage 2  code-detection/main.py       -> NPR/logrank/LRR/DetectGPT AUCs
#            (args_dict in main.py is already pinned to the matching config:
#             dataset=CodeSearchNet, dataset_key=CodeLlama-7b-hf-10000-tp0.2,
#             base_model=codellama/CodeLlama-7b-hf, n_samples=500,
#             n_perturbation_list=50, perturb_type=random-insert-space+newline)
#
# Usage:   bash run_codellama7b.sh [GPU_ID]        # default GPU 0
# Needs:   one GPU with >= 16 GiB free VRAM
#          (CodeLlama-7b fp16 ~13.5 GiB + codet5p-770m ~1.6 GiB + activations)
# =====================================================================
set -euo pipefail

PYTHON=/mnt/workspace/hkl/miniconda3/envs/de/bin/python
REPO=/mnt/cpfs/hkl/repeat/DetectCodeGPT

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="7"   # main.py uses os.environ.setdefault -> respects this

# ---------------- pinned paper configuration ----------------
DATASET=CodeSearchNet
MODEL=codellama/CodeLlama-7b-hf
MAX_NUM=300
TEMPERATURE=0.2
DATASET_KEY="CodeLlama-7b-hf-${MAX_NUM}-tp${TEMPERATURE}"
OUT_DIR="$REPO/code-generation/output/$DATASET/$DATASET_KEY"

# ---------------- stage 1: sample generation ----------------
if [ -s "$OUT_DIR/outputs.txt" ]; then
  echo "[skip] stage 1: $OUT_DIR/outputs.txt already exists"
else
  echo "[stage 1] generating $MAX_NUM samples with $MODEL on GPU $CUDA_VISIBLE_DEVICES (hours) ..."
  cd "$REPO/code-generation"
  "$PYTHON" generate.py \
    --path "data/$DATASET" \
    --model_name "$MODEL" \
    --max_num "$MAX_NUM" \
    --temperature "$TEMPERATURE"
fi

# ---------------- stage 2: zero-shot detection ----------------
echo "[stage 2] running detection on $DATASET/$DATASET_KEY ..."
cd "$REPO/code-detection"
"$PYTHON" main.py

echo "[done] ROC AUCs printed above; distribution plots in $REPO/code-detection/results.pdf"
