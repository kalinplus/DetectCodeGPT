#!/usr/bin/env bash
# =====================================================================
# CodeLlama-7b-hf empirical study: analyze its generated code with the
# 4 code-analysis/ scripts (length, Zipf/Heaps laws, naturalness, token
# category proportions).
#
#   analyze_length.py            -> token/line count distributions
#   analyze_law_and_frequency.py -> Zipf + Heaps laws (CPU, tree-sitter)
#   analyze_naturalness.py       -> per-sample log-likelihood/log-rank
#                                  distributions + AUC  (loads CodeLlama-7B)
#   analyze_proportion.py        -> token-category proportions (CPU)
#
# NOTE: each .py file's target data/scorer is hardcoded in the file
#       (no CLI args). All four are pinned to CodeLlama-7b-hf-100-tp0.2
#       via the switch-model edits; re-pin them before switching back to
#       codeparrot. analyze_proportion uses the codeparrot tokenizer on
#       purpose (the HF-token<->tree-sitter alignment only handles GPT-2
#       BPE markers Ġ/Ċ, not SentencePiece's ▁).
#
# Usage:   bash run_codellama7b_analysis.sh [GPU_ID]   (default 0)
# Needs:   - generated samples at code-generation/output/CodeSearchNet/
#            CodeLlama-7b-hf-100-tp0.2/outputs.txt
#            (run run_codellama7b.sh stage 1, or generate.py, if missing)
#          - one GPU with >= 14 GiB free VRAM for analyze_naturalness.py
#            (CodeLlama-7B fp16 ~13 GiB). The other 3 scripts are CPU-only.
# =====================================================================
set -euo pipefail

GPU="${1:-0}"
PYTHON=/mnt/workspace/hkl/miniconda3/envs/de/bin/python
REPO=/mnt/cpfs/hkl/repeat/DetectCodeGPT
DATASET=CodeSearchNet
DATASET_KEY=CodeLlama-7b-hf-100-tp0.2

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

OUT="$REPO/code-generation/output/$DATASET/$DATASET_KEY/outputs.txt"
if [ ! -s "$OUT" ]; then
  echo "[error] missing generated samples: $OUT"
  echo "        run 'bash run_codellama7b.sh' (stage 1) or generate.py first."
  exit 1
fi

cd "$REPO/code-analysis"
mkdir -p figures

echo "[1/4] analyze_length.py"
"$PYTHON" analyze_length.py

echo "[2/4] analyze_law_and_frequency.py (CPU)"
"$PYTHON" analyze_law_and_frequency.py

echo "[3/4] analyze_naturalness.py (loads CodeLlama-7B on GPU $GPU)"
"$PYTHON" analyze_naturalness.py

echo "[4/4] analyze_proportion.py (CPU)"
"$PYTHON" analyze_proportion.py

echo "[done] figures/tables written under $REPO/code-analysis/figures/"
