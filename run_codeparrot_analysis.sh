#!/usr/bin/env bash
# codeparrot-small empirical study: run the 3 runnable analysis scripts.
# analyze_proportion.py is skipped (token_tagging.py uses the old tree-sitter API).
# Usage: bash run_codeparrot_analysis.sh [GPU_ID]   (default 0)
# Prereq: run run_codeparrot_detect.sh first (needs the generated samples).
set -euo pipefail

GPU="${1:-0}"
PYTHON=/mnt/workspace/hkl/miniconda3/envs/de/bin/python
REPO=/mnt/cpfs/hkl/repeat/DetectCodeGPT
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$REPO/code-analysis"
mkdir -p figures

"$PYTHON" analyze_length.py
"$PYTHON" analyze_law_and_frequency.py
"$PYTHON" analyze_naturalness.py
# skipped: analyze_proportion.py (needs token_tagging port)
