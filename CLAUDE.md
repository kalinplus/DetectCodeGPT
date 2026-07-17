# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official code for the ICSE 2025 paper "Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers" (DetectCodeGPT). It detects machine-generated code **zero-shot**, using only a scoring LLM — no training. Derived from DetectGPT and DetectLLM.

## Pipeline architecture (three stages, three top-level dirs)

Data flows strictly in this order, via files on disk:

1. **`code-generation/generate.py`** — Builds the paired dataset. Loads human code (CodeSearchNet `data/CodeSearchNet/python/train.jsonl` or TheVault `data/TheVault/python/small_train.jsonl`), splits `original_string` on `"""` into docstring-prompt vs. solution body, length-filters, then samples completions from an HF causal LM (CodeLlama, CodeGen, SantaCoder, Phi-1, Incoder, T5+...). Writes JSONL `{prompt, output, solution}` to `code-generation/output/<Dataset>/<model>-<N>-tp<temp>/outputs.txt` — this path scheme is hardcoded downstream.

2. **`code-analysis/`** — The paper's empirical study (skippable for detection): `analyze_length.py`, `analyze_law_and_frequency.py` (Zipf/Heaps), `analyze_proportion.py` (token categories; uses `token_tagging.py` + tree-sitter), `analyze_naturalness.py`. All read `../code-generation/output/<dataset>/<key>/outputs.txt`.

3. **`code-detection/main.py`** — The detector. Computes per-sample log-likelihood and log-rank under the scoring model for originals, machine samples, and N perturbed copies of each, then evaluates ROC AUC for four criteria: logrank, LRR (ll/logrank), DetectGPT z-score, and NPR (perturbed_logrank / original_logrank — the paper's method). Writes `results.pdf`.

**Two model roles:** the *base/scoring model* must be the same model that generated the samples (e.g. `codellama/CodeLlama-7b-hf`); the *mask-filling model* (default `Salesforce/codet5p-770m`) is only used when `perturb_type` involves masking.

**Detection's core idea (the `perturb_type` arg):** the paper's contribution is *whitespace perturbation* — `random-insert-space` / `random-insert-newline` / `random-insert-space+newline` (the default) corrupt code formatting without needing a mask model at all. Use `perturb_type: "random"` to reproduce original DetectGPT/DetectLLM-NPR (T5 mask-fill), or `identifier-masking` (tree-sitter-based).

## Commands

```bash
pip install -r requirements.txt

# Stage 1: generate samples (run from code-generation/)
cd code-generation && python generate.py --path data/CodeSearchNet \
    --model_name codellama/CodeLlama-7b-hf --max_num 10000 --temperature 0.2

# Stage 3: run detection (run from code-detection/)
cd code-detection && python main.py

# Stage 2: empirical study (run from code-analysis/)
cd code-analysis && python analyze_length.py  # etc.
```

There is no test suite or linter. `code-generation/testing/test_truncate.py` is a scratch script, not pytest.

## Gotchas

- **`main.py` is not CLI-configured.** `setup_args()` ignores the command line and parses a hardcoded `args_dict` — edit that dict (dataset, `dataset_key` = the generation output folder name, `base_model_name`, `n_samples`, `n_perturbation_list`, `perturb_type`) to change anything. The `analyze_*.py` scripts follow the same pattern.
- **`CUDA_VISIBLE_DEVICES = "0"` is hardcoded** at import time in `main.py` and the analysis scripts; `baselines/loss.py` additionally routes 13b/20b models to the *last* GPU.
- **Duplicated `baselines/` packages**: `code-analysis/baselines/` and `code-detection/baselines/` are near-copies that have drifted (`all_baselines.py`, `loss.py`, `rank.py` differ). Edit the one in the directory you're running.
- **tree-sitter API mismatch**: `identifier_tagging.py` / `token_tagging.py` use the old tree-sitter API (`Language.build_library`, `parser.set_language`, grammar sources expected under `./tree-sitter/tree-sitter-*` relative to cwd), but `requirements.txt` pins `tree-sitter==0.25.1`, which removed that API. These modules run at import time (`main.py` imports `identifier_tagging`), so expect to either downgrade tree-sitter + vendor the grammar repos, or port to the new API. **(Done for `code-detection/identifier_tagging.py`: ported to the new API — python grammar from `tree_sitter_python`, other languages optional. `code-analysis/token_tagging.py` is still unpatched.)**
- `get_ll`/`get_rank` contain `time.sleep(0.01)` per call — scoring loops are deliberately throttled.

## Compatibility fixes already applied (this machine, transformers v5 / tree-sitter 0.25)

The `de` env has transformers 5.13.1 + torch 2.10, which broke the original code in three places — all patched in `code-detection/` only (search for `compat`/`DEMO` comments):

1. `identifier_tagging.py` — ported to the new tree-sitter API (see above).
2. `baselines/utils/preprocessing.py` — transformers v5 no longer expands `~` in `cache_dir`; the literal path created a `./~` directory under cwd and failed model lookup with a misleading "Unrecognized model... model_type" error. Fixed with `os.path.expanduser`.
3. `baselines/utils/loadmodel.py` — v5 `RobertaTokenizer` crashes on codet5p's dict-form `additional_special_tokens` (`TypeError: Input must be a List[Union[str, AddedToken]]`); fixed by retrying with `additional_special_tokens` passed as 100 plain `<extra_id_i>` strings.

Datasets/models for the demo were fetched via `HF_ENDPOINT=https://hf-mirror.com` (GitHub raw is blocked): CodeSearchNet python train parquet → `code-generation/data/CodeSearchNet/python/train.jsonl` (first 100k rows, key renamed `whole_func_string`→`original_string`); models in the default HF cache.

## Demo status (verified 2026-07-17)

End-to-end pipeline runs on this machine: `generate.py` with `codeparrot/codeparrot-small` (1000 samples, GPU light) → `main.py` demo config gives AUCs ≈ logrank 0.98 / LRR 0.94 / NPR 0.975 and writes `code-detection/results.pdf`. Numbers are sanity checks only — a weak 110M generator is easy to detect.

For the paper-faithful run (CodeLlama-7b-hf, 10000 samples, tp0.2, n_samples=500, n_perturbation_list=50): `main.py`'s `args_dict` is already pinned to that config, and **`run_codellama7b.sh` at the repo root runs both stages** — `bash run_codellama7b.sh [GPU_ID]` (checks for ≥16 GiB free VRAM first; generation is per-sample and takes hours; stage 1 auto-skips if `outputs.txt` exists). Two supporting tweaks: `main.py` now uses `os.environ.setdefault("CUDA_VISIBLE_DEVICES", ...)` so the script can pick the GPU, and `generate.py` sets `pad_token_id=eos` for llama-family tokenizers (v5 `generate` requires a pad token). This full run has NOT been executed yet on this machine.
