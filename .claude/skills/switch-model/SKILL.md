---
name: switch-model
description: Use when switching this DetectCodeGPT repo's generation/scoring model to a different HF causal LM (e.g. codeparrot-small demo -> CodeLlama-7b-hf, CodeGen, SantaCoder, Phi-1). Triggers - changing base_model_name or dataset_key, pointing the pipeline at a new model, reproducing the paper with a different generator, "switch the model to X".
---

# Switch Model

Switch the model that is **both the sample generator and the zero-shot scoring model** (DetectGPT requires them to be the same) across generation, detection, analysis, and the run scripts. This skill does **not** touch the mask-filling model (`codet5p-770m`).

## Confirm inputs first

If not given, ask the user for:
- `MODEL` — full HF id, e.g. `codellama/CodeLlama-7b-hf`
- `MAX_NUM` — sample count (`1000` demo / `10000` paper)
- `TEMPERATURE` — usually `0.2`
- `DATASET` — usually `CodeSearchNet`
- GPU id to run on

## Derive (write these down — they recur everywhere)

- `MODEL_SHORT = MODEL.rsplit('/', 1)[-1]`  → e.g. `CodeLlama-7b-hf`
- `DATASET_KEY = f"{MODEL_SHORT}-{MAX_NUM}-tp{TEMPERATURE}"`  → e.g. `CodeLlama-7b-hf-10000-tp0.2`

`DATASET_KEY` MUST exactly equal the generation output folder name (it is built in `code-generation/generate.py` from `model_name.split('/')[-1]` + max_num + tp). A mismatch silently points everything at the wrong/missing data.

## Edit every point (set the KEY's value — do not pattern-match the old model name)

| File | Set this | To |
|------|----------|----|
| `code-detection/main.py` → `args_dict` | `dataset_key` | `DATASET_KEY` |
| `code-detection/main.py` → `args_dict` | `base_model_name` | `MODEL` |
| `code-analysis/analyze_length.py` (module bottom) | `tokenizer_name` | `MODEL` |
| `code-analysis/analyze_length.py` (module bottom) | `dataset_key` | `DATASET_KEY` |
| `code-analysis/analyze_law_and_frequency.py` (`__main__`, 2 calls) | `load_data(..., key=...)` | `DATASET_KEY` |
| `code-analysis/analyze_naturalness.py` → `args_dict` | `dataset_key` | `DATASET_KEY` |
| `code-analysis/analyze_naturalness.py` → `args_dict` | `base_model_name` | `MODEL` |
| `run_codeparrot_detect.sh` (config block) | `MODEL`, `MAX_NUM`, `TEMPERATURE` | new values (`DATASET_KEY`/`OUT` are derived from these via `${MODEL##*/}`) |
| `run_codeparrot_analysis.sh` | — | generic; no per-model edit. The dataset_key lives inside the `.py` files (rows above). |

Notes on the table:
- `analyze_length.py` also has a commented `tp1.0` block — leave it commented unless a tp1.0 set was also generated.
- `analyze_proportion.py` is intentionally **skipped**: it does `from token_tagging import *`, and `token_tagging.py` still uses the removed tree-sitter API. Only touch it if the user also wants `token_tagging.py` ported (mirror the already-ported `code-detection/identifier_tagging.py`).
- CUDA is already `os.environ.setdefault(...)` in `main.py`, `analyze_length.py`, and `analyze_naturalness.py`, so a shell-exported `CUDA_VISIBLE_DEVICES` controls the GPU. `analyze_law_and_frequency.py` has no CUDA line (pure CPU frequency counting). Do not hardcode CUDA back anywhere.

## After editing

1. **Generation** (no code change; produces `code-generation/output/<DATASET>/<DATASET_KEY>/outputs.txt`):
   ```
   cd code-generation && python generate.py --path data/<DATASET> \
       --model_name <MODEL> --max_num <MAX_NUM> --temperature <TEMPERATURE>
   ```
   Skip if `outputs.txt` already exists.
2. **Verify**: `python -m py_compile` the 4 edited `.py` files; `bash -n` the two `.sh` scripts.
3. **Sanity**: `ls code-generation/output/<DATASET>/` and confirm a folder named exactly `DATASET_KEY` exists.
4. Run: `bash run_codeparrot_detect.sh <GPU>` then optionally `bash run_codeparrot_analysis.sh <GPU>`.

## Common mistakes

- **base_model ≠ generator** → DetectGPT scores are meaningless. `base_model_name` and the `generate.py --model_name` MUST be the same HF id.
- **Wrong `DATASET_KEY`** → detection/analysis read the wrong `outputs.txt`. Re-derive it from the rules above; don't copy an old one.
- **New model family not handled in `generate_hf`** → `code-generation/generate.py` has per-family tokenizer pad-token branches (codegen/santa/parrot/incoder/phi-1/llama/wizard) and per-family model-load branches (t5p/llama/codegen2). If `MODEL` matches none, add a branch or generation may pad incorrectly / load as wrong architecture.
- **No VRAM pre-check in the scripts** — they just run (per repo `CLAUDE.md`: try once or twice; if OOM, report or pick another GPU — do not loop to grab memory). For ≥7B models set `half`/`base_half`/`int8` in `main.py` `args_dict` so the model fits.
