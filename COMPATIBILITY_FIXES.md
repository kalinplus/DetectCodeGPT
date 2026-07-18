# 兼容性修复记录（问题 - 修复方式）

本机环境：`de` conda 环境，`transformers==5.13.1` + `torch==2.10` + `tree-sitter==0.25.1`（`requirements.txt` 所 pin）。
原仓库代码面向旧版本依赖，在本机跑会报错，以下是已修复并验证的问题清单。每条记录：**问题**（症状/报错）→ **修复方式**（改了哪个文件、改成什么样）。

> 原仓库仅 `code-detection/` 打过补丁（demo 用）；`code-analysis/` 的**同类**问题（tree-sitter + baselines 的 cache_dir / special-tokens）于 2026-07-17 一并修复。注意：`code-analysis/baselines/` 与 `code-detection/baselines/` 是近乎重复且已漂移的两套副本，改一个不会影响另一个。

---

## 1. tree-sitter `Language.build_library` 不存在（导入即崩溃）

**问题**
`tree-sitter==0.25.1` 移除了旧 API（`Language.build_library`、`Language(path, name)`、`parser.set_language`）。
模块在 import 时就调用这些 API，所以一启动就崩：
```
AttributeError: type object 'tree_sitter.Language' has no attribute 'build_library'
```
受影响文件（三个，问题代码块相同）：
- `code-detection/identifier_tagging.py`（`main.py` 导入它，crash 会阻断整个检测流程）
- `code-analysis/token_tagging.py`（被 `analyze_proportion.py`、`analyze_naturalness.py` 导入）
- `code-analysis/analyze_law_and_frequency.py`（自身含一份拷贝）

**修复方式**
改用“语法独立 pip 包 + 新 Language API”。删除原来的 `build/my-languages.so` 构建块，替换为：
```python
import tree_sitter_python

LANGUAGE_MAP = {
    'python': Language(tree_sitter_python.language()),
}
# 其他语言按需启用（装了对应包才生效）
for _lang, _module in [
    ('java', 'tree_sitter_java'), ('php', 'tree_sitter_php'),
    ('go', 'tree_sitter_go'), ('ruby', 'tree_sitter_ruby'),
    ('javascript', 'tree_sitter_javascript'),
]:
    try:
        _mod = __import__(_module)
        LANGUAGE_MAP[_lang] = Language(_mod.language())
    except ImportError:
        logger.info(f'{_module} not installed, {_lang} tagging unavailable')

parser = Parser()
```
并把 `get_tagged_tokens` / `get_identifier` 里：
```python
parser.set_language(LANGUAGE_MAP[lang])   # 旧
```
改为：
```python
parser.language = LANGUAGE_MAP[lang]      # 新
```
依赖：`pip install tree-sitter-python`（其它语言可选）。三个文件都已按此修复并验证（解析 `def f(x): return x+1`，叶子节点类型正确：`def/identifier/return/+/integer …`）。

---

## 2. transformers v5 不再展开 `cache_dir` 里的 `~`

**问题**
`code-detection/baselines/utils/preprocessing.py` 用 `cache_dir='~/.cache/...'`，v5 不再自动展开 `~`，于是在 **cwd 下建了一个 `./~` 目录**，模型查找失败，报一条误导性错误：
```
Unrecognized model class ... model_type
```
（看起来像模型不认识，其实是缓存路径错了。）

**修复方式**
用 `os.path.expanduser` 手动展开：
```python
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')   # 或原路径外裹 expanduser
```
搜索 `compat` / `DEMO` 注释可定位。

> **2026-07-17 同步**：`code-analysis/baselines/utils/preprocessing.py` 原本漏改。跑 `analyze_naturalness.py`（默认 `--cache_dir '~/.cache/huggingface/hub'`，配合 `--base_model_name codeparrot/codeparrot-small`）时复现同样的误导错误：
> ```
> ValueError: Unrecognized model in codeparrot/codeparrot-small. Should have a `model_type` key in its config.json.
> ```
> 但该模型 config 里其实就有 `"model_type": "gpt2"`——纯粹是 cache_dir 没展开导致查找失败（cwd 下会冒出一个 `~` 垃圾目录作为佐证）。已同样补上 `os.path.expanduser`，加载验证为 `GPT2LMHeadModel` + `GPT2Tokenizer`。

**为什么会冒出 `~` 目录（机制）**：只有 **shell** 做 tilde 展开；Python 里 `~` 就是个普通字符。`os.makedirs('~/.cache/huggingface/hub')` 把它当 cwd 下的相对路径，于是真的建了 `./~/.cache/huggingface/...`。transformers v4 内部会自行 `expanduser`，v5 去掉了这步，所以老代码在 v5 下才暴露。

**源头阻断（两层，都已做）**：
1. **漏斗层**：`preprocess_and_save` 是所有 cache_dir 的必经之路，开头 `os.path.expanduser` 在 `os.makedirs` 之前把 `~` 展开 → 不会再建 `~` 目录（已验证：传 `~/__tilde_block_test__` 落到 `$HOME/...` 而非 `./~/...`）。
2. **默认值层**：把 `analyze_naturalness.py` / `analyze_proportion.py` 里 4 处字面量 `"~/.cache/huggingface/hub"`（argparse `default=` + `args_dict`）改成 `os.path.expanduser("...")`，让 `~` 在解析阶段就消失，运行期拿到的永远是绝对路径。

---

## 3. `RobertaTokenizer` 拒绝 codet5p 的 dict 形式 `additional_special_tokens`

**问题**
默认 mask 模型 `Salesforce/codet5p-770m` 的 tokenizer 里，`additional_special_tokens` 是 **dict 形式**。v5 的 `RobertaTokenizer` 只接受 `List[Union[str, AddedToken]]`，加载时直接抛：
```
TypeError: Input must be a List[Union[str, AddedToken]]
```

**修复方式**
`code-detection/baselines/utils/loadmodel.py`：先按原方式加载，捕获异常后重试，把 `additional_special_tokens` 改成 100 个纯字符串 `<extra_id_0>` … `<extra_id_99>`：
```python
extra = [f'<extra_id_{i}>' for i in range(100)]
tokenizer = AutoTokenizer.from_pretrained(name, additional_special_tokens=extra, ...)
```
> **2026-07-17 同步**：`code-analysis/baselines/utils/loadmodel.py` 补上同一 try/except（`analyze_naturalness.py` 默认 mask 模型是 T5 系，启用 mask 填充时才会触发）。

---

## 4. `main.py` 硬编码 `CUDA_VISIBLE_DEVICES`

**问题**
`main.py`（及 `code-analysis/analyze_*.py`）在 **import 时**写死 `CUDA_VISIBLE_DEVICES = "0"`，导致无法从外部指定 GPU。`baselines/loss.py` 另有逻辑：13b/20b 模型路由到最后一张卡。

**修复方式**
`main.py` 改为 `setdefault`，允许运行前指定、否则回退到 `"0"`：
```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
```
（这样 `CUDA_VISIBLE_DEVICES=2 python main.py` 或在脚本里 export 都能生效。）
注意 `code-analysis/analyze_*.py` 仍是硬编码 `"0"`，如需换卡需手动改或同样改成 setdefault。

---

## 5. llama 家族 tokenizer `generate` 缺 pad token

**问题**
transformers v5 的 `generate` 要求有 pad token；llama 家族（CodeLlama 等）tokenizer **没有 pad_token**，生成时报错。

**修复方式**
`code-generation/generate.py` 中显式补上：
```python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

---

## 6. 环境层注意点（非代码 bug，但影响运行）

- **网络**：`HF_ENDPOINT=https://hf-mirror.com` 已设置，下载模型/数据集约 11 MB/s，可用；huggingface.co 也能直连。但 **`raw.githubusercontent.com` 被墙/超时**——需要 GitHub raw 资源时改用 HF 托管的等价物。
- **HF 下载方式**：`de` 环境里的 `huggingface-cli download` 只打印帮助（新 CLI 行为变了），改用 Python：
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id=..., repo_type='model'/ 'dataset')
  ```
- **pip**：`de` 走 aliyun 镜像，很快。
- **GPU**：8× A100-80GB 与他人共享且占用重，每张卡空闲显存在 ~4 GB–~16 GB 之间波动。小模型（≤2 GB）几乎随时能跑；7B fp16（~14 GB）只有恰好遇到空闲窗口才放得下。**只试 1–2 次**，放不下就报告，**不要 kill 任何进程、不要轮询占用**（见父目录 CLAUDE.md）。

---

## 7. matplotlib `ax.grid(b=...)` 关键字被移除

**问题**
本机 `matplotlib==3.10.9`。`Axes.grid()` 的 `b=` 参数在 3.5 弃用、之后移除，改成 `visible=`。`analyze_naturalness.py` 的 `vislualize_distribution` 里：
```python
ax.grid(b=True, which='major', color='gray', linestyle='-', alpha=0.4)
```
报错（检测和 AUC 都算完了，最后画图这步崩）：
```
ValueError: keyword grid_b is not recognized; valid keywords are [..., 'grid_visible', ...]
```

**修复方式**
`b=True` → `visible=True`（`grid(b=...)` 全仓库仅此一处）。注：`plt.grid(True, ...)` 那种**位置参数**写法（`analyze_law_and_frequency.py` / `token_tagging.py` 里）不受影响，位置参数自动映射到 `visible`。已验证 mpl 3.10 下 `ax.grid(visible=True, ...)` 正常。

---

## 复现入口（验证状态）

- **Demo（轻量，已跑通）**：`code-generation/generate.py`（codeparrot-small，1000 样本）→ `code-detection/main.py`，AUC 约 logrank 0.98 / LRR 0.94 / NPR 0.975，写出 `results.pdf`。数值仅 sanity check（110M 弱生成器容易判）。
- **论文对齐（未在本机完整跑过）**：CodeLlama-7b-hf，10000 样本，tp0.2，`n_samples=500`，`n_perturbation_list=50`。`main.py` 的 `args_dict` 已 pin 该配置；仓库根目录 `run_codellama7b.sh [GPU_ID]` 一键跑两阶段（先查 ≥16 GiB 空闲；stage 1 若 `outputs.txt` 已存在则自动跳过）。
- **`code-analysis/` 经验研究**：tree-sitter 导入问题已修（见第 1 条），`analyze_law_and_frequency.py` / `analyze_proportion.py` / `analyze_naturalness.py` 现可导入并运行。其 `__main__` 各自硬编码读取 `../code-generation/output/<dataset>/<key>/outputs.txt`，换数据集时改对应 `load_data(...)` 的 `key=`。
