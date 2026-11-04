# nanoGPT 项目上下文

Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)：用尽量少的代码训练 / 微调中等规模 GPT。核心是两份约 300 行的文件——`model.py`（模型）和 `train.py`（训练循环）。仓库已偏旧，作者更推荐 [nanochat](https://github.com/karpathy/nanochat)，本仓库仍适合学习 GPT 训练流水线。

## 根目录地图

入口脚本留根；分析 / 测试 / 产物下沉。

| 路径 | 角色 |
| ------ | ------ |
| `model.py` / `train.py` / `sample.py` / `configurator.py` | 原版入口 |
| `gpt2_weights.py` / `bench.py` | 权重边界、吞吐基准（仍根目录） |
| `config/` / `data/` / `assets/` | 预设、语料、配图 |
| `tests/` | 回归测试 |
| `notebooks/` | 参数量 / 缩放律分析 |
| `doc/` | `_context`、推理跟踪脚本 |
| `out-*/` | 训练产物（gitignore） |

## 典型流水线

```text
prepare.py  →  train.bin / val.bin (+ meta.pkl)
     ↓
train.py (+ config/*.py)  →  out*/ckpt.pt
     ↓
sample.py  →  文本生成
```

---

## 根目录 Python 文件

### `model.py` — GPT 模型定义

单文件完整 GPT Language Model，对齐 OpenAI GPT-2 / HuggingFace 实现。

| 组件 | 作用 |
| ------ | ------ |
| `LayerNorm` | 可选 bias 的 LayerNorm |
| `CausalSelfAttention` | 因果自注意力（Flash Attention 可用时走高效路径） |
| `MLP` | 前馈层（GELU） |
| `Layer` | Transformer layer：Attn + MLP + 残差 |
| `GPTConfig` | 超参：`context_length / vocab_size / n_layer / n_head / n_embd / dropout / bias` |
| `GPT` | 完整模型：token/pos embedding、多层 Layer、`lm_head`（与 `tok_emb` weight tying） |

主要方法：

- `forward(idx, targets=None)`：训练时算 cross-entropy loss；推理时只对最后位置出 logits
- `from_pretrained(model_type)`：从 HuggingFace 加载 GPT-2；边界转换见 `gpt2_weights.convert_gpt2_state_dict`
- `configure_optimizers(...)`：分组 AdamW（2D 参数 weight decay，bias/LN 不 decay）
- `generate(...)`：自回归采样（temperature / top_k）
- `estimate_mfu(...)`：估算相对 A100 bf16 峰值的 MFU
- `crop_context_length(...)`：缩小上下文长度（微调小 context 时用）

### `gpt2_weights.py` — 权重格式转换（唯一边界）

三种格式（`WeightFormat`）：

| 格式 | 键名 | 权重布局 |
| ------ | ------ | ---------- |
| `HF_GPT2` | GPT-2（`wte`/`c_attn`/`h`） | Conv1D `(in, out)` |
| `NANOGPT_LEGACY` | 同上 | Linear `(out, in)`（原版 nanoGPT ckpt） |
| `NATIVE` | 本仓（`tok_emb`/`qkv_proj`/`up_proj`/`down_proj`/`layers`） | Linear |

- `convert_gpt2_state_dict`：按完整参数表转换；缺键/余键/形状不符即报错
- `GPT.from_pretrained`：声明 `HF_GPT2` → NATIVE，再 `load_state_dict`
- train / sample / trace：**只** `load_state_dict` NATIVE；旧 ckpt 先 migrate（含 `blocks`→`layers` 键名）

```bash
python gpt2_weights.py migrate out-gpt2/ckpt.pt
python gpt2_weights.py migrate out-gpt2/ckpt.pt -o out-gpt2/ckpt.native.pt
```

### 回归测试

```bash
python -m unittest tests.test_regression -v
```

覆盖：格式探测、HF/legacy 往返、缺键/余键拒绝、真实 `out-gpt2` native↔legacy bak 逐张量 + logits 一致。

默认配置接近 GPT-2 124M：`n_layer=12, n_head=12, n_embd=768, context_length=1024`。

---

### `train.py` — 训练入口

单卡调试与多卡 DDP 共用同一脚本。

**能力概览：**

- 从零训练（`init_from='scratch'`）、断点续训（`resume`）、或加载预训练 GPT-2（`gpt2*`）
- `np.memmap` 读 `data/<dataset>/train.bin`、`val.bin`
- 梯度累积、学习率 warmup + cosine decay、grad clip
- AMP（fp16/bf16）、可选 `torch.compile`
- 可选 wandb；定期 eval 并写 `out_dir/ckpt.pt`
- DDP：`torchrun --standalone --nproc_per_node=N train.py ...`

**常用启动：**

```bash
# 字符级莎士比亚（小模型，几分钟）
python train.py config/train_shakespeare_char.py

# 复现 GPT-2 124M（需多卡 + OpenWebText）
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# 命令行覆盖超参
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

默认超参面向 OpenWebText 上训练 GPT-2 124M（`max_iters=600000` 等）。

---

### `sample.py` — 从已训模型采样

加载 checkpoint 或官方 GPT-2，按 prompt 生成文本。

关键参数：

| 参数 | 含义 |
| ------ | ------ |
| `init_from` | `'resume'`（读 `out_dir/ckpt.pt`）或 `'gpt2'` / `'gpt2-xl'` 等 |
| `out_dir` | checkpoint 目录（仅 `resume` 时有效） |
| `start` | 起始 prompt；也可用 `"FILE:prompt.txt"` |
| `num_samples` / `max_new_tokens` | 条数与每条长度 |
| `temperature` / `top_k` | 采样随机性控制 |

编解码：若数据集有 `meta.pkl`（字符级）则用其中的 `stoi/itos`；否则默认 GPT-2 tiktoken BPE。

```bash
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --start="Hello"
```

---

### `configurator.py` — 轻量配置覆盖

不是独立模块，由 `train.py` / `sample.py` / `bench.py` 通过 `exec(open('configurator.py').read())` 注入。

解析 `sys.argv`：

1. 不含 `=` 的参数 → 当作配置文件路径，`exec` 其内容覆盖 globals
2. `--key=value` → 覆盖同名全局变量（`literal_eval` 解析类型）

因此支持：`python train.py config/xxx.py --batch_size=32`。

---

### `bench.py` — 训练性能基准

`train.py` 的精简版，只测吞吐，不做完整训练/存盘。

- 默认 GPT-2 124M 规模 + OpenWebText（或假数据）
- 先 burn-in 再计时，打印 **ms/iter** 与 **MFU**
- `profile=True` 时走 PyTorch Profiler，写出 `./bench_log`

用于对比 compile、dtype、硬件等对训练速度的影响。

---

## `config/` — 训练 / 评测预设

这些文件被 `configurator` exec，用来覆盖 `train.py` 里的默认全局变量。

| 文件 | 用途 |
| ------ | ------ |
| `train_shakespeare_char.py` | 小字符级莎士比亚：6 层 / 384 dim，快速上手 |
| `train_gpt2.py` | OpenWebText 上复现 GPT-2 124M（8×A100 级配置） |
| `finetune_shakespeare.py` | 用 GPT-2 XL 微调 token 级莎士比亚 |
| `eval_gpt2.py` | 评测官方 gpt2（124M），`eval_only=True` |
| `eval_gpt2_medium.py` | 评测 gpt2-medium（350M） |
| `eval_gpt2_large.py` | 评测 gpt2-large（774M） |
| `eval_gpt2_xl.py` | 评测 gpt2-xl（1.5B） |

---

## `data/*/prepare.py` — 数据预处理

训练前必须先跑对应 `prepare.py`，生成 `train.bin` / `val.bin`（`uint16` token id 流）。

| 脚本 | 数据集 | 编码 | 产出 |
| ------ | -------- | ------ | ------ |
| `data/shakespeare_char/prepare.py` | Tiny Shakespeare | 字符级（~65 vocab） | `train.bin`, `val.bin`, `meta.pkl` |
| `data/shakespeare/prepare.py` | Tiny Shakespeare | GPT-2 BPE（tiktoken） | `train.bin`, `val.bin` |
| `data/openwebtext/prepare.py` | OpenWebText（HF） | GPT-2 BPE | 大规模 `train.bin` / `val.bin` |

`train.py` 通过 `dataset` 配置名定位到 `data/<dataset>/`。

---

## 其他相关资产

| 路径 | 说明 |
| ------ | ------ |
| `notebooks/transformer_sizing.ipynb` | Transformer 参数量 / FLOPs 估算 |
| `notebooks/scaling_laws.ipynb` | 缩放律相关分析 |
| `tests/test_regression.py` | 权重格式 / native ckpt 回归 |
| `assets/` | README 配图 |
| `out-gpt2/` 等 | 训练输出目录（checkpoint）；`out-gpt2/ckpt.pt` 为本地转换的 GPT-2 124M |
| `doc/trace_inference.py` | **一次推理跟踪**：按层打印张量形状 + 概念要点，对照 `model.py` 学习 |

### 学习路径：一次推理跟踪

目标：先跟张量形状，概念嵌在每一步。需已有 `out-gpt2/ckpt.pt`（或改 `--ckpt`）。

```bash
# 在仓库根目录
python doc/trace_inference.py
python doc/trace_inference.py --prompt "Hello" --max-new-tokens 5
```

脚本阶段与 `model.py` 对应关系：

| 阶段 | 代码 | 关键形状 |
| ------ | ------ | ---------- |
| token/pos 嵌入 | `tok_emb` + `pos_emb` | `(B,T)` → `(B,T,C)` |
| Layer × N | `CausalSelfAttention` + `MLP` + 残差 | 保持 `(B,T,C)`；注意力分值 `(B,nh,T,T)` |
| 读出 | `ln_final` + `lm_head` | 最后位置 → `(B,vocab)` logits |
| 生成 | `generate` | 采样 1 个 id，拼回 `idx`，再 forward |

**Serving（SGLang / prefill·decode / cache hit）** 不在本仓：见 mach_kb [`software/cloud_infer/sglang_learn/_context_sglang_learn.md`](../../../mach_kb/software/cloud_infer/sglang_learn/_context_sglang_learn.md)（LPAI 上上游小模型；与本仓 nanoGPT 并行）。

---

## 依赖（简要）

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

- **torch**：训练与推理
- **numpy**：memmap 读二进制语料
- **transformers**：加载官方 GPT-2 权重
- **datasets**：下载 OpenWebText
- **tiktoken**：GPT-2 BPE
- **wandb** / **tqdm**：可选日志与进度条

---

## 最小可跑路径（字符级）

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

CPU / Mac：

```bash
python train.py config/train_shakespeare_char.py \
  --device=cpu --compile=False \
  --context_length=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000
```
