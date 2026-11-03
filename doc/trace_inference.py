"""
一次推理跟踪：用 GPT-2 权重跑短 prompt，按层打印张量形状 + 概念要点。

面向 LLM 初学者：先对照下面总览图，再顺着 banner 0..6 看运行输出。
本文件注释含 ASCII 图；终端打印逻辑未改。

边界（先分清，再读图）：
  字符串 → tokenizer → ids → 模型 → logits → 采样 → ids → decode → 字符串
  tokenizer / decode 不属于 GPT 权重；模型入口是 token ids（banner 0），
  出口是 logits；采样与拼回序列在 generate 循环里（banner 5/6）。

全链路（一次「预测下一个 token」的 forward）：

  prompt ──tokenizer──> token ids       # Tokenizer banner（网外）
                        |
                        v
               输入 ids (B,T)            # banner 0 模型入口
                        |
                        v
           [tok_emb]+[pos_emb] ──> x    # banner 1  残差流 (B,T,C)
                         |
           +-------------+-------------+  # banner 2/3  × n_layer
           |  LN -> Attn -> +x         |
           |  LN -> MLP  -> +x         |
           +-------------+-------------+
                         |
                         v
           ln_final -> lm_head -> logits # banner 4 模型出口
                         |
                         v
                采样 next id -> 拼回序列   # banner 5/6（网外循环）
                再 forward ...（自回归）

形状约定（读代码时反复出现）：
  B = batch（本脚本通常为 1）
  T = 序列长度（token 个数）
  C = n_embd（每个位置的向量宽度，GPT-2 小模型为 768）
  nh / hs = 注意力头数 / 每头维度（C = nh * hs）

用法（在仓库根目录）:
  python doc/trace_inference.py
  python doc/trace_inference.py --prompt "Hello" --max-new-tokens 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
import tiktoken

# 允许从仓库根目录 import model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import GPT, GPTConfig  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def note(text: str) -> None:
    print(f"  · {text}")


def show(name: str, t: torch.Tensor, extra: str = "") -> None:
    # 只看形状 + 粗略数值范围，避免刷屏
    flat = t.detach().float().reshape(-1)
    n = min(8, flat.numel())
    preview = ", ".join(f"{v:.3f}" for v in flat[:n].tolist())
    more = " ..." if flat.numel() > n else ""
    suffix = f"  |  {extra}" if extra else ""
    print(f"{name:28s} shape={tuple(t.shape)}  "
          f"mean={t.float().mean():+.4f}  "
          f"sample=[{preview}{more}]{suffix}")


def load_model(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def trace_forward(
    model: GPT,
    idx: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
) -> torch.Tensor:
    """镜像 GPT.forward，但逐步打印中间量。返回最后位置的 logits [B, vocab]。

    与 model(idx) 数学路径一致；本函数多了中间张量打印，方便对照学习。
    """
    cfg = model.config
    B, T = idx.size()
    assert T <= cfg.block_size

    banner(f"0) 模型入口：输入 token ids  B={B}, T={T}")
    ids = idx[0].tolist()
    print(f"  ids   = {ids}")
    print(f"  text  = {decode(ids)!r}")
    note("GPT.forward 从这里开始；每个整数是词表里的一个 token。")
    note("模型只吃整数 ids，不吃字符串（字符串已在网外 tokenizer 处理完）。")

    # --- embeddings ---
    # ids[i] ──tok_emb──> tok_emb[i]   (词是什么)
    # pos[i] ──pos_emb──> pos_emb[i]   (在第几位)
    # x = tok_emb + pos_emb          (相加，不是拼接)
    # 之后整网主要在 (B, T, C) 的「残差流」x 上改写表示。
    #
    # 注解说明：nn.Module.__call__ 在 stub 里是 -> Any；标成 Tensor 后
    # .split/.view/.transpose 等才能 Hover / 跳转。
    banner("1) Token + Position Embedding → 残差流起点 x")
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    tok_emb: torch.Tensor = model.transformer.tok_emb(idx)   # (B, T, C)
    pos_emb: torch.Tensor = model.transformer.pos_emb(pos)   # (T, C)  broadcast 到 batch
    x: torch.Tensor = model.transformer.emb_dropout(tok_emb + pos_emb)
    C = cfg.n_embd
    show("tok_emb = tok_emb(idx)", tok_emb, f"查表: vocab={cfg.vocab_size} → C={C}")
    show("pos_emb = pos_emb(pos)", pos_emb, "位置 0..T-1 各自一个向量")
    show("x = emb_dropout(tok+pos)", x)
    note("概念：离散 token → 连续向量；位置信息加进同一空间（加法，不是拼接）。")
    note(f"之后整网主要在形状 (B, T, C={C}) 的「残差流」上改写表示。")

    # --- blocks ---
    # 单层 Block（Pre-LN + 两段残差）：
    #
    #   x --ln1--> Attn --+--> x --ln2--> MLP --+--> x
    #      ^______________|        ^______________|
    #
    # 即：x = x + Attn(LN(x));  x = x + MLP(LN(x))
    nh, hs = cfg.n_head, C // cfg.n_head
    # ModuleList 迭代类型是 Module；用 iter_blocks() 收窄成 Block，才能点进 attn/mlp
    for layer_i, block in enumerate(model.transformer.iter_blocks()):
        # 只详细打印第 0 层，其余只打出口形状
        detail = layer_i == 0
        if detail:
            banner(f"2) Transformer Block[{layer_i}] 详解  "
                   f"(共 {cfg.n_layer} 层；其余层结构相同)")
        elif layer_i == 1:
            banner(f"3) Block[1..{cfg.n_layer - 1}] 摘要（同构堆叠）")

        x_in = x

        # Pre-LN → Attention
        ln1: torch.Tensor = block.ln1(x)
        # qkv_proj 一次算出 Q|K|V 拼在一起：(B,T,3C) → 再均分三份
        qkv: torch.Tensor = block.attn.qkv_proj(ln1)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.split(C, dim=2)
        # (B,T,C) → (B,T,nh,hs) → (B,nh,T,hs)：把 C 拆成 nh 个头，每头 hs 维
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # 注意力：每个 query 位置用「和各 key 的相似度」对 value 做加权平均
        #
        #   att = (Q @ K^T) / sqrt(hs)     形状 (B, nh, T, T)
        #   因果 mask：只能看过去（下三角），未来位置填 -inf
        #   probs = softmax(att) ;  y = probs @ V
        #
        # 因果 mask 示意（行=query 位置 t，列=key 位置 j；* 可见，. 屏蔽）：
        #
        #         j=0  j=1  j=2  j=3
        #   t=0    *    .    .    .
        #   t=1    *    *    .    .
        #   t=2    *    *    *    .
        #   t=3    *    *    *    *
        #
        # 这是「根据已有上文预测下一个词」，而不是让位置偷看未来再改写自己。
        att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att_prob: torch.Tensor = F.softmax(att, dim=-1)
        y: torch.Tensor = att_prob @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 多头拼回 (B,T,C)
        attn_out: torch.Tensor = block.attn.resid_dropout(block.attn.out_proj(y))
        x = x_in + attn_out

        if detail:
            show("ln1(x)", ln1, "Pre-LN：先归一化再进注意力")
            show("q / k / v", q, f"nh={nh}, head_dim={hs}")
            show("att scores (masked)", att, "(B, nh, T, T) 上三角为 -inf")
            # 打印 head0 的注意力行，看因果结构
            print("  head0 attention probs (rows=query pos, cols=key pos):")
            probs = att_prob[0, 0].cpu()
            for t in range(T):
                row = " ".join(f"{probs[t, j]:.2f}" for j in range(T))
                print(f"    t={t}: [{row}]")
            note("因果：位置 t 只能看 0..t，不能看未来 → 下三角非零。")
            note("概念：softmax(QKᵀ/√d) V = 用「相似度」对 value 做加权平均。")
            show("attn_out", attn_out)
            show("x = x + attn_out", x, "残差连接：保留原信息 + 注意力更新")

        # Pre-LN → MLP（逐位置的通道混合 / 非线性；不在 token 之间通信）
        x_mid = x
        ln2: torch.Tensor = block.ln2(x)
        h: torch.Tensor = block.mlp.gelu(block.mlp.fc_in(ln2))       # (B, T, 4C)
        mlp_out: torch.Tensor = block.mlp.dropout(block.mlp.fc_out(h))
        x = x_mid + mlp_out

        if detail:
            show("mlp hidden (4C)", h, f"扩展到 4*{C}={4 * C}，再压回 C")
            show("mlp_out", mlp_out)
            show("x = x + mlp_out", x, "残差：通道混合 / 非线性变换")
            note("Block 公式：x ← x + Attn(LN(x));  x ← x + MLP(LN(x))")
        else:
            if layer_i < 3 or layer_i == cfg.n_layer - 1:
                show(f"block[{layer_i}] out", x)
            elif layer_i == 3:
                print(f"  ... (省略 block[3..{cfg.n_layer - 2}])")

    # --- final norm + head ---
    # lm_head：把最后位置的 C 维向量映到 vocab 维分数（logits）
    # weight tying：读出矩阵与 tok_emb 嵌入共用同一权重（既嵌入又读出）
    banner("4) 模型出口：Final LayerNorm + lm_head → logits")
    x = model.transformer.ln_final(x)
    show("ln_final(x)", x)
    # 与 GPT.forward 推理路径一致：只算最后一个位置
    logits_last: torch.Tensor = model.lm_head(x[:, [-1], :])  # (B, 1, vocab)
    show("logits (last pos)", logits_last, f"vocab={cfg.vocab_size}")
    note("GPT.forward 到 logits 为止；采样 / decode 在网外（banner 5/6）。")
    note("lm_head 与 tok_emb 权重共享（weight tying）：同一矩阵既嵌入又读出。")
    note("logits[i] 越大 → 模型认为下一个 token 是词表第 i 个的「分数」越高。")

    logits: torch.Tensor = logits_last[:, 0, :]  # (B, vocab)
    probs: torch.Tensor = F.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs[0], 5)
    print("  top-5 next-token 预测:")
    for p, i in zip(topv.tolist(), topi.tolist()):
        print(f"    p={p:.4f}  id={i:5d}  {decode([i])!r}")
    return logits


@torch.no_grad()
def trace_one_generate_step(
    model: GPT,
    idx: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
) -> torch.Tensor:
    # 自回归一步（生成 N 个新 token ≈ 调用 forward N 次）：
    #
    #   完整序列 ids ──forward──> 只取最后位置 logits
    #                                 |
    #                        temperature / top_k
    #                                 |
    #                              采样一个 id
    #                                 |
    #                        cat 到序列末尾 → 重复
    #
    # 这里走官方 model.forward，不打印中间层；与上面 trace_forward 数学同路径。
    # 用 .forward 而非 model(idx)：后者经 Module.__call__，返回类型是 Any。
    banner("5) 自回归一步：logits → 采样 → 拼回序列")
    logits, _ = model.forward(idx)  # 走官方 forward（推理只算最后位）
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs: torch.Tensor = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    print(f"  采样得到 id={idx_next.item()}  text={decode(idx_next[0].tolist())!r}")
    note("generate() 就是把新 token 拼到 idx 右边，再重复 forward，共 max_new_tokens 次。")
    idx = torch.cat((idx, idx_next), dim=1)
    print(f"  新序列 ids = {idx[0].tolist()}")
    print(f"  新序列 text = {decode(idx[0].tolist())!r}")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one nanoGPT inference")
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "out-gpt2", "ckpt.pt"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)

    banner("加载权重")
    print(f"  ckpt = {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    cfg = model.config
    print(f"  GPT-2: n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_embd={cfg.n_embd} block_size={cfg.block_size} vocab={cfg.vocab_size}")

    banner("Tokenizer：prompt → token ids（tiktoken GPT-2 BPE）")
    print(f"  prompt = {args.prompt!r}")
    ids = encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)
    print(f"  ids    = {ids}")
    print(f"  text   = {decode(ids)!r}")
    note("tokenizer = 字符串 → 子词 → 词表整数；模型只吃整数，不吃字符串。")
    note("下一行把 ids 包成 (B,T) 张量，本身不是 tokenize。")

    # 详细跟踪一次 forward（对照文件顶部总览图 banner 0..4）
    trace_forward(model, idx, decode)

    # 先拆开看「采样一步」，再用官方 generate 跑完对照（banner 5/6）
    banner("6) 自回归采样")
    print(f"  prompt = {args.prompt!r}")
    trace_one_generate_step(
        model, idx, decode,
        temperature=args.temperature, top_k=args.top_k,
    )
    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
    print(f"\n  官方 generate 完整输出 ({args.max_new_tokens} new tokens):")
    print(f"  {decode(out[0].tolist())!r}")
    banner("结束")
    note("对照阅读：model.py 里 GPT.forward / CausalSelfAttention / Block / generate")
    note("下一问可以点名某一段输出（例如 attention 矩阵）深入讲。")


if __name__ == "__main__":
    main()
