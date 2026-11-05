"""
一次推理跟踪：用 GPT-2 权重跑短 prompt，按层打印张量形状 + 概念要点。

面向 LLM 初学者：先对照下面总览图，再顺着 banner 0..6 看运行输出。
本文件注释含 ASCII 图；终端打印逻辑未改。

边界（先分清，再读图）：
  字符串 → tokenizer → ids → 模型 → logits → 采样 → ids → decode → 字符串
  tokenizer / decode 不属于 GPT 权重；模型入口是 token ids（banner 0），
  出口是 logits。KV cache 在每层 Attention 里写出（见 banner 2）。

全链路：

  prompt ──tokenizer──> token ids       # Tokenizer banner（网外）
                        |
                        v
               输入 ids (B,T)            # banner 0 模型入口
                        |
                        v
           [tok_emb]+[pos_emb] ──> x    # banner 1  残差流 (B,T,C)
                         |
           +-------------+-------------+  # banner 2/3  × n_layer  ← Prefill
           |  LN -> Attn -> +x         |     Attn 算出 K/V 并写入 kv_cache[i]
           |  LN -> MLP  -> +x         |
           +-------------+-------------+
                         |
                         v
           ln_final -> lm_head -> logits # banner 4
                         |
                         v
                采样 next id             # banner 5（网外；用已有 logits）
                         |
                         v
           Decode: 新 token + 复用 cache # banner 5b（Layer0 再看一遍 Attn）
           generate 循环 …               # banner 6

  trace_forward 手写镜像 Attention，与 CausalSelfAttention 同路径；顺带收集每层 (K,V)。

形状约定（读代码时反复出现）：
  B = batch（本脚本通常为 1）
  T = 序列长度（token 个数）
  C = n_embd（每个位置的向量宽度，GPT-2 小模型为 768）
  nh / hs = 注意力头数 / 每头维度（C = nh * hs）
  T_cache = 已缓存的 token 数（decode 时 T=1，T_cache 递增）

用法（在仓库根目录）:
  python doc/trace_inference.py
  python doc/trace_inference.py --prompt "Hello, I am" --max-new-tokens 5
  python doc/trace_inference.py --no-kv-cache   # Decode 重喂整段，对照有 cache
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import tiktoken

# 允许从仓库根目录 import model
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gpt2_weights import normalize_native_state_dict  # noqa: E402
from model import GPT, GPTConfig, KVCache  # noqa: E402


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
    config = GPTConfig.from_model_args(ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(normalize_native_state_dict(ckpt["model"]))
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def trace_forward(
    model: GPT,
    idx: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
    use_kv_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """镜像 Prefill：逐步打印中间量。

    use_kv_cache=True 时收集每层 (K,V)；False 时仍算 Attn，但不保存 cache。
    返回 (最后位置 logits [B, vocab], kv_cache | None)。
    """
    cfg = model.config
    B, T = idx.size()
    assert T <= cfg.context_length

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

    # --- layers ---
    # 单层 Layer（Pre-LN + 两段残差）：
    #
    #   x --ln1--> Attn --+--> x --ln2--> MLP --+--> x
    #      ^______________|        ^______________|
    #
    # 即：x = x + Attn(LN(x));  x = x + MLP(LN(x))
    # Prefill 时每层 Attn 算出 (K,V)；use_kv_cache 时写入 kv_cache[layer] 供 Decode 复用。
    nh, hs = cfg.n_head, C // cfg.n_head
    kv_cache: Optional[KVCache] = [] if use_kv_cache else None
    # ModuleList 迭代类型是 Module；用 iter_layers() 收窄成 Layer，才能点进 attn/mlp
    for layer_i, layer in enumerate(model.transformer.iter_layers()):
        # 只详细打印第 0 层，其余只打出口形状
        detail = layer_i == 0
        if detail:
            banner(f"2) Transformer Layer[{layer_i}] 详解  "
                   f"(共 {cfg.n_layer} 层；其余层结构相同)")
        elif layer_i == 1:
            banner(f"3) Layer[1..{cfg.n_layer - 1}] 摘要（同构堆叠）")

        x_in = x

        # Pre-LN → Attention
        ln1: torch.Tensor = layer.ln1(x)
        # qkv_proj 一次算出 Q|K|V 拼在一起：(B,T,3C) → 再均分三份
        qkv: torch.Tensor = layer.attn.qkv_proj(ln1)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.split(C, dim=2)
        # (B,T,C) → (B,T,nh,hs) → (B,nh,T,hs)：把 C 拆成 nh 个头，每头 hs 维
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        if kv_cache is not None:
            # 与 CausalSelfAttention 在 use_cache 时保存的条目相同
            kv_cache.append((k, v))

        # 注意力：每个 query 位置用「和各 key 的相似度」对 value 做加权平均
        #
        #   attn_weights = (Q @ K^T) / sqrt(hs)     形状 (B, nh, T, T)
        #   因果 mask：只能看过去（下三角），未来位置填 -inf
        #   probs = softmax(attn_weights) ;  y = probs @ V
        #
        # batched 矩阵乘 (@)：前面维是批，最后两维才是矩阵。
        # 对每个 (b, head) 各做一次普通 matmul，共 B*nh 次，合成一个张量：
        #
        #   q:              (B, nh, T, hs)
        #   k^T:            (B, nh, hs, T)   <- k.transpose(-2, -1)
        #   q @ k^T:        (B, nh, T,  T)
        #
        # --- 其它维：对齐 / 广播（@ 只吃最后两维；前面维跟 +/* 一样走广播）---
        #
        #   规则：除最后两维外，对应位置要么相等，要么有一边是 1（可扩）。
        #   本行两边前面都是 (B, nh)，完全对齐，无需广播：
        #
        #            批维对齐（相等）     矩阵乘（最后两维）
        #            ---------------     ------------------
        #   q:       ( B , nh ,          T  , hs )
        #   k^T:     ( B , nh ,          hs , T  )
        #             |    |              |    |
        #             v    v              v    v
        #   out:     ( B , nh ,          T  , T  )
        #             ^----^ 逐槽位配对      ^--^ Q@K^T
        #
        #   槽位示意（每个格子是一次独立的 T×hs @ hs×T）：
        #
        #            head0      head1     ...  head{nh-1}
        #   batch0   [matmul]   [matmul]  ...  [matmul]
        #   batch1   [matmul]   [matmul]  ...  [matmul]
        #    ...
        #   batch{B-1} ...
        #
        #   若一边批维为 1，则广播复用（本注意力路径不会发生，仅示意）：
        #
        #   A: (1, T, hs)     B: (B, hs, T)
        #       |                  |
        #       +---- 1 -> B ------+   广播后对每个 b 各做一次 matmul
        #   out: (B, T, T)
        #
        #   固定一个 (b,h)，就是：
        #
        #        Q_(T x hs)  @  K^T_(hs x T)  =  Att_(T x T)
        #
        #        [q0 ·········]   [k0 k1 k2 k3]   [s00 s01 s02 s03]
        #        [q1 ·········]   [·  ·  ·  · ]   [s10 s11 s12 s13]
        #        [q2 ·········] @ [·  ·  ·  · ] = [s20 s21 s22 s23]
        #        [q3 ·········]   [·  ·  ·  · ]   [s30 s31 s32 s33]
        #                           ^ hs 行
        #
        #   每个 sij = qi · kj（点积相似度）；batch/head 维不参与求和，只是并行。
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
        attn_weights: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        # --- softmax → 加权 V → 拼头 → 投影 → 残差 ---
        #
        # 1) softmax(attn_weights, dim=-1)：对每个 query 行，把分数变成概率（行和=1）
        #    -inf 位置 → 概率 0。固定一个 (b,h)，一行示意：
        #
        #         j=0   j=1   j=2   j=3
        #   t=2  [0.1 , 0.3 , 0.6 , 0.0 ]   <- 已 mask 的未来为 0
        #         ^-----------^ 只看 0..t
        #
        # 2) y = attn_prob @ v：用概率对 value 做加权平均
        #
        #   attn_prob: (B, nh, T, T)
        #   v:         (B, nh, T, hs)
        #   y:         (B, nh, T, hs)
        #
        #   固定 (b,h)、固定 query 位置 t：
        #
        #     y[t] = p[t,0]*v[0] + p[t,1]*v[1] + ... + p[t,T-1]*v[T-1]
        #            (每个 v[j] 是 hs 维向量)
        #
        #        probs_(T x T)  @  V_(T x hs)  =  Y_(T x hs)
        #
        #        [p00 p01 p02 p03]   [v0····]   [y0····]
        #        [p10 p11 p12 p13] @ [v1····] = [y1····]
        #        [p20 p21 p22 p23]   [v2····]   [y2····]
        #        [p30 p31 p32 p33]   [v3····]   [y3····]
        #
        # 3) 多头拼回 (B,T,C)：各头仍分开，要把 nh 个头沿通道维拼接
        #
        #   y: (B, nh, T, hs)
        #        |
        #        |  transpose(1, 2)  →  (B, T, nh, hs)
        #        |  contiguous()     →  按新顺序重排 storage（transpose 只改 stride）
        #        |  view(B, T, C)    →  (B, T, nh*hs)  即 (B, T, C)
        #        v
        #
        #   固定一个 token t（略 batch），拼头示意 nh=2, hs=2 → C=4：
        #
        #     拼前 (nh, hs):     head0=[a b]  head1=[c d]
        #     拼后 (C,):         [a b c d]
        #
        #   为何要 contiguous？view 要求内存按新 shape 连续排布。
        #   transpose 后逻辑顺序变了，storage 还没搬家，直接 view 会失败：
        #
        #     storage 仍按 (nh,T,hs) 排:  h0t0 | h0t1 | h0t2 | h1t0 | h1t1 | h1t2
        #     想 view 成 t0 的一行:       h0t0 , h1t0  ← 中间隔着 h0t1,h0t2，不连续
        #     contiguous() 拷成:          h0t0 | h1t0 | h0t1 | h1t1 | ... 再 view
        #
        # 4) out_proj：线性层再混一次通道（学「怎么合并各头」）+ resid dropout
        # 5) 残差：x = x_in + attn_out（保留进注意力前的信息 + 注意力更新）
        #
        attn_prob: torch.Tensor = F.softmax(attn_weights, dim=-1)
        y: torch.Tensor = attn_prob @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 多头拼回 (B,T,C)
        attn_out: torch.Tensor = layer.attn.resid_dropout(layer.attn.out_proj(y))
        x = x_in + attn_out

        if detail:
            show("ln1(x)", ln1, "Pre-LN：先归一化再进注意力")
            show("q / k / v", q, f"nh={nh}, head_dim={hs}")
            if use_kv_cache:
                show("kv_cache[0].K ← k", k, f"T_cache={k.size(2)}；本层 Attn 写出，供 Decode 复用")
                show("kv_cache[0].V ← v", v)
                note(f"每一层各自存一份 (K,V)；共 {cfg.n_layer} 层 → kv_cache 长度 {cfg.n_layer}。")
                note("Prefill：K/V 覆盖整段 prompt；Decode：只算新 token 的 K/V，再 cat 到 cache。")
            else:
                show("k / v（本步算出，未写入 cache）", k, "--no-kv-cache：Decode 将重算整段")
                note("关闭 KV cache：每步自回归都要重新 forward 全部已生成 tokens。")
            show("attn scores (masked)", attn_weights, "(B, nh, T, T) 上三角为 -inf")
            # 打印 head0 的注意力行，看因果结构
            print("  head0 attention probs (rows=query pos, cols=key pos):")
            probs = attn_prob[0, 0].cpu()
            for t in range(T):
                row = " ".join(f"{probs[t, j]:.2f}" for j in range(T))
                print(f"    t={t}: [{row}]")
            note("因果：位置 t 只能看 0..t，不能看未来 → 下三角非零。")
            note("概念：softmax(QKᵀ/√d) V = 用「相似度」对 value 做加权平均。")
            show("attn_out", attn_out)
            show("x = x + attn_out", x, "残差连接：保留原信息 + 注意力更新")

        # Pre-LN → MLP（逐位置的通道混合 / 非线性；不在 token 之间通信）
        x_mid = x
        ln2: torch.Tensor = layer.ln2(x)
        h: torch.Tensor = layer.mlp.gelu(layer.mlp.up_proj(ln2))       # (B, T, 4C)
        mlp_out: torch.Tensor = layer.mlp.resid_dropout(layer.mlp.down_proj(h))
        x = x_mid + mlp_out

        if detail:
            show("mlp hidden (4C)", h, f"扩展到 4*{C}={4 * C}，再压回 C")
            show("mlp_out", mlp_out)
            show("x = x + mlp_out", x, "残差：通道混合 / 非线性变换")
            note("Layer 公式：x ← x + Attn(LN(x));  x ← x + MLP(LN(x))")
        else:
            if layer_i < 3 or layer_i == cfg.n_layer - 1:
                show(f"layer[{layer_i}] out", x)
            elif layer_i == 3:
                print(f"  ... (省略 layer[3..{cfg.n_layer - 2}])")

    # --- final norm + head ---
    # lm_head：把最后位置的 C 维向量映到 vocab 维分数（logits）
    # weight tying：读出矩阵与 tok_emb 嵌入共用同一权重（既嵌入又读出）
    banner("4) 模型出口：Final LayerNorm + lm_head → logits")
    x = model.transformer.ln_final(x)
    show("ln_final(x)", x)
    # 与 GPT.forward 推理路径一致：只算最后一个位置
    logits_last: torch.Tensor = model.lm_head(x[:, [-1], :])  # (B, 1, vocab)
    show("logits (last pos)", logits_last, f"vocab={cfg.vocab_size}")
    if use_kv_cache:
        note("Prefill 结束：logits 出站；KV 已在各层 Attention 写入（见 banner 2）。")
    else:
        note("Prefill 结束：logits 出站；未保存 KV（--no-kv-cache）。")
    note("lm_head 与 tok_emb 权重共享（weight tying）：同一矩阵既嵌入又读出。")
    note("logits[i] 越大 → 模型认为下一个 token 是词表第 i 个的「分数」越高。")

    logits: torch.Tensor = logits_last[:, 0, :]  # (B, vocab)
    probs: torch.Tensor = F.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs[0], 5)
    print("  top-5 next-token 预测:")
    for p, i in zip(topv.tolist(), topi.tolist()):
        print(f"    p={p:.4f}  id={i:5d}  {decode([i])!r}")
    return logits, kv_cache


@torch.no_grad()
def sample_next(
    logits: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
    temperature: float = 1.0,
    top_k: Optional[int] = 40,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """网外采样：只用已有 logits，不再跑模型。"""
    banner("5) 采样：logits → next token（网外）")
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs: torch.Tensor = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    print(f"  采样得到 id={idx_next.item()}  text={decode(idx_next[0].tolist())!r}")
    if use_kv_cache:
        note("下一步 Decode 只喂这个新 token；上文的 K/V 已在 Prefill 各层写好。")
    else:
        note("下一步将重喂「prompt + 新 token」整段（无 cache）。")
    return idx_next


@torch.no_grad()
def trace_decode_nocache(
    model: GPT,
    idx_next: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
    idx_so_far: torch.Tensor,
) -> torch.Tensor:
    """无 KV cache：把整段序列再 forward 一次（对照有 cache 时只喂 T=1）。"""
    idx = torch.cat((idx_so_far, idx_next), dim=1)
    banner("5b) Decode（无 KV cache）：整段序列再 forward")
    show("输入 idx（全文）", idx, f"T={idx.size(1)}，含 prompt + 新 token")
    note("每一步自回归都重算全部位置的 K/V；代价随长度增长。")
    logits, _ = model.forward(idx)
    print(f"  新序列 ids = {idx[0].tolist()}")
    print(f"  新序列 text = {decode(idx[0].tolist())!r}")
    return logits[:, 0, :]


@torch.no_grad()
def trace_decode_step(
    model: GPT,
    idx_next: torch.Tensor,
    kv_cache: KVCache,
    decode: Callable[[Sequence[int]], str],
    idx_so_far: torch.Tensor,
) -> Tuple[torch.Tensor, KVCache]:
    """Decode 一步：只喂新 token；Layer[0] 详解里展示 K/V cat 进 cache。"""
    cfg = model.config
    B, T = idx_next.size()
    assert T == 1
    past_len = kv_cache[0][0].size(2)
    C = cfg.n_embd
    nh, hs = cfg.n_head, C // cfg.n_head

    banner("5b) Decode：Layer[0] Attention + 复用 KV cache")
    show("输入 idx_next", idx_next, f"T=1；past_len={past_len}")
    note("位置从 past_len 起算；不再重算 prompt 的 K/V。")

    pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx_next.device)
    tok_emb: torch.Tensor = model.transformer.tok_emb(idx_next)
    pos_emb: torch.Tensor = model.transformer.pos_emb(pos)
    x: torch.Tensor = model.transformer.emb_dropout(tok_emb + pos_emb)

    new_cache: KVCache = []
    for layer_i, layer in enumerate(model.transformer.iter_layers()):
        detail = layer_i == 0
        x_in = x
        ln1: torch.Tensor = layer.ln1(x)
        qkv: torch.Tensor = layer.attn.qkv_proj(ln1)
        q, k_new, v_new = qkv.split(C, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k_new = k_new.view(B, T, nh, hs).transpose(1, 2)
        v_new = v_new.view(B, T, nh, hs).transpose(1, 2)

        pk, pv = kv_cache[layer_i]
        k = torch.cat([pk, k_new], dim=2)
        v = torch.cat([pv, v_new], dim=2)
        new_cache.append((k, v))
        T_k = k.size(2)

        # Decode：Q 只有新位置；K/V 含历史 → attn (B, nh, 1, T_k)
        attn_weights: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        attn_prob: torch.Tensor = F.softmax(attn_weights, dim=-1)
        y: torch.Tensor = attn_prob @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        attn_out: torch.Tensor = layer.attn.resid_dropout(layer.attn.out_proj(y))
        x = x_in + attn_out

        if detail:
            show("k_new / v_new（仅新 token）", k_new, "T=1")
            show("kv_cache[0].K after cat", k, f"T_cache={T_k}（= past+1）")
            show("attn scores", attn_weights, f"(B, nh, 1, {T_k})：一行 query × 全部 cached keys")
            row = " ".join(f"{p:.2f}" for p in attn_prob[0, 0, 0].cpu().tolist())
            print(f"  head0 attention probs: [{row}]")
            note("历史 K 前缀应与 Prefill 的 kv_cache[0].K 一致（只在末尾多一列）。")

        x_mid = x
        ln2: torch.Tensor = layer.ln2(x)
        h: torch.Tensor = layer.mlp.gelu(layer.mlp.up_proj(ln2))
        mlp_out: torch.Tensor = layer.mlp.resid_dropout(layer.mlp.down_proj(h))
        x = x_mid + mlp_out

    x = model.transformer.ln_final(x)
    logits_last: torch.Tensor = model.lm_head(x[:, [-1], :])
    idx = torch.cat((idx_so_far, idx_next), dim=1)
    print(f"  新序列 ids = {idx[0].tolist()}")
    print(f"  新序列 text = {decode(idx[0].tolist())!r}")
    return logits_last[:, 0, :], new_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one nanoGPT inference")
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "out-gpt2", "ckpt.pt"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--kv-cache",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Decode/generate 是否使用 KV cache（默认开；--no-kv-cache 关闭）",
    )
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
          f"n_embd={cfg.n_embd} context_length={cfg.context_length} vocab={cfg.vocab_size}")
    print(f"  kv_cache = {args.kv_cache}")

    banner("Tokenizer：prompt → token ids（tiktoken GPT-2 BPE）")
    print(f"  prompt = {args.prompt!r}")
    ids = encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)
    print(f"  ids    = {ids}")
    print(f"  text   = {decode(ids)!r}")
    note("tokenizer = 字符串 → 子词 → 词表整数；模型只吃整数，不吃字符串。")
    note("下一行把 ids 包成 (B,T) 张量，本身不是 tokenize。")

    # Prefill → 采样 → Decode（有/无 cache）→ generate
    print(f"\n  prompt = {args.prompt!r}")
    logits, kv_cache = trace_forward(model, idx, decode, use_kv_cache=args.kv_cache)
    idx_next = sample_next(
        logits, decode,
        temperature=args.temperature, top_k=args.top_k,
        use_kv_cache=args.kv_cache,
    )
    if args.kv_cache:
        if kv_cache is None:
            raise RuntimeError("use_kv_cache=True but kv_cache is None")
        trace_decode_step(model, idx_next, kv_cache, decode, idx)
        gen_banner = "6) 官方 generate（1×Prefill 写 cache + N×Decode）"
    else:
        trace_decode_nocache(model, idx_next, decode, idx)
        gen_banner = "6) 官方 generate（use_cache=False：每步重喂全文）"
    banner(gen_banner)
    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            use_cache=args.kv_cache,
        )
    print(f"  ({args.max_new_tokens} new tokens): {decode(out[0].tolist())!r}")
    banner("结束")
    note("对照：model.py CausalSelfAttention / GPT.forward(use_cache) / generate")
    if args.kv_cache:
        note("2=Attn 写 KV；5=采样；5b=Decode cat KV；6=完整 generate。")
    else:
        note("2=算出 K/V 但不存；5=采样；5b/6=每步重喂全文（--no-kv-cache）。")


if __name__ == "__main__":
    main()
