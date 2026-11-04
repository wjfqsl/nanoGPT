"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt2_weights import WeightFormat, convert_gpt2_state_dict


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    @classmethod
    def from_model_args(cls, args: Mapping[str, Any]) -> GPTConfig:
        """Build config from checkpoint/cli dict; accept legacy key block_size."""
        d = dict(args)
        if "context_length" not in d and "block_size" in d:
            d["context_length"] = d["block_size"]
        known = {f.name for f in fields(cls)}
        return cls(**{k: d[k] for k in known if k in d})


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    weight: nn.Parameter
    bias: Optional[nn.Parameter]

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    qkv_proj: nn.Linear
    out_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout
    n_head: int
    n_embd: int
    dropout_p: float
    flash: bool

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # lower-triangular causal mask for the slow path; not a checkpointed parameter
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.context_length, config.context_length))
                    .view(1, 1, config.context_length, config.context_length),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nn.Module.__call__ is typed as -> Any; annotate so Tensor methods (split/view/...) resolve.
        qkv: torch.Tensor = self.qkv_proj(x)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            attn_weights: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn_weights = attn_weights.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            y = attn_weights @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):

    up_proj: nn.Linear
    gelu: nn.GELU
    down_proj: nn.Linear
    resid_dropout: nn.Dropout

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        x = self.resid_dropout(x)
        return x


class Block(nn.Module):

    ln1: LayerNorm
    attn: CausalSelfAttention
    ln2: LayerNorm
    mlp: MLP

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Token/position embed + Block stack + final LayerNorm."""

    tok_emb: nn.Embedding
    pos_emb: nn.Embedding
    emb_dropout: nn.Dropout
    blocks: nn.ModuleList
    ln_final: LayerNorm

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = LayerNorm(config.n_embd, bias=config.bias)

    def iter_blocks(self) -> Iterator[Block]:
        # ModuleList.__iter__ is typed as Iterator[Module]; narrow for attribute navigation.
        for module in self.blocks:
            if not isinstance(module, Block):
                raise TypeError(f"expected Block, got {type(module)}")
            yield module


class GPT(nn.Module):

    config: GPTConfig
    transformer: Transformer
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config

        # decoder-only GPT stack (data flow):
        #   idx --+--> tok_emb ----+
        #         |                +--> (+) --> emb_dropout --> blocks[0..n) --> ln_final --> lm_head --> logits
        #   pos --+--> pos_emb ----+
        # tok_emb: token embedding  (vocab_size -> n_embd)
        # pos_emb: position embedding (context_length -> n_embd)
        # blocks:  n_layer transformer Blocks (attn + mlp, residual)
        # ln_final: final LayerNorm before the language-model head
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # (n_embd -> vocab_size)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.tok_emb.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, context_length is only {self.config.context_length}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)

        # forward the GPT model itself
        tok_emb: torch.Tensor = self.transformer.tok_emb(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb: torch.Tensor = self.transformer.pos_emb(pos) # position embeddings of shape (T, n_embd)
        x: torch.Tensor = self.transformer.emb_dropout(tok_emb + pos_emb)
        for block in self.transformer.iter_blocks():
            x = block(x)
        x = self.transformer.ln_final(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits: torch.Tensor = self.lm_head(x)
            loss: Optional[torch.Tensor] = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_context_length(self, context_length: int) -> None:
        # model surgery to decrease the context length if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (context_length 1024)
        # but want to use a smaller context for some smaller, simpler model
        assert context_length <= self.config.context_length
        self.config.context_length = context_length
        self.transformer.pos_emb.weight = nn.Parameter(self.transformer.pos_emb.weight[:context_length])
        for block in self.transformer.iter_blocks():
            if hasattr(block.attn, 'causal_mask'):
                block.attn.causal_mask = block.attn.causal_mask[:,:,:context_length,:context_length]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None) -> GPT:
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args: dict = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, context_length=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['context_length'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)

        # HF weights are HF_GPT2 format; convert once at the boundary (see gpt2_weights.py)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        native = convert_gpt2_state_dict(
            model_hf.state_dict(),
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            weight_format=WeightFormat.HF_GPT2,
        )
        model.load_state_dict(native)
        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.context_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            # call .forward (not self(...)): Module.__call__ is stubbed as -> Any
            logits, _ = self.forward(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
