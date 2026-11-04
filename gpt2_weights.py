"""
GPT-2 ecosystem weight formats → this repo's native GPT state_dict.

Three parameter-key / layout formats appear in practice:

1. HF_GPT2 — HuggingFace / OpenAI naming (`wte`, `h`, `c_attn`, …) with Conv1D
   weight layout `(in_features, out_features)`.
2. NANOGPT_LEGACY — same GPT-2 naming, but tensors already stored as `nn.Linear`
   layout `(out_features, in_features)` (original nanoGPT checkpoints).
3. NATIVE — this repo's module names (`tok_emb`, `layers`, `qkv_proj`,
   `up_proj`/`down_proj`, …) and Linear layout.

Conversion is an explicit format boundary (see `WeightFormat`), not a string-replace
shim inside the model. Train / sample / resume only load NATIVE state_dicts.
"""

from __future__ import annotations

import argparse
from enum import Enum, auto
from typing import Dict, Iterator, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor

_COMPILE_PREFIX = "_orig_mod."
_LEGACY_BLOCKS_PREFIX = "transformer.blocks."
_LAYERS_PREFIX = "transformer.layers."


class WeightFormat(Enum):
    NATIVE = auto()
    HF_GPT2 = auto()          # GPT-2 names + Conv1D weight layout
    NANOGPT_LEGACY = auto()   # GPT-2 names + Linear weight layout


def unwrap_compiled_state_dict(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Undo torch.compile's `_orig_mod.` key prefix when the whole dict uses it."""
    if not state_dict:
        return {}
    flags = [k.startswith(_COMPILE_PREFIX) for k in state_dict]
    if all(flags):
        n = len(_COMPILE_PREFIX)
        return {k[n:]: v for k, v in state_dict.items()}
    if any(flags):
        raise ValueError(
            "state_dict mixes `_orig_mod.`-prefixed and bare keys; "
            "save the uncompiled module (raw_model.state_dict()) or unwrap uniformly"
        )
    return dict(state_dict)


def normalize_native_state_dict(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Unwrap compile prefix and rename legacy transformer.blocks.* → layers.*."""
    sd = unwrap_compiled_state_dict(state_dict)
    has_blocks = any(k.startswith(_LEGACY_BLOCKS_PREFIX) for k in sd)
    has_layers = any(k.startswith(_LAYERS_PREFIX) for k in sd)
    if has_blocks and has_layers:
        raise ValueError("state_dict mixes transformer.blocks.* and transformer.layers.*")
    if not has_blocks:
        return sd
    out: Dict[str, Tensor] = {}
    n = len(_LEGACY_BLOCKS_PREFIX)
    for k, v in sd.items():
        if k.startswith(_LEGACY_BLOCKS_PREFIX):
            out[_LAYERS_PREFIX + k[n:]] = v
        else:
            out[k] = v
    return out


def iter_gpt2_named_param_specs(n_layer: int, n_embd: int) -> Iterator[Tuple[str, str, int, int]]:
    """
    Yield (gpt2_key, native_key, linear_out, linear_in) for every learnable parameter.

    `linear_out` / `linear_in` are the destination nn.Linear weight shapes
    (and ignored for biases / embeddings / LayerNorm, where out=in=0 sentinel).
    """
    if n_layer < 1:
        raise ValueError(f"n_layer must be >= 1, got {n_layer}")
    if n_embd < 1:
        raise ValueError(f"n_embd must be >= 1, got {n_embd}")

    def emb(src: str, dst: str) -> Tuple[str, str, int, int]:
        return (src, dst, 0, 0)

    yield emb("transformer.wte.weight", "transformer.tok_emb.weight")
    yield emb("transformer.wpe.weight", "transformer.pos_emb.weight")
    yield emb("transformer.ln_f.weight", "transformer.ln_final.weight")
    yield emb("transformer.ln_f.bias", "transformer.ln_final.bias")
    yield emb("lm_head.weight", "lm_head.weight")

    for i in range(n_layer):
        g = f"transformer.h.{i}"
        n = f"transformer.layers.{i}"
        yield emb(f"{g}.ln_1.weight", f"{n}.ln1.weight")
        yield emb(f"{g}.ln_1.bias", f"{n}.ln1.bias")
        yield (f"{g}.attn.c_attn.weight", f"{n}.attn.qkv_proj.weight", 3 * n_embd, n_embd)
        yield emb(f"{g}.attn.c_attn.bias", f"{n}.attn.qkv_proj.bias")
        yield (f"{g}.attn.c_proj.weight", f"{n}.attn.out_proj.weight", n_embd, n_embd)
        yield emb(f"{g}.attn.c_proj.bias", f"{n}.attn.out_proj.bias")
        yield emb(f"{g}.ln_2.weight", f"{n}.ln2.weight")
        yield emb(f"{g}.ln_2.bias", f"{n}.ln2.bias")
        yield (f"{g}.mlp.c_fc.weight", f"{n}.mlp.up_proj.weight", 4 * n_embd, n_embd)
        yield emb(f"{g}.mlp.c_fc.bias", f"{n}.mlp.up_proj.bias")
        yield (f"{g}.mlp.c_proj.weight", f"{n}.mlp.down_proj.weight", n_embd, 4 * n_embd)
        yield emb(f"{g}.mlp.c_proj.bias", f"{n}.mlp.down_proj.bias")


def _is_skippable_gpt2_buffer(key: str) -> bool:
    return key.endswith(".attn.bias") or key.endswith(".attn.masked_bias")


def _copy_or_transpose_weight(
    tensor: Tensor,
    *,
    src_key: str,
    linear_out: int,
    linear_in: int,
    weight_format: WeightFormat,
) -> Tensor:
    """Map a 2D GPT-2-named weight onto nn.Linear (out, in) for the given format."""
    if linear_out == 0:
        return tensor

    linear_shape = (linear_out, linear_in)
    conv1d_shape = (linear_in, linear_out)
    shape = tuple(tensor.shape)

    if weight_format is WeightFormat.NANOGPT_LEGACY:
        if shape != linear_shape:
            raise ValueError(
                f"{src_key}: NANOGPT_LEGACY expects Linear shape {linear_shape}, got {shape}"
            )
        return tensor

    if weight_format is WeightFormat.HF_GPT2:
        if shape != conv1d_shape:
            raise ValueError(
                f"{src_key}: HF_GPT2 expects Conv1D shape {conv1d_shape}, got {shape}"
            )
        return tensor.t().contiguous()

    raise ValueError(f"cannot convert weights with format {weight_format}")


def detect_weight_format(state_dict: Mapping[str, Tensor], *, n_embd: int) -> WeightFormat:
    """
    Classify a state_dict by key schema and (for GPT-2 names) weight layout.

    Layout is read from `transformer.h.0.attn.c_attn.weight`, whose Linear shape is
    `(3 * n_embd, n_embd)` and Conv1D shape is `(n_embd, 3 * n_embd)`.
    """
    sd = unwrap_compiled_state_dict(state_dict)
    has_native = "transformer.tok_emb.weight" in sd or any(".attn.qkv_proj." in k for k in sd)
    has_gpt2 = "transformer.wte.weight" in sd or any(".attn.c_attn." in k for k in sd)

    if has_native and has_gpt2:
        raise ValueError("state_dict mixes native and GPT-2 parameter names")
    if has_native:
        return WeightFormat.NATIVE
    if not has_gpt2:
        raise ValueError(
            "state_dict is neither NATIVE (tok_emb/qkv_proj) nor GPT-2-named (wte/c_attn)"
        )

    probe = "transformer.h.0.attn.c_attn.weight"
    if probe not in sd:
        raise KeyError(f"GPT-2-named state_dict missing layout probe key {probe!r}")
    shape = tuple(sd[probe].shape)
    linear_shape = (3 * n_embd, n_embd)
    conv1d_shape = (n_embd, 3 * n_embd)
    if shape == linear_shape:
        return WeightFormat.NANOGPT_LEGACY
    if shape == conv1d_shape:
        return WeightFormat.HF_GPT2
    raise ValueError(
        f"{probe}: shape {shape} matches neither Linear {linear_shape} "
        f"nor Conv1D {conv1d_shape} for n_embd={n_embd}"
    )


def convert_gpt2_state_dict(
    gpt2_state_dict: Mapping[str, Tensor],
    *,
    n_layer: int,
    n_embd: int,
    weight_format: Optional[WeightFormat] = None,
) -> Dict[str, Tensor]:
    """
    Convert HF_GPT2 or NANOGPT_LEGACY parameters into NATIVE keys + Linear layout.

    `weight_format` may be omitted; it is then detected from keys and the c_attn probe.
    NATIVE input is rejected — caller's job is not to "convert" what is already native.
    """
    src = unwrap_compiled_state_dict(gpt2_state_dict)
    fmt = weight_format if weight_format is not None else detect_weight_format(src, n_embd=n_embd)
    if fmt is WeightFormat.NATIVE:
        raise ValueError("convert_gpt2_state_dict expects HF_GPT2 or NANOGPT_LEGACY, not NATIVE")
    if fmt not in (WeightFormat.HF_GPT2, WeightFormat.NANOGPT_LEGACY):
        raise ValueError(f"unsupported format for conversion: {fmt}")

    out: Dict[str, Tensor] = {}
    used: set[str] = set()

    for src_key, dst_key, linear_out, linear_in in iter_gpt2_named_param_specs(n_layer, n_embd):
        if src_key not in src:
            raise KeyError(f"missing GPT-2 parameter {src_key!r} (n_layer={n_layer})")
        tensor = src[src_key]
        if linear_out != 0:
            tensor = _copy_or_transpose_weight(
                tensor,
                src_key=src_key,
                linear_out=linear_out,
                linear_in=linear_in,
                weight_format=fmt,
            )
        out[dst_key] = tensor
        used.add(src_key)

    leftover = [k for k in src if k not in used and not _is_skippable_gpt2_buffer(k)]
    if leftover:
        preview = ", ".join(repr(k) for k in sorted(leftover)[:8])
        more = "" if len(leftover) <= 8 else f", ... ({len(leftover)} total)"
        raise KeyError(f"unconverted GPT-2 keys: {preview}{more}")

    return out


def migrate_checkpoint(
    ckpt: MutableMapping[str, object],
    *,
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    weight_format: Optional[WeightFormat] = None,
) -> MutableMapping[str, object]:
    """
    In-place: ensure ckpt['model'] is NATIVE.

    NATIVE → unwrap compile prefix only.
    HF_GPT2 / NANOGPT_LEGACY → full convert via `convert_gpt2_state_dict`.
    """
    model_sd = ckpt["model"]
    if not isinstance(model_sd, Mapping):
        raise TypeError(f"ckpt['model'] must be a mapping, got {type(model_sd)}")

    model_args = ckpt.get("model_args")
    if n_layer is None or n_embd is None:
        if not isinstance(model_args, Mapping):
            raise KeyError("n_layer/n_embd not given and ckpt['model_args'] missing")
        if n_layer is None:
            n_layer = int(model_args["n_layer"])
        if n_embd is None:
            n_embd = int(model_args["n_embd"])

    fmt = weight_format if weight_format is not None else detect_weight_format(model_sd, n_embd=n_embd)
    if fmt is WeightFormat.NATIVE:
        ckpt["model"] = normalize_native_state_dict(model_sd)
        return ckpt

    ckpt["model"] = convert_gpt2_state_dict(
        model_sd,
        n_layer=n_layer,
        n_embd=n_embd,
        weight_format=fmt,
    )
    return ckpt


def _cli_migrate(path: str, output: Optional[str]) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, MutableMapping):
        raise TypeError(f"expected checkpoint mapping in {path}")
    model_args = ckpt["model_args"]
    assert isinstance(model_args, Mapping)
    n_embd = int(model_args["n_embd"])
    before = detect_weight_format(ckpt["model"], n_embd=n_embd)  # type: ignore[arg-type]
    migrate_checkpoint(ckpt)
    after = detect_weight_format(ckpt["model"], n_embd=n_embd)  # type: ignore[arg-type]
    out_path = output or path
    torch.save(ckpt, out_path)
    print(f"wrote {out_path} ({before.name} -> {after.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    mig = sub.add_parser("migrate", help="convert a ckpt.pt model dict to NATIVE names/layout")
    mig.add_argument("checkpoint", help="path to ckpt.pt")
    mig.add_argument("-o", "--output", default=None, help="output path (default: overwrite input)")

    args = parser.parse_args()
    if args.cmd == "migrate":
        _cli_migrate(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
