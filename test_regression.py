"""
Regression tests for the rename + GPT-2 weight-format boundary.

Run from repo root:
  python -m unittest test_regression.py -v
"""

from __future__ import annotations

import copy
import os
import unittest
from typing import Dict

import torch
from torch import Tensor

from gpt2_weights import (
    WeightFormat,
    convert_gpt2_state_dict,
    detect_weight_format,
    iter_gpt2_named_param_specs,
    migrate_checkpoint,
    unwrap_compiled_state_dict,
)
from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_NATIVE = os.path.join(ROOT, "out-gpt2", "ckpt.pt")
CKPT_LEGACY_BAK = os.path.join(ROOT, "out-gpt2", "ckpt.pt.gpt2-format.bak")


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        context_length=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )


def _native_to_gpt2_named(
    native: Dict[str, Tensor],
    *,
    n_layer: int,
    n_embd: int,
    as_hf_conv1d: bool,
) -> Dict[str, Tensor]:
    """Inverse of convert: NATIVE → GPT-2 names (optionally Conv1D layout)."""
    out: Dict[str, Tensor] = {}
    for gpt2_key, native_key, linear_out, linear_in in iter_gpt2_named_param_specs(n_layer, n_embd):
        tensor = native[native_key]
        if linear_out != 0 and as_hf_conv1d:
            tensor = tensor.t().contiguous()
        out[gpt2_key] = tensor
    return out


class TestUnwrapCompiled(unittest.TestCase):
    def test_uniform_prefix(self):
        sd = {"_orig_mod.a": torch.tensor(1.0), "_orig_mod.b": torch.tensor(2.0)}
        out = unwrap_compiled_state_dict(sd)
        self.assertEqual(set(out), {"a", "b"})

    def test_no_prefix(self):
        sd = {"a": torch.tensor(1.0)}
        self.assertEqual(unwrap_compiled_state_dict(sd), sd)

    def test_mixed_prefix_rejected(self):
        sd = {"_orig_mod.a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        with self.assertRaises(ValueError):
            unwrap_compiled_state_dict(sd)


class TestFormatDetectAndConvert(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _tiny_config()
        self.model = GPT(self.cfg)
        self.native = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def test_detect_native(self):
        self.assertIs(
            detect_weight_format(self.native, n_embd=self.cfg.n_embd),
            WeightFormat.NATIVE,
        )

    def test_nanogpt_legacy_roundtrip_tensors(self):
        legacy = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=False
        )
        self.assertIs(
            detect_weight_format(legacy, n_embd=self.cfg.n_embd),
            WeightFormat.NANOGPT_LEGACY,
        )
        back = convert_gpt2_state_dict(
            legacy,
            n_layer=self.cfg.n_layer,
            n_embd=self.cfg.n_embd,
            weight_format=WeightFormat.NANOGPT_LEGACY,
        )
        self.assertEqual(set(back), set(self.native))
        for k in back:
            self.assertTrue(torch.equal(back[k], self.native[k]), k)

    def test_hf_gpt2_roundtrip_tensors(self):
        hf = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=True
        )
        self.assertIs(detect_weight_format(hf, n_embd=self.cfg.n_embd), WeightFormat.HF_GPT2)
        # probe must be Conv1D-shaped
        self.assertEqual(
            tuple(hf["transformer.h.0.attn.c_attn.weight"].shape),
            (self.cfg.n_embd, 3 * self.cfg.n_embd),
        )
        back = convert_gpt2_state_dict(
            hf,
            n_layer=self.cfg.n_layer,
            n_embd=self.cfg.n_embd,
            weight_format=WeightFormat.HF_GPT2,
        )
        self.assertEqual(set(back), set(self.native))
        for k in back:
            self.assertTrue(torch.equal(back[k], self.native[k]), k)

    def test_convert_rejects_native(self):
        with self.assertRaises(ValueError):
            convert_gpt2_state_dict(
                self.native,
                n_layer=self.cfg.n_layer,
                n_embd=self.cfg.n_embd,
            )

    def test_wrong_layout_for_declared_format(self):
        legacy = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=False
        )
        with self.assertRaises(ValueError):
            convert_gpt2_state_dict(
                legacy,
                n_layer=self.cfg.n_layer,
                n_embd=self.cfg.n_embd,
                weight_format=WeightFormat.HF_GPT2,
            )

    def test_missing_key_rejected(self):
        legacy = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=False
        )
        del legacy["transformer.wte.weight"]
        with self.assertRaises(KeyError):
            convert_gpt2_state_dict(
                legacy,
                n_layer=self.cfg.n_layer,
                n_embd=self.cfg.n_embd,
                weight_format=WeightFormat.NANOGPT_LEGACY,
            )

    def test_extra_key_rejected(self):
        legacy = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=False
        )
        legacy["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(KeyError):
            convert_gpt2_state_dict(
                legacy,
                n_layer=self.cfg.n_layer,
                n_embd=self.cfg.n_embd,
                weight_format=WeightFormat.NANOGPT_LEGACY,
            )

    def test_attn_bias_buffer_ignored(self):
        legacy = _native_to_gpt2_named(
            self.native, n_layer=self.cfg.n_layer, n_embd=self.cfg.n_embd, as_hf_conv1d=False
        )
        legacy["transformer.h.0.attn.bias"] = torch.ones(1, 1, 8, 8)
        back = convert_gpt2_state_dict(
            legacy,
            n_layer=self.cfg.n_layer,
            n_embd=self.cfg.n_embd,
            weight_format=WeightFormat.NANOGPT_LEGACY,
        )
        self.assertNotIn("transformer.blocks.0.attn.causal_mask", back)
        self.assertEqual(set(back), set(self.native))


class TestModelInvariants(unittest.TestCase):
    def test_causal_mask_not_checkpointed(self):
        cfg = _tiny_config()
        # force slow path buffer registration by monkeypatching flash off after init is hard;
        # instead construct attention alone conceptually: if flash exists, mask may be absent.
        model = GPT(cfg)
        keys = model.state_dict().keys()
        self.assertFalse(any(k.endswith("causal_mask") for k in keys))

    def test_weight_tying(self):
        model = GPT(_tiny_config())
        self.assertIs(model.transformer.tok_emb.weight, model.lm_head.weight)

    def test_legacy_convert_load_matches_native_logits(self):
        torch.manual_seed(1)
        cfg = _tiny_config()
        model_a = GPT(cfg)
        native = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
        legacy = _native_to_gpt2_named(
            native, n_layer=cfg.n_layer, n_embd=cfg.n_embd, as_hf_conv1d=False
        )
        model_b = GPT(cfg)
        model_b.load_state_dict(
            convert_gpt2_state_dict(
                legacy,
                n_layer=cfg.n_layer,
                n_embd=cfg.n_embd,
                weight_format=WeightFormat.NANOGPT_LEGACY,
            )
        )
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            la, _ = model_a(x)
            lb, _ = model_b(x)
        self.assertTrue(torch.equal(la, lb))

    def test_hf_convert_load_matches_native_logits(self):
        torch.manual_seed(2)
        cfg = _tiny_config()
        model_a = GPT(cfg)
        native = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
        hf = _native_to_gpt2_named(
            native, n_layer=cfg.n_layer, n_embd=cfg.n_embd, as_hf_conv1d=True
        )
        model_b = GPT(cfg)
        model_b.load_state_dict(
            convert_gpt2_state_dict(
                hf,
                n_layer=cfg.n_layer,
                n_embd=cfg.n_embd,
                weight_format=WeightFormat.HF_GPT2,
            )
        )
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            la, _ = model_a(x)
            lb, _ = model_b(x)
        self.assertTrue(torch.equal(la, lb))

    def test_migrate_checkpoint_roundtrip(self):
        torch.manual_seed(3)
        cfg = _tiny_config()
        model = GPT(cfg)
        native = {k: v.detach().clone() for k, v in model.state_dict().items()}
        legacy = _native_to_gpt2_named(
            native, n_layer=cfg.n_layer, n_embd=cfg.n_embd, as_hf_conv1d=False
        )
        ckpt = {
            "model": legacy,
            "model_args": {
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "context_length": cfg.context_length,
                "bias": cfg.bias,
                "vocab_size": cfg.vocab_size,
                "dropout": cfg.dropout,
            },
        }
        migrate_checkpoint(ckpt)
        self.assertIs(
            detect_weight_format(ckpt["model"], n_embd=cfg.n_embd),
            WeightFormat.NATIVE,
        )
        # second migrate is a no-op on keys
        again = copy.deepcopy(ckpt)
        migrate_checkpoint(again)
        for k in ckpt["model"]:
            self.assertTrue(torch.equal(ckpt["model"][k], again["model"][k]), k)


@unittest.skipUnless(
    os.path.isfile(CKPT_NATIVE) and os.path.isfile(CKPT_LEGACY_BAK),
    "out-gpt2 native + legacy bak checkpoints required",
)
class TestRealGpt2Checkpoint(unittest.TestCase):
    """End-to-end numerical lock against the real GPT-2 124M weights on disk."""

    @classmethod
    def setUpClass(cls):
        cls.native_ckpt = torch.load(CKPT_NATIVE, map_location="cpu", weights_only=False)
        cls.legacy_ckpt = torch.load(CKPT_LEGACY_BAK, map_location="cpu", weights_only=False)
        cls.model_args = cls.native_ckpt["model_args"]
        cls.cfg = GPTConfig.from_model_args(cls.model_args)

    def test_disk_formats(self):
        n_embd = self.model_args["n_embd"]
        self.assertIs(
            detect_weight_format(self.native_ckpt["model"], n_embd=n_embd),
            WeightFormat.NATIVE,
        )
        self.assertIs(
            detect_weight_format(self.legacy_ckpt["model"], n_embd=n_embd),
            WeightFormat.NANOGPT_LEGACY,
        )

    def test_legacy_bak_converts_to_native_bitexact(self):
        converted = convert_gpt2_state_dict(
            self.legacy_ckpt["model"],
            n_layer=self.model_args["n_layer"],
            n_embd=self.model_args["n_embd"],
            weight_format=WeightFormat.NANOGPT_LEGACY,
        )
        self.assertEqual(set(converted), set(self.native_ckpt["model"]))
        for k in converted:
            self.assertTrue(
                torch.equal(converted[k], self.native_ckpt["model"][k]),
                f"mismatch after migrate: {k}",
            )

    def test_native_load_and_logits_match_legacy_convert(self):
        torch.manual_seed(42)
        x = torch.randint(0, self.cfg.vocab_size, (1, 16))

        model_native = GPT(self.cfg)
        model_native.load_state_dict(self.native_ckpt["model"])
        model_native.eval()

        model_from_legacy = GPT(self.cfg)
        model_from_legacy.load_state_dict(
            convert_gpt2_state_dict(
                self.legacy_ckpt["model"],
                n_layer=self.cfg.n_layer,
                n_embd=self.cfg.n_embd,
                weight_format=WeightFormat.NANOGPT_LEGACY,
            )
        )
        model_from_legacy.eval()

        with torch.no_grad():
            logits_a, _ = model_native(x)
            logits_b, _ = model_from_legacy(x)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertEqual(tuple(logits_a.shape), (1, 1, self.cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
