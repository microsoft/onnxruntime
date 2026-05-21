# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for FusedAdam CPU fallback (issue #17403).

These tests patch torch.cuda.is_available to return False so they run
deterministically on both CPU-only and CUDA machines.

Import strategy:
  * Load ``_multi_tensor_apply.py`` directly from the optim/ source
    directory via ``importlib.util.spec_from_file_location`` (it is pure
    Python with no external deps). ``sys.path`` is not modified.
  * Pre-register that module in ``sys.modules`` under a synthetic package
    name so the relative ``from ._multi_tensor_apply import ...`` inside
    ``fused_adam.py`` can resolve.
  * Load ``fused_adam.py`` via ``importlib.util.spec_from_file_location``
    with ``__package__`` set to that synthetic package name so the
    relative import binds to the entry from the previous step.

This avoids touching the training/__init__.py which requires the compiled
onnxruntime C extension (not available in the source-tree environment).
The CUDA extension import inside fused_adam.__init__ is guarded by
``if torch.cuda.is_available():`` and never runs with the mock in place.
"""

import importlib.util
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Locate the optim source directory.
# File layout:
#   orttraining/orttraining/test/python/   <- __file__  (parents[0])
#   orttraining/orttraining/test/          <- parents[1]
#   orttraining/orttraining/               <- parents[2]
#   orttraining/orttraining/python/training/optim/  <- _OPTIM_DIR
# ---------------------------------------------------------------------------
_OPTIM_DIR = Path(__file__).resolve().parents[2] / "python" / "training" / "optim"
assert _OPTIM_DIR.is_dir(), f"optim dir not found: {_OPTIM_DIR}"

# Step 1: load _multi_tensor_apply as a top-level module (no package needed,
# it has zero external imports) and register it under the name that the
# relative import inside fused_adam.py expects.
_PKG = "fused_adam_pkg"
_mta_spec = importlib.util.spec_from_file_location(
    f"{_PKG}._multi_tensor_apply",
    _OPTIM_DIR / "_multi_tensor_apply.py",
)
_mta_mod = importlib.util.module_from_spec(_mta_spec)
sys.modules[f"{_PKG}._multi_tensor_apply"] = _mta_mod
_mta_spec.loader.exec_module(_mta_mod)

# Step 2: load fused_adam.py with __package__ = _PKG so its relative import
# "from ._multi_tensor_apply import ..." resolves to the entry above.
_fa_spec = importlib.util.spec_from_file_location(
    f"{_PKG}.fused_adam",
    _OPTIM_DIR / "fused_adam.py",
)
_fa_mod = importlib.util.module_from_spec(_fa_spec)
_fa_mod.__package__ = _PKG
_fa_spec.loader.exec_module(_fa_mod)

AdamWMode = _fa_mod.AdamWMode
FusedAdam = _fa_mod.FusedAdam


def _make_param(shape=(3, 3)):
    """Return an nn.Parameter with a synthetic gradient."""
    p = nn.Parameter(torch.randn(*shape))
    p.grad = torch.randn(*shape)
    return p


@patch("torch.cuda.is_available", return_value=False)
class TestFusedAdamCpuFallback(unittest.TestCase):
    """All tests run with CUDA disabled to exercise the CPU fallback path."""

    def test_instantiation_warns_and_succeeds(self, _mock_cuda):
        """FusedAdam must instantiate without error and emit a UserWarning."""
        param = nn.Parameter(torch.randn(3, 3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = FusedAdam([param], lr=1e-3)

        self.assertIsNotNone(opt, "FusedAdam should instantiate on CPU")
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreaterEqual(len(user_warnings), 1, "Expected at least one UserWarning about CPU fallback")
        messages = [str(w.message).lower() for w in user_warnings]
        self.assertTrue(
            any("cuda" in m and ("fallback" in m or "falling back" in m) for m in messages),
            f"Expected a warning mentioning 'cuda' and 'fallback'/'falling back', got: {messages}",
        )

    def test_step_updates_params_like_adamw(self, _mock_cuda):
        """After one step, params must change in the same direction as torch.optim.AdamW."""
        torch.manual_seed(42)
        weight_init = torch.randn(4, 4)
        grad = torch.randn(4, 4)

        # FusedAdam (CPU fallback) path — ADAMW_TORCH maps to torch.optim.AdamW
        p_fused = nn.Parameter(weight_init.clone())
        p_fused.grad = grad.clone()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt_fused = FusedAdam([p_fused], lr=1e-3, adam_w_mode=AdamWMode.ADAMW_TORCH)
        opt_fused.step()

        # Reference: plain torch.optim.AdamW with matching hyperparams
        p_ref = nn.Parameter(weight_init.clone())
        p_ref.grad = grad.clone()
        opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)
        opt_ref.step()

        self.assertFalse(
            torch.allclose(p_fused.data, weight_init),
            "Parameters should have changed after step",
        )
        self.assertTrue(
            torch.allclose(p_fused.data, p_ref.data, atol=1e-5),
            f"FusedAdam CPU fallback should match torch.optim.AdamW.\n"
            f"Max diff: {(p_fused.data - p_ref.data).abs().max().item()}",
        )

    def test_adam_l2_mode_instantiates_and_steps(self, _mock_cuda):
        """AdamWMode.ADAM_L2_REGULARIZATION must instantiate and step without error."""
        param = _make_param()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = FusedAdam([param], lr=1e-3, adam_w_mode=AdamWMode.ADAM_L2_REGULARIZATION)

        before = param.data.clone()
        opt.step()
        self.assertFalse(torch.allclose(param.data, before), "Parameters should change after step")

    def test_bias_correction_false_adamw_torch_raises(self, _mock_cuda):
        """FusedAdam with ADAMW_TORCH and bias_correction=False must raise RuntimeError on CPU."""
        param = nn.Parameter(torch.randn(3, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as ctx:
                FusedAdam([param], lr=1e-3, adam_w_mode=AdamWMode.ADAMW_TORCH, bias_correction=False)
        self.assertIn("bias_correction", str(ctx.exception))

    def test_bias_correction_false_adam_l2_raises(self, _mock_cuda):
        """FusedAdam with ADAM_L2_REGULARIZATION and bias_correction=False must raise RuntimeError on CPU."""
        param = nn.Parameter(torch.randn(3, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as ctx:
                FusedAdam([param], lr=1e-3, adam_w_mode=AdamWMode.ADAM_L2_REGULARIZATION, bias_correction=False)
        self.assertIn("bias_correction", str(ctx.exception))

    def test_bias_correction_false_adamw_transformers_raises(self, _mock_cuda):
        """FusedAdam with ADAMW_TRANSFORMERS and bias_correction=False must raise RuntimeError on CPU.

        The unified guard fires before the try/except transformers import, so this
        works regardless of whether the transformers package is installed.
        """
        param = nn.Parameter(torch.randn(3, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as ctx:
                FusedAdam([param], lr=1e-3, adam_w_mode=AdamWMode.ADAMW_TRANSFORMERS, bias_correction=False)
        self.assertIn("bias_correction", str(ctx.exception))

    def test_per_group_lr_dicts(self, _mock_cuda):
        """Per-group lr specified via param dicts must be preserved in the fallback optimizer."""
        p1 = nn.Parameter(torch.randn(2, 2))
        p2 = nn.Parameter(torch.randn(2, 2))
        param_groups = [
            {"params": [p1], "lr": 0.5},
            {"params": [p2], "lr": 0.1},
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = FusedAdam(param_groups, lr=1e-3)

        lrs = [g["lr"] for g in opt.param_groups]
        self.assertEqual(lrs[0], 0.5, f"Expected first group lr=0.5, got {lrs[0]}")
        self.assertEqual(lrs[1], 0.1, f"Expected second group lr=0.1, got {lrs[1]}")

    def test_state_dict_round_trip(self, _mock_cuda):
        """state_dict / load_state_dict must survive a round trip on the CPU fallback path."""
        param = _make_param()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = FusedAdam([param], lr=1e-3)

        # Run one step to populate optimizer state.
        opt.step()
        sd = opt.state_dict()

        # Fresh optimizer, load the saved state.
        param2 = _make_param()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt2 = FusedAdam([param2], lr=1e-3)
        opt2.load_state_dict(sd)

        sd2 = opt2.state_dict()
        # The 'state' entries (exp_avg, exp_avg_sq) must be present after load.
        self.assertEqual(len(sd["state"]), len(sd2["state"]), "State dict round trip should preserve all state entries")


if __name__ == "__main__":
    unittest.main()
