# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rewrite rules for fusing decomposed LayerNorm into LayerNormalization.

Models exported from older ONNX opsets or other frameworks may represent
LayerNormalization as a chain of primitive ops:

.. code-block:: text

    mean = ReduceMean(x, axes=[-1])
    diff = Sub(x, mean)
    sq = Pow(diff, 2)
    var = ReduceMean(sq, axes=[-1])
    var_eps = Add(var, epsilon)
    std = Sqrt(var_eps)
    normalized = Div(diff, std)
    scaled = Mul(normalized, weight)
    result = Add(scaled, bias)            # optional

This module fuses that chain into a single ``LayerNormalization`` op.

These rules are **not applied by default**.  Apply them post-export::

    from onnx_ir_fusions import layer_norm_fusion_rules
    from onnxscript.rewriter import rewrite

    rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())
"""

from __future__ import annotations

import math

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)


def _check_constant_scalar(value, expected: float, name: str, *, rel_tol: float = 1e-4) -> str | None:
    """Return an error message if *value* is not a constant ≈ *expected*."""
    if value.const_value is None:
        return f"{name} is not a constant"
    actual = float(value.const_value.numpy().flat[0])
    if not math.isclose(actual, expected, rel_tol=rel_tol):
        return f"{name} is {actual}, expected {expected}"
    return None


def _check_axes_minus_one(value, name: str) -> str | None:
    """Return an error message if *value* is not the constant ``[-1]``."""
    if value.const_value is None:
        return f"ReduceMean {name} axes is not a constant"
    axes = list(value.const_value.numpy().flat)
    if axes != [-1]:
        return f"ReduceMean {name} axes={axes}, expected [-1]"
    return None


class LayerNormFusion(RewriteRuleClassBase):
    """Fuse decomposed LayerNorm (with bias) into LayerNormalization.

    **Matched pattern (opset ≥ 18, axes as input):**

    .. code-block:: text

        mean = ReduceMean(x, axes=[-1])
        diff = Sub(x, mean)
        sq = Pow(diff, 2)
        var = ReduceMean(sq, axes=[-1])
        var_eps = Add(var, epsilon)
        std = Sqrt(var_eps)
        normalized = Div(diff, std)
        scaled = Mul(normalized, weight)
        result = Add(scaled, bias)

    **Replacement:**

    .. code-block:: text

        result = LayerNormalization(x, weight, bias, epsilon=eps, axis=-1)
    """

    def pattern(self, op, x, axes1, exponent, axes2, epsilon, weight, bias):
        mean = op.ReduceMean(x, axes1, _allow_other_attributes=True)
        diff = op.Sub(x, mean)
        sq = op.Pow(diff, exponent)
        var = op.ReduceMean(sq, axes2, _allow_other_attributes=True)
        var_eps = op.Add(var, epsilon)
        std = op.Sqrt(var_eps)
        normalized = op.Div(diff, std)
        scaled = op.Mul(normalized, weight)
        return op.Add(scaled, bias)

    def check(self, context, exponent, epsilon, axes1, axes2, **_):
        result = MatchResult()

        err = _check_constant_scalar(exponent, 2.0, "Pow exponent")
        if err:
            return result.fail(err)

        if epsilon.const_value is None:
            return result.fail("Epsilon is not a constant")
        eps_val = float(epsilon.const_value.numpy().flat[0])
        if eps_val <= 0 or eps_val > 1.0:
            return result.fail(f"Epsilon {eps_val} out of expected range (0, 1]")

        for tag, axes in [("first", axes1), ("second", axes2)]:
            err = _check_axes_minus_one(axes, tag)
            if err:
                return result.fail(err)

        return result

    def rewrite(self, op, x, weight, bias, epsilon, **_):
        eps_val = float(epsilon.const_value.numpy().flat[0])
        return op.LayerNormalization(x, weight, bias, axis=-1, epsilon=eps_val)


class LayerNormFusionNoBias(RewriteRuleClassBase):
    """Fuse decomposed LayerNorm (without bias) into LayerNormalization.

    Same as :class:`LayerNormFusion` but the pattern ends at the ``Mul``
    (scale) step — there is no final ``Add(bias)``.

    **Replacement:**

    .. code-block:: text

        result = LayerNormalization(x, weight, epsilon=eps, axis=-1)
    """

    def pattern(self, op, x, axes1, exponent, axes2, epsilon, weight):
        mean = op.ReduceMean(x, axes1, _allow_other_attributes=True)
        diff = op.Sub(x, mean)
        sq = op.Pow(diff, exponent)
        var = op.ReduceMean(sq, axes2, _allow_other_attributes=True)
        var_eps = op.Add(var, epsilon)
        std = op.Sqrt(var_eps)
        normalized = op.Div(diff, std)
        return op.Mul(normalized, weight)

    def check(self, context, exponent, epsilon, axes1, axes2, **_):
        result = MatchResult()

        err = _check_constant_scalar(exponent, 2.0, "Pow exponent")
        if err:
            return result.fail(err)

        if epsilon.const_value is None:
            return result.fail("Epsilon is not a constant")
        eps_val = float(epsilon.const_value.numpy().flat[0])
        if eps_val <= 0 or eps_val > 1.0:
            return result.fail(f"Epsilon {eps_val} out of expected range (0, 1]")

        for tag, axes in [("first", axes1), ("second", axes2)]:
            err = _check_axes_minus_one(axes, tag)
            if err:
                return result.fail(err)

        return result

    def rewrite(self, op, x, weight, epsilon, **_):
        eps_val = float(epsilon.const_value.numpy().flat[0])
        return op.LayerNormalization(x, weight, axis=-1, epsilon=eps_val)


def layer_norm_fusion_rules() -> RewriteRuleSet:
    """Return rules that fuse decomposed LayerNorm into LayerNormalization.

    These rules match the classic ``ReduceMean``-based decomposition of
    LayerNormalization and replace it with the standard ONNX
    ``LayerNormalization`` op (opset 17+).

    Includes both the biased variant (scale + bias) and the bias-free
    variant (scale only).

    Returns:
        :class:`RewriteRuleSet` containing the LayerNorm fusion rules.
    """
    return RewriteRuleSet(
        [
            LayerNormFusion().rule(),
            LayerNormFusionNoBias().rule(),
        ]
    )
