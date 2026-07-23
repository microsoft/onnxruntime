# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rewrite rules for fusing decomposed GeLU into the native Gelu op.

Models exported from older ONNX opsets or non-PyTorch frameworks may
represent GeLU as a chain of primitive ops.

**Exact GeLU:**

.. code-block:: text

    x_div = Div(x, sqrt(2))
    erf_out = Erf(x_div)
    add_one = Add(erf_out, 1)
    mul_x = Mul(x, add_one)
    result = Mul(mul_x, 0.5)

**Approximate (tanh) GeLU:**

.. code-block:: text

    x3 = Pow(x, 3)
    inner = Add(x, Mul(0.044715, x3))
    tanh_out = Tanh(Mul(sqrt(2/pi), inner))
    result = Mul(Mul(x, Add(tanh_out, 1)), 0.5)

These rules fuse these chains into the standard ONNX ``Gelu`` op
(opset 20+).

These rules are applied automatically by
the model optimizer for all execution providers
(GeluFusion stage).  They can also be applied manually::

    from onnx_ir_fusions import gelu_fusion_rules
    from onnxscript.rewriter import rewrite

    rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())
"""

from __future__ import annotations

import math

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)

_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFF = 0.044715


def _check_const(value, expected: float, name: str) -> str | None:
    """Return error string if *value* is not a constant ≈ *expected*."""
    if value.const_value is None:
        return f"{name} is not a constant"
    actual = float(value.const_value.numpy().flat[0])
    if not math.isclose(actual, expected, rel_tol=1e-3):
        return f"{name} is {actual}, expected ~{expected}"
    return None


# ── Exact GeLU: Div(x, sqrt2) → Erf → Add(1) → Mul(x, ·) → Mul(·, 0.5) ──


class ExactGeluFusion(RewriteRuleClassBase):
    """Fuse decomposed exact GeLU into ``Gelu(x, approximate="none")``.

    **Matched pattern:**

    .. code-block:: text

        x_div = Div(x, sqrt2)          # sqrt2 ≈ 1.4142
        erf_out = Erf(x_div)
        add_one = Add(erf_out, one)     # one = 1.0
        mul_x = Mul(x, add_one)
        result = Mul(mul_x, half)       # half = 0.5

    **Replacement:**

    .. code-block:: text

        result = Gelu(x, approximate="none")
    """

    def pattern(self, op, x, sqrt2, one_val, half_val):
        x_div = op.Div(x, sqrt2)
        erf_out = op.Erf(x_div)
        add_one = op.Add(erf_out, one_val)
        mul_x = op.Mul(x, add_one)
        return op.Mul(mul_x, half_val)

    def check(self, context, sqrt2, one_val, half_val, **_):
        result = MatchResult()

        err = _check_const(sqrt2, _SQRT2, "sqrt(2) divisor")
        if err:
            return result.fail(err)
        err = _check_const(one_val, 1.0, "Add constant")
        if err:
            return result.fail(err)
        err = _check_const(half_val, 0.5, "Mul half constant")
        if err:
            return result.fail(err)

        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="none")


class ExactGeluFusionHalfFirst(RewriteRuleClassBase):
    """Variant where the 0.5 multiply comes first: ``Mul(0.5, Mul(x, ·))``.

    Some exporters emit ``0.5 * (x * (1 + erf(x / sqrt(2))))`` with the
    scalar half on the left of the outer Mul.
    """

    def pattern(self, op, x, sqrt2, one_val, half_val):
        x_div = op.Div(x, sqrt2)
        erf_out = op.Erf(x_div)
        add_one = op.Add(erf_out, one_val)
        mul_x = op.Mul(x, add_one)
        return op.Mul(half_val, mul_x)

    def check(self, context, sqrt2, one_val, half_val, **_):
        result = MatchResult()

        err = _check_const(sqrt2, _SQRT2, "sqrt(2) divisor")
        if err:
            return result.fail(err)
        err = _check_const(one_val, 1.0, "Add constant")
        if err:
            return result.fail(err)
        err = _check_const(half_val, 0.5, "Mul half constant")
        if err:
            return result.fail(err)

        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="none")


# ── Approximate (tanh) GeLU ─────────────────────────────────────────────


class ApproxGeluFusion(RewriteRuleClassBase):
    """Fuse decomposed tanh-approximate GeLU into ``Gelu(x, approximate="tanh")``.

    **Matched pattern:**

    .. code-block:: text

        x3 = Pow(x, 3)
        scaled_cube = Mul(coeff, x3)        # coeff ≈ 0.044715
        inner = Add(x, scaled_cube)
        scaled = Mul(sqrt_2pi, inner)        # sqrt(2/π) ≈ 0.7979
        tanh_out = Tanh(scaled)
        add_one = Add(tanh_out, one)
        mul_x = Mul(x, add_one)
        result = Mul(mul_x, half)

    **Replacement:**

    .. code-block:: text

        result = Gelu(x, approximate="tanh")
    """

    def pattern(self, op, x, three, coeff, sqrt_2pi, one_val, half_val):
        x3 = op.Pow(x, three)
        scaled_cube = op.Mul(coeff, x3)
        inner = op.Add(x, scaled_cube)
        scaled = op.Mul(sqrt_2pi, inner)
        tanh_out = op.Tanh(scaled)
        add_one = op.Add(tanh_out, one_val)
        mul_x = op.Mul(x, add_one)
        return op.Mul(mul_x, half_val)

    def check(self, context, three, coeff, sqrt_2pi, one_val, half_val, **_):
        result = MatchResult()

        err = _check_const(three, 3.0, "Pow exponent")
        if err:
            return result.fail(err)
        err = _check_const(coeff, _GELU_COEFF, "GeLU coefficient")
        if err:
            return result.fail(err)
        err = _check_const(sqrt_2pi, _SQRT_2_OVER_PI, "sqrt(2/pi) constant")
        if err:
            return result.fail(err)
        err = _check_const(one_val, 1.0, "Add constant")
        if err:
            return result.fail(err)
        err = _check_const(half_val, 0.5, "Mul half constant")
        if err:
            return result.fail(err)

        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="tanh")


class ApproxGeluFusionHalfFirst(RewriteRuleClassBase):
    """Variant where 0.5 is on the left: ``Mul(half, Mul(x, ·))``."""

    def pattern(self, op, x, three, coeff, sqrt_2pi, one_val, half_val):
        x3 = op.Pow(x, three)
        scaled_cube = op.Mul(coeff, x3)
        inner = op.Add(x, scaled_cube)
        scaled = op.Mul(sqrt_2pi, inner)
        tanh_out = op.Tanh(scaled)
        add_one = op.Add(tanh_out, one_val)
        mul_x = op.Mul(x, add_one)
        return op.Mul(half_val, mul_x)

    def check(self, context, three, coeff, sqrt_2pi, one_val, half_val, **_):
        result = MatchResult()

        err = _check_const(three, 3.0, "Pow exponent")
        if err:
            return result.fail(err)
        err = _check_const(coeff, _GELU_COEFF, "GeLU coefficient")
        if err:
            return result.fail(err)
        err = _check_const(sqrt_2pi, _SQRT_2_OVER_PI, "sqrt(2/pi) constant")
        if err:
            return result.fail(err)
        err = _check_const(one_val, 1.0, "Add constant")
        if err:
            return result.fail(err)
        err = _check_const(half_val, 0.5, "Mul half constant")
        if err:
            return result.fail(err)

        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="tanh")


def gelu_fusion_rules() -> RewriteRuleSet:
    """Return rules that fuse decomposed GeLU into the native Gelu op.

    Handles both exact GeLU (Erf-based) and approximate GeLU (tanh-based)
    decompositions.  For each variant, two operand orderings of the final
    ``Mul(…, 0.5)`` are covered.

    Returns:
        :class:`RewriteRuleSet` containing the GeLU fusion rules.
    """
    return RewriteRuleSet(
        [
            ExactGeluFusion().rule(),
            ExactGeluFusionHalfFirst().rule(),
            ApproxGeluFusion().rule(),
            ApproxGeluFusionHalfFirst().rule(),
        ]
    )
