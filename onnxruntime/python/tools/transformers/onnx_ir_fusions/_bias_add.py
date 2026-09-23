# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BiasAdd fusion as an onnxscript rewrite rule.

Replaces the ``FusionBiasAdd`` (``fusion_bias_add.py``) pattern, which is built
on the proto-based ``OnnxModel`` helper API, with an ``onnx-ir`` /
``onnxscript.rewriter`` rule:

    (X + bias) + skip  -->  com.microsoft.BiasAdd(X, bias, skip)

where ``bias`` is a 1-D constant. ``Add`` is commutative, so both operand
orders of the inner bias-add are handled by two rule variants.
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

from ._common import FLOAT_OR_FLOAT16, constant_array, static_shape


class _FuseBiasAddBase(RewriteRuleClassBase):
    def check(self, context, x, bias, skip, **_) -> MatchResult:
        result = MatchResult()
        bias_array = constant_array(bias)
        if bias_array is None or bias_array.ndim != 1:
            return result.fail("bias must be a 1-D constant")

        x_shape = static_shape(x)
        if x_shape is None or len(x_shape) != 3:
            return result.fail("BiasAdd input must have a static rank-3 shape")
        if x.dtype not in FLOAT_OR_FLOAT16:
            return result.fail("BiasAdd input must have float or float16 element type")

        if bias_array.shape[0] != x_shape[-1]:
            return result.fail("bias length must equal the last input dimension")

        skip_shape = static_shape(skip)
        if skip_shape != x_shape:
            return result.fail("skip must have the exact same shape as input")

        return result

    def rewrite(self, op, x, bias, skip, **_):
        return op.op("BiasAdd", x, bias, skip, _domain="com.microsoft")


class FuseBiasAdd(_FuseBiasAddBase):
    """``(X + bias) + skip`` (bias second) -> ``BiasAdd(X, bias, skip)``."""

    def pattern(self, op, x, bias, skip):
        return op.Add(op.Add(x, bias), skip)


class FuseBiasAddBiasFirst(_FuseBiasAddBase):
    """``(bias + X) + skip`` (bias first) -> ``BiasAdd(X, bias, skip)``."""

    def pattern(self, op, x, bias, skip):
        return op.Add(op.Add(bias, x), skip)


def bias_add_rules() -> RewriteRuleSet:
    """Return the BiasAdd fusion rule set (both bias operand orders)."""
    return RewriteRuleSet([FuseBiasAdd().rule(), FuseBiasAddBiasFirst().rule()])
