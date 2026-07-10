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

from ._common import is_constant_with_rank


class _FuseBiasAddBase(RewriteRuleClassBase):
    def check(self, context, bias, **_) -> MatchResult:
        result = MatchResult()
        if not is_constant_with_rank(bias, 1):
            return result.fail("bias must be a 1-D constant")
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
