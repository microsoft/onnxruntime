# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QuickGelu fusion as an onnxscript rewrite rule.

Replaces the ``FusionQuickGelu`` (``fusion_quickgelu.py``) pattern, which is
built on the proto-based ``OnnxModel`` helper API, with an ``onnx-ir`` /
``onnxscript.rewriter`` rule:

    x --> Mul(x, alpha) --> Sigmoid --> Mul(x, .) --> y     (alpha ~= 1.702)

becomes ``com.microsoft.QuickGelu(x)`` with attribute ``alpha``.
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

from ._common import scalar_constant

# The QuickGelu sigmoid approximation constant (matches fusion_quickgelu.py).
_QUICK_GELU_ALPHA = 1.7021484375
_ALPHA_TOLERANCE = 1e-3


class FuseQuickGelu(RewriteRuleClassBase):
    """``x * sigmoid(alpha * x)`` -> ``com.microsoft.QuickGelu(x, alpha=alpha)``."""

    def pattern(self, op, x, alpha):
        scaled = op.Mul(x, alpha)
        sigmoid = op.Sigmoid(scaled)
        return op.Mul(x, sigmoid)

    def check(self, context, alpha, **_) -> MatchResult:
        result = MatchResult()
        value = scalar_constant(alpha)
        if value is None or abs(value - _QUICK_GELU_ALPHA) >= _ALPHA_TOLERANCE:
            return result.fail("multiplier is not the QuickGelu approximation constant (~1.702)")
        return result

    def rewrite(self, op, x, alpha, **_):
        value = scalar_constant(alpha)
        return op.op("QuickGelu", x, _domain="com.microsoft", alpha=float(value))


def quick_gelu_rules() -> RewriteRuleSet:
    """Return the QuickGelu fusion rule set."""
    return RewriteRuleSet([FuseQuickGelu().rule()])
