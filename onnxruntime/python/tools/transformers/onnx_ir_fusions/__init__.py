# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""onnx-ir / onnxscript-rewriter based graph fusions.

This package incrementally re-implements the proto-``OnnxModel`` based fusions
under ``onnxruntime/python/tools/transformers`` using ``onnx-ir`` and
``onnxscript.rewriter``.  See ``README.md`` for the migration approach.

Only fusions that are not already covered elsewhere in the onnx-ir ecosystem
are re-implemented here; the first batch covers the self-contained fusions
``QuickGelu``, ``BiasAdd`` and ``GroupNorm``.
"""

from __future__ import annotations

from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from ._bias_add import bias_add_rules
from ._group_norm import group_norm_rules
from ._quick_gelu import quick_gelu_rules

__all__ = [
    "all_rules",
    "bias_add_rules",
    "group_norm_rules",
    "quick_gelu_rules",
]


def all_rules() -> RewriteRuleSet:
    """Return a combined rule set with every fusion in this package."""
    rules = []
    for factory in (quick_gelu_rules, bias_add_rules, group_norm_rules):
        rules.extend(factory().rules)
    return RewriteRuleSet(rules)
