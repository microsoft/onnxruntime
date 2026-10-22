# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""onnx-ir / onnxscript-rewriter based graph fusions.

This package incrementally re-implements the proto-``OnnxModel`` based fusions
under ``onnxruntime/python/tools/transformers`` using ``onnx-ir`` and
``onnxscript.rewriter``.  See ``README.md`` for the migration approach.

Only fusions that are not already covered elsewhere in the onnx-ir ecosystem
need to be authored from scratch; fusions that already exist as onnx-ir
rewrite rules (BiasGelu, Gelu, LayerNormalization, SkipLayerNormalization,
SkipSimplifiedLayerNormalization) are ported directly, while the self-contained
``QuickGelu``, ``BiasAdd`` and ``GroupNorm`` fusions are new here.
"""

from __future__ import annotations

from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from ._bias_add import bias_add_rules
from ._bias_gelu import bias_gelu_rules
from ._gelu_fusion import gelu_fusion_rules
from ._group_norm import group_norm_rules
from ._layer_norm_fusion import layer_norm_fusion_rules
from ._quick_gelu import quick_gelu_rules
from ._skip_layer_norm import skip_layer_norm_rules
from ._skip_norm import skip_norm_rules

__all__ = [
    "all_rules",
    "bias_add_rules",
    "bias_gelu_rules",
    "gelu_fusion_rules",
    "group_norm_rules",
    "layer_norm_fusion_rules",
    "quick_gelu_rules",
    "skip_layer_norm_rules",
    "skip_norm_rules",
]


def all_rules() -> RewriteRuleSet:
    """Return a combined rule set with every fusion in this package."""
    rules = []
    for factory in (
        quick_gelu_rules,
        bias_add_rules,
        group_norm_rules,
        bias_gelu_rules,
        gelu_fusion_rules,
        layer_norm_fusion_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
    ):
        rules.extend(factory().rules)
    return RewriteRuleSet(rules)
