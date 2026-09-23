# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rewrite rules for fusing Add + Gelu into BiasGelu or FastGelu.

In the FFN / MLP pattern of GPT-2-like models, the first linear projection
adds a bias before the Gelu activation: ``Gelu(MatMul(x, w) + bias)``.
The ``com.microsoft::BiasGelu`` and ``com.microsoft::FastGelu`` custom ops
fuse the bias addition and exact or tanh-approximate activations,
respectively, into a single kernel.

These rules are **not applied by default**.  Apply them post-export::

    from onnx_ir_fusions import bias_gelu_rules
    from onnxscript.rewriter import rewrite

    model = ir.load("model.onnx")
    rewrite(model, pattern_rewrite_rules=bias_gelu_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _AddGeluToFusedGeluBase(RewriteRuleClassBase):
    """Replace Add + Gelu with a semantically compatible fused Gelu op.

    **Matched pattern:**

    .. code-block:: text

        add_out = Add(x, bias)
        gelu_out = Gelu(add_out)

    Where ``add_out`` has exactly one consumer (the Gelu node).

    **Replacement:**

    .. code-block:: text

        gelu_out = BiasGelu(x, bias) or FastGelu(x, bias)

    Both inputs to Add are passed directly to the compatible fused op, which
    computes ``Gelu(x + bias)`` in a single fused kernel.
    """

    target_op_type: str
    approximate: str

    def check(self, context, x, bias, gelu_out, **_):
        result = MatchResult()

        gelu_node = gelu_out.producer()
        if gelu_node is not None:
            approx = gelu_node.attributes.get("approximate", None)
            approx_val = approx.value if approx is not None else "none"
            if approx_val != self.approximate:
                return result.fail(
                    f"Gelu uses approximate='{approx_val}', {self.target_op_type} requires '{self.approximate}'"
                )

        x_shape = x.shape
        if x_shape is None or len(x_shape) < 1:
            return result.fail(f"{self.target_op_type} input must have rank at least 1")

        bias_shape = bias.shape
        if bias_shape is None or len(bias_shape) != 1:
            return result.fail(f"{self.target_op_type} bias must have rank 1")

        if not x_shape.is_static(-1) or not bias_shape.is_static(0) or x_shape[-1] != bias_shape[0]:
            return result.fail(f"{self.target_op_type} bias length must equal the last input dimension")

        return result

    def rewrite(self, op, x, bias, **_):
        return op.op(
            self.target_op_type,
            x,
            bias,
            _domain="com.microsoft",
        )


class AddGeluToFastGelu(_AddGeluToFusedGeluBase):
    """``Gelu(Add(x, bias), approximate="tanh")`` -> ``FastGelu(x, bias)``."""

    target_op_type = "FastGelu"
    approximate = "tanh"

    def pattern(self, op, x, bias):
        return op.Gelu(op.Add(x, bias), _outputs=["gelu_out"])


class AddGeluBiasFirstToFastGelu(_AddGeluToFusedGeluBase):
    """``Gelu(Add(bias, x), approximate="tanh")`` -> ``FastGelu(x, bias)``."""

    target_op_type = "FastGelu"
    approximate = "tanh"

    def pattern(self, op, x, bias):
        return op.Gelu(op.Add(bias, x), _outputs=["gelu_out"])


class AddGeluToBiasGelu(_AddGeluToFusedGeluBase):
    """``Gelu(Add(x, bias), approximate="none")`` -> ``BiasGelu(x, bias)``."""

    target_op_type = "BiasGelu"
    approximate = "none"

    def pattern(self, op, x, bias):
        return op.Gelu(op.Add(x, bias), _outputs=["gelu_out"])


class AddGeluBiasFirstToBiasGelu(_AddGeluToFusedGeluBase):
    """``Gelu(Add(bias, x), approximate="none")`` -> ``BiasGelu(x, bias)``."""

    target_op_type = "BiasGelu"
    approximate = "none"

    def pattern(self, op, x, bias):
        return op.Gelu(op.Add(bias, x), _outputs=["gelu_out"])


def bias_gelu_rules() -> RewriteRuleSet:
    """Return rules that fuse Add + Gelu into BiasGelu or FastGelu.

    These rules match the ``Add(x, bias) → Gelu`` pattern common in the
    FFN layers of GPT-2, BERT, and other models. Exact Gelu is replaced with
    the fused Microsoft ``BiasGelu`` custom op, and tanh-approximate Gelu is
    replaced with ``FastGelu``.

    Returns:
        :class:`RewriteRuleSet` containing the Add+Gelu fusion rules.
    """
    return RewriteRuleSet(
        [
            AddGeluToFastGelu().rule(),
            AddGeluBiasFirstToFastGelu().rule(),
            AddGeluToBiasGelu().rule(),
            AddGeluBiasFirstToBiasGelu().rule(),
        ]
    )
