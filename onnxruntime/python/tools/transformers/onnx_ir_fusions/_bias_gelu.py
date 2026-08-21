# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rewrite rules for fusing Add + Gelu into BiasGelu.

In the FFN / MLP pattern of GPT-2-like models, the first linear projection
adds a bias before the Gelu activation: ``Gelu(MatMul(x, w) + bias)``.
The ``com.microsoft::BiasGelu`` custom op fuses the bias addition and
activation into a single kernel, reducing memory traffic.

These rules are **not applied by default**.  Apply them post-export::

    from onnx_ir_fusions import bias_gelu_rules
    from onnxscript.rewriter import rewrite

    model = ir.load("model.onnx")
    rewrite(model, pattern_rewrite_rules=bias_gelu_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class AddGeluToBiasGelu(RewriteRuleClassBase):
    """Replace Add + Gelu with BiasGelu.

    **Matched pattern:**

    .. code-block:: text

        add_out = Add(x, bias)
        gelu_out = Gelu(add_out)

    Where ``add_out`` has exactly one consumer (the Gelu node).

    **Replacement:**

    .. code-block:: text

        gelu_out = BiasGelu(x, bias)

    Both inputs to Add are passed directly to BiasGelu, which computes
    ``Gelu(x + bias)`` in a single fused kernel.
    """

    def pattern(self, op, add_out):
        return op.Gelu(add_out, _outputs=["gelu_out"])

    def check(self, context, add_out, gelu_out, **_):
        result = MatchResult()

        # ORT BiasGelu uses tanh approximation — only match Gelu nodes
        # with approximate="tanh" (skip exact Gelu which has no attribute
        # or approximate="none")
        gelu_node = gelu_out.producer()
        if gelu_node is not None:
            approx = gelu_node.attributes.get("approximate", None)
            approx_val = approx.value if approx is not None else "none"
            if approx_val != "tanh":
                return result.fail(f"Gelu uses approximate='{approx_val}', BiasGelu requires 'tanh'")

        # add_out must come from an Add node
        producer = add_out.producer()
        if producer is None or producer.op_type != "Add":
            return result.fail("Input to Gelu is not from an Add node")

        # The Add must have only 1 consumer (the Gelu) — if the Add
        # output is also used elsewhere, we cannot safely remove it
        uses = list(add_out.uses())
        if len(uses) != 1:
            return result.fail(f"Add output has {len(uses)} consumers, expected exactly 1 (the Gelu node)")

        return result

    def rewrite(self, op, add_out, gelu_out, **_):
        add_node = add_out.producer()
        input_a = add_node.inputs[0]
        input_b = add_node.inputs[1]

        return op.op(
            "BiasGelu",
            input_a,
            input_b,
            _domain="com.microsoft",
        )


def bias_gelu_rules() -> RewriteRuleSet:
    """Return rules that fuse Add + Gelu into BiasGelu.

    These rules match the ``Add(x, bias) → Gelu`` pattern common in the
    FFN layers of GPT-2, BERT, and other models that use Gelu activation,
    and replace it with the fused Microsoft ``BiasGelu`` custom op.

    Returns:
        :class:`RewriteRuleSet` containing the Add+Gelu fusion rule.
    """
    return RewriteRuleSet([AddGeluToBiasGelu().rule()])
