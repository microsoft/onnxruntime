# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rewrite rules for fusing Add + LayerNormalization into SkipLayerNormalization.

In the standard decoder layer pattern (GPT-2, BERT, Phi, etc.), a residual
Add is followed by LayerNormalization, and the Add output is also passed
forward as the running residual.  The ``com.microsoft::SkipLayerNormalization``
custom op fuses these into a single node with four outputs: the normalized
result, mean, inv_std_var, and the skip (unnormalized sum).

This complements the ``skip_norm_rules`` which handles Add + RMSNormalization
→ SkipSimplifiedLayerNormalization for models using RMSNorm.

These rules are applied automatically by
the model optimizer for EPs that support
SkipLayerNormalization (``supports_skip_layer_norm=True``; all EPs except
TRT-RTX).  They can also be applied manually::

    from onnx_ir_fusions import skip_layer_norm_rules
    from onnxscript.rewriter import rewrite

    model = ir.load("model.onnx")
    rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class AddLayerNormToSkipLayerNorm(RewriteRuleClassBase):
    """Replace Add + LayerNormalization with SkipLayerNormalization.

    **Matched pattern:**

    .. code-block:: text

        norm_out = LayerNormalization(add_out, weight, bias?, epsilon=eps)

    Where ``add_out`` is produced by an ``Add`` node with at least two
    consumers (the LayerNorm and a downstream residual connection).

    **Replacement:**

    .. code-block:: text

        norm_out, _, _, skip_out = SkipLayerNormalization(
            add_input_0, add_input_1, weight, bias?, epsilon=eps,
        )

    ``skip_out`` (= add_input_0 + add_input_1) replaces the original
    ``add_out`` in all downstream consumers except the matched LayerNorm.
    """

    def pattern(self, op, add_out, weight, bias):
        return op.LayerNormalization(
            add_out,
            weight,
            bias,
            _allow_other_attributes=True,
            _outputs=["norm_out"],
        )

    def check(self, context, add_out, norm_out, **_):
        result = MatchResult()

        # add_out must come from an Add node
        producer = add_out.producer()
        if producer is None or producer.op_type != "Add":
            return result.fail("Input to LayerNorm is not from an Add node")

        # Both Add inputs must have at least 2 dimensions.  This prevents
        # fusing a bias-Add (e.g. MatMul + 1D bias → LayerNorm) which
        # would produce a SkipLayerNormalization with a 1D skip input
        # that ORT rejects.  Unknown shapes are allowed through since
        # most intermediate values lack static shape info.
        for i, inp in enumerate(producer.inputs):
            if inp is not None and inp.shape is not None:
                rank = len(inp.shape)
                if rank < 2:
                    return result.fail(f"Add input[{i}] has rank {rank}, need ≥ 2 for skip connection")

        # Don't fuse if add_out is itself a graph output — that indicates we're inside
        # an ONNX function body where replace_all_uses_with would fail or produce
        # nested fusion.
        graph = producer.graph
        if graph is not None and add_out in graph.outputs:
            return result.fail("Add output is a graph output — skip to avoid nested fusion")

        # Verify LayerNormalization has epsilon attribute and correct axis
        ln = norm_out.producer()
        if ln.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute on LayerNormalization")

        # SkipLayerNormalization always normalizes over the last axis
        axis = ln.attributes.get_int("axis", -1)
        if axis != -1:
            return result.fail(f"LayerNorm axis={axis}, expected -1 for SkipLayerNormalization compatibility")

        return result

    def rewrite(self, op, add_out, weight, bias, norm_out, **_):
        ln = norm_out.producer()
        epsilon = ln.attributes.get_float("epsilon")

        # Get the two inputs of the Add node
        add_node = add_out.producer()
        input_a = add_node.inputs[0]
        input_b = add_node.inputs[1]

        outputs = op.SkipLayerNormalization(
            input_a,
            input_b,
            weight,
            bias,
            _domain="com.microsoft",
            epsilon=epsilon,
            _outputs=4,
        )
        new_norm_out = outputs[0]
        skip_out = outputs[3]

        # Replace add_out with skip_out in all other consumers
        add_out.replace_all_uses_with(skip_out)

        return new_norm_out


class AddLayerNormNoBiasToSkipLayerNorm(RewriteRuleClassBase):
    """Replace Add + bias-free LayerNormalization with SkipLayerNormalization.

    Same as :class:`AddLayerNormToSkipLayerNorm` but matches
    ``LayerNormalization`` with only 2 inputs (input, weight) — no bias.
    Some models (e.g. modern BERT variants) omit the LayerNorm bias.

    The fused ``SkipLayerNormalization`` receives only ``[skip_a, skip_b,
    gamma]`` with the optional ``beta`` omitted.
    """

    def pattern(self, op, add_out, weight):
        return op.LayerNormalization(
            add_out,
            weight,
            _allow_other_attributes=True,
            _outputs=["norm_out"],
        )

    def check(self, context, add_out, norm_out, **_):
        result = MatchResult()

        producer = add_out.producer()
        if producer is None or producer.op_type != "Add":
            return result.fail("Input to LayerNorm is not from an Add node")

        # Both Add inputs must have at least 2 dimensions — reject
        # bias-Add patterns (e.g. MatMul + 1D bias → LayerNorm).
        # Unknown shapes are allowed through since most intermediate
        # values lack static shape info.
        for i, inp in enumerate(producer.inputs):
            if inp is not None and inp.shape is not None:
                rank = len(inp.shape)
                if rank < 2:
                    return result.fail(f"Add input[{i}] has rank {rank}, need ≥ 2 for skip connection")

        # Don't fuse if add_out is itself a graph output — that indicates we're inside
        # an ONNX function body where replace_all_uses_with would fail or produce
        # nested fusion.
        graph = producer.graph
        if graph is not None and add_out in graph.outputs:
            return result.fail("Add output is a graph output — skip to avoid nested fusion")

        ln = norm_out.producer()
        if ln.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute on LayerNormalization")

        axis = ln.attributes.get_int("axis", -1)
        if axis != -1:
            return result.fail(f"LayerNorm axis={axis}, expected -1 for SkipLayerNormalization compatibility")

        # Ensure this is truly bias-free (2 inputs, not 3)
        if len(ln.inputs) > 2:
            return result.fail("LayerNorm has bias — use the 3-input rule")

        return result

    def rewrite(self, op, add_out, weight, norm_out, **_):
        ln = norm_out.producer()
        epsilon = ln.attributes.get_float("epsilon")

        add_node = add_out.producer()
        input_a = add_node.inputs[0]
        input_b = add_node.inputs[1]

        # SkipLayerNormalization with gamma only (no beta)
        outputs = op.SkipLayerNormalization(
            input_a,
            input_b,
            weight,
            _domain="com.microsoft",
            epsilon=epsilon,
            _outputs=4,
        )
        new_norm_out = outputs[0]
        skip_out = outputs[3]

        add_out.replace_all_uses_with(skip_out)

        return new_norm_out


def skip_layer_norm_rules() -> RewriteRuleSet:
    """Return rules that fuse Add + LayerNorm into SkipLayerNormalization.

    These rules match the residual Add + LayerNormalization pattern common
    in decoder layers (GPT-2, BERT, Phi, etc.) and replace it with the
    fused Microsoft ``SkipLayerNormalization`` custom op.

    Includes both the 3-input variant (input, weight, bias) and the
    2-input variant (input, weight only — no bias) so that models with
    bias-free LayerNorm are also fused.

    Returns:
        :class:`RewriteRuleSet` containing the Add+LayerNorm fusion rules.
    """
    return RewriteRuleSet(
        [
            AddLayerNormToSkipLayerNorm().rule(),
            AddLayerNormNoBiasToSkipLayerNorm().rule(),
        ]
    )
