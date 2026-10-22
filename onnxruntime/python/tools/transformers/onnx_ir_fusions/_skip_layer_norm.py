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

from ._common import FLOAT_OR_FLOAT16_OR_BFLOAT16, static_shape


def _get_skip_layer_norm_inputs(add_node):
    """Return input/skip operands in the order accepted by the fused kernel."""
    first, second = add_node.inputs
    first_shape = static_shape(first)
    second_shape = static_shape(second)
    if _is_valid_input_skip_pair(first_shape, second_shape):
        return first, second
    if _is_valid_input_skip_pair(second_shape, first_shape):
        return second, first
    return None


def _is_valid_input_skip_pair(input_shape: tuple[int, ...] | None, skip_shape: tuple[int, ...] | None) -> bool:
    """Check the rank-2/3 skip layouts accepted by SkipLayerNormalization."""
    if input_shape is None or skip_shape is None:
        return False
    if len(input_shape) not in (2, 3) or len(skip_shape) not in (2, 3):
        return False
    if input_shape == skip_shape:
        return True
    if len(input_shape) != 3:
        return False
    if len(skip_shape) == 2:
        return skip_shape == input_shape[1:]
    return skip_shape[0] == 1 and skip_shape[1:] == input_shape[1:]


def _check_affine_parameter(value, name: str, hidden_size: int) -> str | None:
    """Return an error if an affine parameter does not match the fused kernel contract."""
    shape = static_shape(value)
    if shape != (hidden_size,):
        return f"{name} must have static shape [{hidden_size}]"
    return None


class _AddLayerNormToSkipLayerNormBase(RewriteRuleClassBase):
    def _check_common(self, add_out, weight, bias, norm_out) -> MatchResult:
        result = MatchResult()

        producer = add_out.producer()
        if producer is None or producer.op_type != "Add":
            return result.fail("Input to LayerNorm is not from an Add node")

        target_inputs = _get_skip_layer_norm_inputs(producer)
        if target_inputs is None:
            return result.fail("Add inputs do not match the SkipLayerNormalization rank/skip-shape contract")
        if target_inputs[0].dtype not in FLOAT_OR_FLOAT16_OR_BFLOAT16:
            return result.fail("SkipLayerNormalization input must have float, float16, or bfloat16 element type")
        input_shape = static_shape(target_inputs[0])

        # Don't fuse if add_out is itself a graph output — that indicates we're inside
        # an ONNX function body where replace_all_uses_with would fail or produce
        # nested fusion.
        graph = producer.graph
        if graph is not None and add_out in graph.outputs:
            return result.fail("Add output is a graph output — skip to avoid nested fusion")

        ln = norm_out.producer()
        if ln.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute on LayerNormalization")

        # SkipLayerNormalization always normalizes over the last axis.
        axis = ln.attributes.get_int("axis", -1)
        if axis != -1:
            return result.fail(f"LayerNorm axis={axis}, expected -1 for SkipLayerNormalization compatibility")

        err = _check_affine_parameter(weight, "gamma", input_shape[-1])
        if err:
            return result.fail(err)
        if bias is not None:
            err = _check_affine_parameter(bias, "beta", input_shape[-1])
            if err:
                return result.fail(err)

        return result

    def _rewrite_common(self, op, add_out, weight, bias, norm_out):
        ln = norm_out.producer()
        epsilon = ln.attributes.get_float("epsilon")

        add_node = add_out.producer()
        input_value, skip_value = _get_skip_layer_norm_inputs(add_node)
        inputs = [input_value, skip_value, weight]
        if bias is not None:
            inputs.append(bias)
        outputs = op.op(
            "SkipLayerNormalization",
            *inputs,
            _domain="com.microsoft",
            epsilon=epsilon,
            _outputs=4,
        )
        new_norm_out = outputs[0]
        skip_out = outputs[3]

        add_out.replace_all_uses_with(skip_out)
        return new_norm_out


class AddLayerNormToSkipLayerNorm(_AddLayerNormToSkipLayerNormBase):
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

    def check(self, context, add_out, weight, bias, norm_out, **_):
        return self._check_common(add_out, weight, bias, norm_out)

    def rewrite(self, op, add_out, weight, bias, norm_out, **_):
        return self._rewrite_common(op, add_out, weight, bias, norm_out)


class AddLayerNormNoBiasToSkipLayerNorm(_AddLayerNormToSkipLayerNormBase):
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

    def check(self, context, add_out, weight, norm_out, **_):
        result = self._check_common(add_out, weight, None, norm_out)
        if not result:
            return result

        ln = norm_out.producer()
        # Ensure this is truly bias-free (2 inputs, not 3)
        if len(ln.inputs) > 2:
            return result.fail("LayerNorm has bias — use the 3-input rule")

        return result

    def rewrite(self, op, add_out, weight, norm_out, **_):
        return self._rewrite_common(op, add_out, weight, None, norm_out)


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
