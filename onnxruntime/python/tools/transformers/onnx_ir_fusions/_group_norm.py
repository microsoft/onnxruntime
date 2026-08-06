# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GroupNorm fusion as an onnxscript rewrite rule.

Replaces the ``FusionGroupNorm`` (``fusion_group_norm.py``) pattern, which is
built on the proto-based ``OnnxModel`` helper API, with an ``onnx-ir`` /
``onnxscript.rewriter`` rule.

Pattern (channels_last, optional swish activation)::

        +---------------- Shape ---------------------------------+
        |                                                        v
    [root] -> Reshape([0,G,-1]) -> InstanceNorm -> Reshape -> Mul(w) -> Add(b) [-> Sigmoid -> Mul]
    BxCxHxW               (scale=ones(G), bias=zeros(G))          (Cx1x1)  (Cx1x1)

becomes ``Transpose(NCHW->NHWC) -> com.microsoft.GroupNorm -> Transpose(NHWC->NCHW)``
with ``gamma``/``beta`` = flattened ``w``/``b`` and ``groups`` = ``G``. The Mul +
Sigmoid tail (swish) is folded into the GroupNorm ``activation`` attribute.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

from ._common import constant_array


class _GroupNormBase(RewriteRuleClassBase):
    def _check_common(self, root, shape_3d, in_scale, in_bias, shape_4d, weight, bias) -> MatchResult:
        result = MatchResult()

        scale = constant_array(in_scale)
        inorm_bias = constant_array(in_bias)
        if scale is None or inorm_bias is None or scale.ndim != 1:
            return result.fail("InstanceNormalization scale/bias must be 1-D constants")
        num_groups = int(scale.shape[0])
        if not np.allclose(scale, 1.0) or not np.allclose(inorm_bias, 0.0):
            return result.fail("InstanceNormalization must have scale=ones and bias=zeros")

        target_3d = constant_array(shape_3d)
        if target_3d is None or target_3d.tolist() != [0, num_groups, -1]:
            return result.fail("first Reshape must target [0, num_groups, -1]")

        # The second Reshape target must be Shape(root) so the tensor is restored
        # to its original 4-D layout.
        producer = shape_4d.producer()
        if producer is None or producer.op_type != "Shape" or producer.inputs[0] is not root:
            return result.fail("second Reshape target must be Shape(root)")

        weight_array = constant_array(weight)
        bias_array = constant_array(bias)
        if weight_array is None or bias_array is None:
            return result.fail("GroupNorm weight/bias must be constants")
        for name, array in (("weight", weight_array), ("bias", bias_array)):
            if not (array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == 1):
                return result.fail(f"{name} must have shape [C, 1, 1]")
        if weight_array.shape[0] != bias_array.shape[0]:
            return result.fail("weight/bias channel counts differ")

        return result

    def _build(self, op, root, weight, bias, inorm_out, activation: int):
        gamma = constant_array(weight).reshape(-1).astype(np.float32)
        beta = constant_array(bias).reshape(-1).astype(np.float32)
        inorm = inorm_out.producer()
        epsilon = inorm.attributes.get_float("epsilon", 1e-5)
        num_groups = int(constant_array(inorm.inputs[1]).shape[0])

        gamma_value = op.Constant(value=ir.tensor(gamma, name="groupnorm_gamma"))
        beta_value = op.Constant(value=ir.tensor(beta, name="groupnorm_beta"))

        # com.microsoft.GroupNorm operates on channels-last (NHWC) tensors.
        nhwc = op.Transpose(root, perm=[0, 2, 3, 1])
        group_norm = op.op(
            "GroupNorm",
            nhwc,
            gamma_value,
            beta_value,
            _domain="com.microsoft",
            epsilon=float(epsilon),
            groups=num_groups,
            activation=activation,
        )
        return op.Transpose(group_norm, perm=[0, 3, 1, 2])


class FuseGroupNorm(_GroupNormBase):
    """Reshape/InstanceNorm/Reshape/Mul/Add -> GroupNorm (no activation)."""

    def pattern(self, op, root, shape_3d, in_scale, in_bias, shape_4d, weight, bias):
        reshaped_3d = op.Reshape(root, shape_3d)
        inorm = op.InstanceNormalization(
            reshaped_3d, in_scale, in_bias, _allow_other_attributes=True, _outputs=["inorm_out"]
        )
        reshaped_4d = op.Reshape(inorm, shape_4d)
        scaled = op.Mul(reshaped_4d, weight)
        return op.Add(scaled, bias, _outputs=["gn_add_out"])

    def check(
        self, context, root, shape_3d, in_scale, in_bias, shape_4d, weight, bias, inorm_out, gn_add_out, **_
    ) -> MatchResult:
        result = self._check_common(root, shape_3d, in_scale, in_bias, shape_4d, weight, bias)
        if not result:
            return result
        # If the Add feeds a Sigmoid, this is a swish activation; defer to
        # FuseGroupNormSwish so the activation is folded into the GroupNorm node.
        for consumer, _index in gn_add_out.uses():
            if consumer.op_type == "Sigmoid":
                return result.fail("Add feeds a swish activation; handled by FuseGroupNormSwish")
        return result

    def rewrite(self, op, root, weight, bias, inorm_out, **_):
        return self._build(op, root, weight, bias, inorm_out, activation=0)


class FuseGroupNormSwish(_GroupNormBase):
    """Reshape/InstanceNorm/Reshape/Mul/Add + Sigmoid*Mul (swish) -> GroupNorm(activation=1)."""

    def pattern(self, op, root, shape_3d, in_scale, in_bias, shape_4d, weight, bias):
        reshaped_3d = op.Reshape(root, shape_3d)
        inorm = op.InstanceNormalization(
            reshaped_3d, in_scale, in_bias, _allow_other_attributes=True, _outputs=["inorm_out"]
        )
        reshaped_4d = op.Reshape(inorm, shape_4d)
        scaled = op.Mul(reshaped_4d, weight)
        biased = op.Add(scaled, bias)
        return op.Mul(biased, op.Sigmoid(biased))

    def check(self, context, root, shape_3d, in_scale, in_bias, shape_4d, weight, bias, inorm_out, **_) -> MatchResult:
        return self._check_common(root, shape_3d, in_scale, in_bias, shape_4d, weight, bias)

    def rewrite(self, op, root, weight, bias, inorm_out, **_):
        return self._build(op, root, weight, bias, inorm_out, activation=1)


def group_norm_rules() -> RewriteRuleSet:
    """Return the GroupNorm fusion rule set (swish variant first)."""
    return RewriteRuleSet([FuseGroupNormSwish().rule(), FuseGroupNorm().rule()])
