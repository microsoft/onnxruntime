# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the GroupNorm onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx_ir_fusions import group_norm_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_group_norm_model(
    swish: bool = False,
    channels: int = 8,
    groups: int = 4,
    height: int = 2,
    width: int = 2,
    instance_scale: float = 1.0,
    root_shape: list[int] | None = None,
    affine_channels: int | None = None,
    data_type: int = onnx.TensorProto.FLOAT,
) -> onnx.ModelProto:
    root_shape = [1, channels, height, width] if root_shape is None else root_shape
    affine_channels = channels if affine_channels is None else affine_channels
    initializers = [
        helper.make_tensor("shape3d", onnx.TensorProto.INT64, [3], [0, groups, -1]),
        helper.make_tensor("in_scale", data_type, [groups], np.full(groups, instance_scale, np.float32)),
        helper.make_tensor("in_bias", data_type, [groups], np.zeros(groups, np.float32)),
        helper.make_tensor(
            "weight",
            data_type,
            [affine_channels, 1, 1],
            np.arange(affine_channels, dtype=np.float32).reshape(affine_channels, 1, 1),
        ),
        helper.make_tensor(
            "bias",
            data_type,
            [affine_channels, 1, 1],
            (np.arange(affine_channels, dtype=np.float32) + 0.5).reshape(affine_channels, 1, 1),
        ),
    ]
    nodes = [
        helper.make_node("Shape", ["root"], ["shape4d"]),
        helper.make_node("Reshape", ["root", "shape3d"], ["r3"]),
        helper.make_node("InstanceNormalization", ["r3", "in_scale", "in_bias"], ["inorm"], epsilon=1e-5),
        helper.make_node("Reshape", ["inorm", "shape4d"], ["r4"]),
        helper.make_node("Mul", ["r4", "weight"], ["mul"]),
        helper.make_node("Add", ["mul", "bias"], ["add"]),
    ]
    if swish:
        nodes += [
            helper.make_node("Sigmoid", ["add"], ["sig"]),
            helper.make_node("Mul", ["add", "sig"], ["y"]),
        ]
    else:
        nodes.append(helper.make_node("Identity", ["add"], ["y"]))

    graph = helper.make_graph(
        nodes,
        "groupnorm",
        [helper.make_tensor_value_info("root", data_type, root_shape)],
        [helper.make_tensor_value_info("y", data_type, root_shape)],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestGroupNormFusion(unittest.TestCase):
    def _assert_fused(self, swish: bool):
        model = to_ir(_build_group_norm_model(swish=swish))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("GroupNorm", 0), 1)
        self.assertEqual(counts.get("InstanceNormalization", 0), 0)
        # NCHW->NHWC and NHWC->NCHW transposes wrap the channels-last GroupNorm.
        self.assertEqual(counts.get("Transpose", 0), 2)
        # No leftover activation nodes.
        self.assertEqual(counts.get("Sigmoid", 0), 0)

        node = next(n for n in model.graph if n.op_type == "GroupNorm")
        self.assertEqual(node.domain, "com.microsoft")
        self.assertEqual(node.attributes.get_int("groups"), 4)
        self.assertEqual(node.attributes.get_int("activation"), 1 if swish else 0)

    def test_fuses_group_norm(self):
        self._assert_fused(swish=False)

    def test_fuses_group_norm_with_swish(self):
        self._assert_fused(swish=True)

    def test_does_not_fuse_non_identity_instance_norm(self):
        # InstanceNormalization scale != ones means this is not a plain GroupNorm.
        model = to_ir(_build_group_norm_model(instance_scale=2.0))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("GroupNorm", 0), 0)
        self.assertEqual(counts.get("InstanceNormalization", 0), 1)

    def test_does_not_fuse_rank_3_root(self):
        model = to_ir(_build_group_norm_model(root_shape=[1, 8, 4]))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        self.assertEqual(op_counts(model).get("GroupNorm", 0), 0)

    def test_does_not_fuse_mismatched_affine_channels(self):
        model = to_ir(_build_group_norm_model(affine_channels=4))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        self.assertEqual(op_counts(model).get("GroupNorm", 0), 0)

    def test_does_not_fuse_indivisible_channel_count(self):
        model = to_ir(_build_group_norm_model(channels=6, groups=4))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        self.assertEqual(op_counts(model).get("GroupNorm", 0), 0)

    def test_does_not_fuse_double_input(self):
        model = to_ir(_build_group_norm_model(data_type=onnx.TensorProto.DOUBLE))
        rewrite(model, pattern_rewrite_rules=group_norm_rules())

        self.assertEqual(op_counts(model).get("GroupNorm", 0), 0)


if __name__ == "__main__":
    unittest.main()
