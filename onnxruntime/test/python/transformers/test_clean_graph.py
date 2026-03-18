#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Tests for BertOnnxModel.clean_graph() optimizations."""

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from onnx_model_bert import BertOnnxModel
else:
    from onnxruntime.transformers.onnx_model_bert import BertOnnxModel


def _make_constantofshape_cast_model():
    """Create a model with ConstantOfShape → Cast pattern that clean_graph should merge.

    Graph structure (matching the pattern in clean_graph):
      input_ids → Shape → Gather(indices=0) → Unsqueeze ─────────┐
                    │                                              v
                    └→ Shape → Gather(indices=1) → Unsqueeze → Concat → ConstantOfShape → Cast(to=int64) → ReduceSum → output

    After clean_graph:
      input_ids → Shape → ConstantOfShape(value=int64(1)) → ReduceSum → output
    """
    batch_size = 2
    seq_len = 8

    # Graph inputs/outputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, seq_len])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [1])

    # Shape node
    shape_node = helper.make_node("Shape", inputs=["input_ids"], outputs=["shape_out"], name="shape_0")

    # Gather indices=0 (batch dim)
    gather_0_idx = numpy_helper.from_array(np.array(0, dtype=np.int64), name="gather_0_idx")
    gather_0 = helper.make_node(
        "Gather", inputs=["shape_out", "gather_0_idx"], outputs=["gather_0_out"], name="gather_0", axis=0
    )

    # Gather indices=1 (seq dim)
    gather_1_idx = numpy_helper.from_array(np.array(1, dtype=np.int64), name="gather_1_idx")
    gather_1 = helper.make_node(
        "Gather", inputs=["shape_out", "gather_1_idx"], outputs=["gather_1_out"], name="gather_1", axis=0
    )

    # Unsqueeze both
    unsqueeze_0_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_0_axes")
    unsqueeze_0 = helper.make_node(
        "Unsqueeze", inputs=["gather_0_out", "unsqueeze_0_axes"], outputs=["unsqueeze_0_out"], name="unsqueeze_0"
    )

    unsqueeze_1_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_1_axes")
    unsqueeze_1 = helper.make_node(
        "Unsqueeze", inputs=["gather_1_out", "unsqueeze_1_axes"], outputs=["unsqueeze_1_out"], name="unsqueeze_1"
    )

    # Concat
    concat = helper.make_node(
        "Concat", inputs=["unsqueeze_0_out", "unsqueeze_1_out"], outputs=["concat_out"], name="concat_0", axis=0
    )

    # ConstantOfShape with float value
    cos_value = numpy_helper.from_array(np.array([1.0], dtype=np.float32))
    constant_of_shape = helper.make_node(
        "ConstantOfShape",
        inputs=["concat_out"],
        outputs=["cos_out"],
        name="constant_of_shape_0",
        value=cos_value,
    )

    # Cast to int64
    cast = helper.make_node("Cast", inputs=["cos_out"], outputs=["cast_out"], name="cast_0", to=TensorProto.INT64)

    # ReduceSum as consumer (index 0 matches op_input_id)
    reduce_sum = helper.make_node("ReduceSum", inputs=["cast_out"], outputs=["output"], name="reduce_sum_0", keepdims=0)

    nodes = [shape_node, gather_0, gather_1, unsqueeze_0, unsqueeze_1, concat, constant_of_shape, cast, reduce_sum]
    initializers = [gather_0_idx, gather_1_idx, unsqueeze_0_axes, unsqueeze_1_axes]

    graph = helper.make_graph(nodes, "cos_cast_test", [input_ids], [output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


class TestCleanGraph(unittest.TestCase):
    """Tests for BertOnnxModel.clean_graph() optimizations."""

    def _get_node_by_op(self, model, op_type):
        """Find all nodes of a given op type."""
        return [n for n in model.graph.node if n.op_type == op_type]

    def test_constantofshape_cast_merge(self):
        """ConstantOfShape → Cast should be merged: Cast removed, ConstantOfShape produces target type."""
        model = _make_constantofshape_cast_model()

        # Verify precondition: Cast node exists
        cast_nodes = self._get_node_by_op(model, "Cast")
        self.assertEqual(len(cast_nodes), 1, "Expected 1 Cast node before clean_graph")

        cos_nodes = self._get_node_by_op(model, "ConstantOfShape")
        self.assertEqual(len(cos_nodes), 1)
        cos_value = numpy_helper.to_array(cos_nodes[0].attribute[0].t)
        self.assertEqual(cos_value.dtype, np.float32, "ConstantOfShape should start as float32")

        # Run clean_graph
        bert_model = BertOnnxModel(model)
        bert_model.clean_graph()
        bert_model.prune_graph()
        cleaned = bert_model.model

        # Verify: Cast node should be removed
        cast_nodes_after = self._get_node_by_op(cleaned, "Cast")
        self.assertEqual(len(cast_nodes_after), 0, "Cast node should be removed after merge")

        # Verify: ConstantOfShape now produces int64
        cos_nodes_after = self._get_node_by_op(cleaned, "ConstantOfShape")
        self.assertEqual(len(cos_nodes_after), 1, "ConstantOfShape should still exist")
        cos_value_after = numpy_helper.to_array(cos_nodes_after[0].attribute[0].t)
        self.assertEqual(cos_value_after.dtype, np.int64, "ConstantOfShape should produce int64 after merge")
        self.assertEqual(cos_value_after.flat[0], 1, "Fill value should be preserved as 1")

    def test_constantofshape_cast_merge_preserves_fill_value(self):
        """The fill value should be preserved across the type change."""
        model = _make_constantofshape_cast_model()

        # Change the fill value to something other than 1.0
        for node in model.graph.node:
            if node.op_type == "ConstantOfShape":
                new_val = numpy_helper.from_array(np.array([0.0], dtype=np.float32))
                node.attribute[0].t.CopyFrom(new_val)

        bert_model = BertOnnxModel(model)
        bert_model.clean_graph()
        bert_model.prune_graph()
        cleaned = bert_model.model

        cos_nodes = self._get_node_by_op(cleaned, "ConstantOfShape")
        self.assertEqual(len(cos_nodes), 1)
        cos_value = numpy_helper.to_array(cos_nodes[0].attribute[0].t)
        self.assertEqual(cos_value.flat[0], 0, "Fill value 0 should be preserved")
        self.assertEqual(cos_value.dtype, np.int64)

    def test_concat_path_simplified(self):
        """The Concat → Unsqueeze → Gather path should be simplified to just Shape."""
        model = _make_constantofshape_cast_model()

        bert_model = BertOnnxModel(model)
        bert_model.clean_graph()
        bert_model.prune_graph()
        cleaned = bert_model.model

        # Concat, Unsqueeze, and Gather should be pruned
        concat_nodes = self._get_node_by_op(cleaned, "Concat")
        self.assertEqual(len(concat_nodes), 0, "Concat should be pruned")

        gather_nodes = self._get_node_by_op(cleaned, "Gather")
        self.assertEqual(len(gather_nodes), 0, "Gather should be pruned")

        # Shape and ConstantOfShape should remain
        shape_nodes = self._get_node_by_op(cleaned, "Shape")
        self.assertGreaterEqual(len(shape_nodes), 1, "Shape node should remain")

        cos_nodes = self._get_node_by_op(cleaned, "ConstantOfShape")
        self.assertEqual(len(cos_nodes), 1, "ConstantOfShape should remain")

    def test_attention_mask_cleanup_after_cast_merge(self):
        """Attention mask_index path should be removed even after Cast is merged.

        When an Attention node consumes ReduceSum → Cast → ConstantOfShape → Shape
        on input[3], the Cast merge fires first (folding Cast into ConstantOfShape).
        The Attention cleanup must still match the post-merge pattern
        (ReduceSum → ConstantOfShape → Shape) and remove the mask_index input.
        """
        batch_size = 2
        seq_len = 8
        hidden_size = 16

        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [batch_size, seq_len])
        attn_output = helper.make_tensor_value_info("attn_output", TensorProto.FLOAT, None)

        # Shape → Gather → Unsqueeze → Concat → ConstantOfShape → Cast → ReduceSum
        # (same mask path as _make_constantofshape_cast_model)
        shape_node = helper.make_node("Shape", inputs=["input_ids"], outputs=["shape_out"], name="shape_0")

        gather_0_idx = numpy_helper.from_array(np.array(0, dtype=np.int64), name="gather_0_idx")
        gather_0 = helper.make_node(
            "Gather", inputs=["shape_out", "gather_0_idx"], outputs=["gather_0_out"], name="gather_0", axis=0
        )
        gather_1_idx = numpy_helper.from_array(np.array(1, dtype=np.int64), name="gather_1_idx")
        gather_1 = helper.make_node(
            "Gather", inputs=["shape_out", "gather_1_idx"], outputs=["gather_1_out"], name="gather_1", axis=0
        )

        unsqueeze_0_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_0_axes")
        unsqueeze_0 = helper.make_node(
            "Unsqueeze", inputs=["gather_0_out", "unsqueeze_0_axes"], outputs=["unsqueeze_0_out"], name="unsqueeze_0"
        )
        unsqueeze_1_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_1_axes")
        unsqueeze_1 = helper.make_node(
            "Unsqueeze", inputs=["gather_1_out", "unsqueeze_1_axes"], outputs=["unsqueeze_1_out"], name="unsqueeze_1"
        )

        concat = helper.make_node(
            "Concat", inputs=["unsqueeze_0_out", "unsqueeze_1_out"], outputs=["concat_out"], name="concat_0", axis=0
        )

        cos_value = numpy_helper.from_array(np.array([1.0], dtype=np.float32))
        constant_of_shape = helper.make_node(
            "ConstantOfShape", inputs=["concat_out"], outputs=["cos_out"], name="cos_0", value=cos_value
        )
        cast = helper.make_node("Cast", inputs=["cos_out"], outputs=["cast_out"], name="cast_0", to=TensorProto.INT64)
        reduce_sum = helper.make_node(
            "ReduceSum", inputs=["cast_out"], outputs=["reduce_sum_out"], name="reduce_sum_0", keepdims=0
        )

        # Dummy QKV initializers for Attention inputs[0:3]
        qkv_data = numpy_helper.from_array(
            np.zeros([batch_size, seq_len, hidden_size], dtype=np.float32), name="qkv_input"
        )
        weight_data = numpy_helper.from_array(np.zeros([hidden_size, hidden_size], dtype=np.float32), name="weight")
        bias_data = numpy_helper.from_array(np.zeros([hidden_size], dtype=np.float32), name="bias")

        # Attention node with mask_index at input[3]
        attention = helper.make_node(
            "Attention",
            inputs=["qkv_input", "weight", "bias", "reduce_sum_out"],
            outputs=["attn_output"],
            name="attention_0",
            num_heads=2,
        )
        attention.domain = "com.microsoft"

        nodes = [
            shape_node,
            gather_0,
            gather_1,
            unsqueeze_0,
            unsqueeze_1,
            concat,
            constant_of_shape,
            cast,
            reduce_sum,
            attention,
        ]
        initializers = [
            gather_0_idx,
            gather_1_idx,
            unsqueeze_0_axes,
            unsqueeze_1_axes,
            qkv_data,
            weight_data,
            bias_data,
        ]

        graph = helper.make_graph(nodes, "attention_mask_test", [input_ids], [attn_output], initializer=initializers)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 16), helper.make_opsetid("com.microsoft", 1)],
        )

        bert_model = BertOnnxModel(model, num_heads=2, hidden_size=hidden_size)
        bert_model.clean_graph()
        bert_model.prune_graph()
        cleaned = bert_model.model

        # Cast should be merged
        cast_nodes = self._get_node_by_op(cleaned, "Cast")
        self.assertEqual(len(cast_nodes), 0, "Cast node should be removed after merge")

        # Attention node should have mask_index removed (3 inputs instead of 4)
        attn_nodes = self._get_node_by_op(cleaned, "Attention")
        self.assertEqual(len(attn_nodes), 1, "Should have exactly 1 Attention node")
        self.assertEqual(
            len(attn_nodes[0].input),
            3,
            f"Attention should have 3 inputs (mask removed), got {len(attn_nodes[0].input)}",
        )


if __name__ == "__main__":
    unittest.main()
