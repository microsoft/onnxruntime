# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model


def float_tensor(name: str, shape: list[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [np.random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


class TestFusion(unittest.TestCase):
    def verify_skip_layer_norm_fusion(
        self,
        model_path: str,
        expected_counter: dict[str, int],
        expected_inputs: list[str],
        expected_outputs: list[str],
    ):
        options = FusionOptions("bert")
        optimized_model = optimize_model(model_path, optimization_options=options, opt_level=0)

        ops = ["Add", "LayerNormalization", "SkipLayerNormalization", "Cast"]
        for op in ops:
            nodes = optimized_model.get_nodes_by_op_type(op)
            print(op, len(nodes), expected_counter[op])
            self.assertEqual(len(nodes), expected_counter[op])

            if op == "SkipLayerNormalization" and expected_counter[op] == 1:
                print(nodes[0].input)
                print(nodes[0].output)
                self.assertEqual(nodes[0].input, expected_inputs)
                self.assertEqual(nodes[0].output, expected_outputs)

    def create_test_model(
        self,
        batch_size: int = 1,
        sequence_length: int = 2,
        hidden_size: int = 3,
        add_graph_output: bool = True,
        bias: int = 0,  # 0 - no bias, 1 - bias in input_1, 2 - bias in input_2
        cast_before_add_bias=False,
    ):
        matmul = helper.make_node("MatMul", ["input_0", "matmul_weight"], ["matmul_output"], "matmul")
        cast_node = helper.make_node("Cast", ["matmul_output"], ["matmul_output_cast"], to=1)
        add_bias = helper.make_node(
            "Add",
            ["matmul_output_cast" if cast_before_add_bias else "matmul_output", "bias"],
            ["input_1" if bias == 1 else "input_2"],
            "add_bias",
        )

        add_before_layer_norm = helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm")
        layer_norm = helper.make_node(
            "LayerNormalization",
            ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
            "layernorm",
            axis=-1,
            epsilon=0.000009999999747378752,
        )

        initializers = [  # initializers
            float_tensor("layer_norm_weight", [hidden_size]),
            float_tensor("layer_norm_bias", [hidden_size]),
        ]

        if bias > 0:
            weight_tensor = float_tensor("matmul_weight", [hidden_size, hidden_size])
            # MatMul weights is float16 when there is Cast node
            if cast_before_add_bias:
                weight_tensor.CopyFrom(
                    numpy_helper.from_array(numpy_helper.to_array(weight_tensor).astype(np.float16), weight_tensor.name)
                )
            initializers.append(weight_tensor)

            bias_tensor = float_tensor("bias", [hidden_size])
            initializers.append(bias_tensor)

        input_0 = helper.make_tensor_value_info(
            "input_0",
            TensorProto.FLOAT16 if cast_before_add_bias else TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
        )

        input_1 = helper.make_tensor_value_info(
            "input_1",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
        )

        input_2 = helper.make_tensor_value_info(
            "input_2",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
        )

        output = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
        )

        layernorm_input = helper.make_tensor_value_info(
            "layernorm_input",
            TensorProto.FLOAT,
            [batch_size, sequence_length, hidden_size],
        )

        nodes = [add_before_layer_norm, layer_norm]
        if bias > 0:
            nodes.insert(0, add_bias)
            if cast_before_add_bias:
                nodes.insert(0, cast_node)
            nodes.insert(0, matmul)

        node_name = "SkipLayerNormFusionModel"
        if bias == 0:
            graph = helper.make_graph(
                nodes,
                node_name,
                [input_1, input_2],  # inputs
                [output, layernorm_input] if add_graph_output else [output],  # outputs
                initializers,
            )
        elif bias == 1:
            graph = helper.make_graph(
                nodes,
                node_name,
                [input_0, input_2],  # inputs
                [output, layernorm_input] if add_graph_output else [output],  # outputs
                initializers,
            )
        else:
            graph = helper.make_graph(
                nodes,
                node_name,
                [input_0, input_1],  # inputs
                [output, layernorm_input] if add_graph_output else [output],  # outputs
                initializers,
            )

        onnx_opset = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
        return helper.make_model(graph, opset_imports=(onnx_opset,))

    def test_skip_layer_norm_no_graph_output(self):
        model = self.create_test_model(batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=False)
        model_name = "skip_layer_norm_add_no_graph_output.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 0,
            },
            ["input_1", "input_2", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_graph_output(self):
        model = self.create_test_model(batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=True)
        model_name = "skip_layer_norm_add_has_graph_output.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 0,
            },
            ["input_1", "input_2", "layer_norm_weight", "layer_norm_bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_graph_output_bias1(self):
        model = self.create_test_model(batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=True, bias=1)
        model_name = "skip_layer_norm_add_has_graph_output_bias1.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 0,
            },
            ["matmul_output", "input_2", "layer_norm_weight", "layer_norm_bias", "bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_graph_output_bias2(self):
        model = self.create_test_model(batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=True, bias=2)
        model_name = "skip_layer_norm_add_has_graph_output_bias1.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 0,
            },
            ["matmul_output", "input_1", "layer_norm_weight", "layer_norm_bias", "bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_graph_output_cast_bias1(self):
        model = self.create_test_model(
            batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=True, bias=1, cast_before_add_bias=True
        )
        model_name = "skip_layer_norm_add_has_graph_output_cast_bias1.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 1,
            },
            ["matmul_output_cast", "input_2", "layer_norm_weight", "layer_norm_bias", "bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_graph_output_cast_bias2(self):
        model = self.create_test_model(
            batch_size=1, sequence_length=2, hidden_size=3, add_graph_output=True, bias=2, cast_before_add_bias=True
        )
        model_name = "skip_layer_norm_add_has_graph_output_cast_bias2.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {
                "Add": 0,
                "LayerNormalization": 0,
                "SkipLayerNormalization": 1,
                "Cast": 1,
            },
            ["matmul_output_cast", "input_1", "layer_norm_weight", "layer_norm_bias", "bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def create_broadcast_test_model(
        self,
        batch_size: int = 2,
        sequence_length: int = 3,
        hidden_size: int = 4,
        skip_shape: str = "2d",  # "2d" for (seq, hidden), "3d_batch1" for (1, seq, hidden)
        skip_on_input: int = 1,  # Which Add input index gets the skip (smaller) shape
        add_graph_output: bool = False,
        simplified: bool = False,  # Use SimplifiedLayerNormalization (RMS LayerNorm) instead
    ):
        """Create a test model where one Add input has a broadcast-compatible shape."""
        if skip_shape == "2d":
            skip_dims = [sequence_length, hidden_size]
        elif skip_shape == "3d_batch1":
            skip_dims = [1, sequence_length, hidden_size]
        else:
            raise ValueError(f"Unknown skip_shape: {skip_shape}")

        full_dims = [batch_size, sequence_length, hidden_size]

        add_before_layer_norm = helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm")

        if simplified:
            layer_norm = helper.make_node(
                "SimplifiedLayerNormalization",
                ["layernorm_input", "layer_norm_weight"],
                ["output"],
                "layernorm",
                axis=-1,
                epsilon=0.000009999999747378752,
            )
            initializers = [float_tensor("layer_norm_weight", [hidden_size])]
        else:
            layer_norm = helper.make_node(
                "LayerNormalization",
                ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
                ["output"],
                "layernorm",
                axis=-1,
                epsilon=0.000009999999747378752,
            )
            initializers = [
                float_tensor("layer_norm_weight", [hidden_size]),
                float_tensor("layer_norm_bias", [hidden_size]),
            ]

        input_1_shape = full_dims if skip_on_input != 0 else skip_dims
        input_2_shape = skip_dims if skip_on_input != 0 else full_dims

        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, full_dims)]
        if add_graph_output:
            outputs.append(helper.make_tensor_value_info("layernorm_input", TensorProto.FLOAT, full_dims))

        graph = helper.make_graph(
            [add_before_layer_norm, layer_norm],
            "SkipLayerNormBroadcastModel",
            [
                helper.make_tensor_value_info("input_1", TensorProto.FLOAT, input_1_shape),
                helper.make_tensor_value_info("input_2", TensorProto.FLOAT, input_2_shape),
            ],
            outputs,
            initializers,
        )

        onnx_opset = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
        return helper.make_model(graph, opset_imports=(onnx_opset,))

    def test_skip_layer_norm_broadcast_2d_skip(self):
        """2D skip (seq, hidden) on input[1] should fuse with input order preserved."""
        model = self.create_broadcast_test_model(skip_shape="2d", skip_on_input=1)
        model_name = "skip_layer_norm_broadcast_2d.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {"Add": 0, "LayerNormalization": 0, "SkipLayerNormalization": 1, "Cast": 0},
            ["input_1", "input_2", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_broadcast_2d_skip_swapped(self):
        """2D skip (seq, hidden) on input[0] should fuse with inputs swapped."""
        model = self.create_broadcast_test_model(skip_shape="2d", skip_on_input=0)
        model_name = "skip_layer_norm_broadcast_2d_swapped.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {"Add": 0, "LayerNormalization": 0, "SkipLayerNormalization": 1, "Cast": 0},
            ["input_2", "input_1", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_broadcast_3d_batch1(self):
        """3D skip (1, seq, hidden) on input[1] should fuse with input order preserved."""
        model = self.create_broadcast_test_model(skip_shape="3d_batch1", skip_on_input=1)
        model_name = "skip_layer_norm_broadcast_3d_batch1.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {"Add": 0, "LayerNormalization": 0, "SkipLayerNormalization": 1, "Cast": 0},
            ["input_1", "input_2", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_broadcast_3d_batch1_swapped(self):
        """3D skip (1, seq, hidden) on input[0] should fuse with inputs swapped."""
        model = self.create_broadcast_test_model(skip_shape="3d_batch1", skip_on_input=0)
        model_name = "skip_layer_norm_broadcast_3d_batch1_swapped.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {"Add": 0, "LayerNormalization": 0, "SkipLayerNormalization": 1, "Cast": 0},
            ["input_2", "input_1", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_broadcast_graph_output(self):
        """Broadcast fusion should preserve Add output when it is a graph output."""
        model = self.create_broadcast_test_model(skip_shape="2d", skip_on_input=1, add_graph_output=True)
        model_name = "skip_layer_norm_broadcast_graph_output.onnx"
        onnx.save(model, model_name)
        self.verify_skip_layer_norm_fusion(
            model_name,
            {"Add": 0, "LayerNormalization": 0, "SkipLayerNormalization": 1, "Cast": 0},
            ["input_1", "input_2", "layer_norm_weight", "layer_norm_bias"],
            ["output", "", "", "layernorm_input"],
        )
        os.remove(model_name)

    def test_skip_layer_norm_broadcast_incompatible_shapes(self):
        """Incompatible broadcast shapes should not fuse.

        Uses (2,3,4) + (1,1,4): broadcastable for Add but not supported by SkipLayerNorm
        kernel (which requires skip seq_len == input seq_len).
        """
        add_node = helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm")
        layer_norm = helper.make_node(
            "LayerNormalization",
            ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
            "layernorm",
            axis=-1,
            epsilon=0.000009999999747378752,
        )
        initializers = [
            float_tensor("layer_norm_weight", [4]),
            float_tensor("layer_norm_bias", [4]),
        ]
        graph = helper.make_graph(
            [add_node, layer_norm],
            "IncompatibleShapesModel",
            [
                helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [2, 3, 4]),
                helper.make_tensor_value_info("input_2", TensorProto.FLOAT, [1, 1, 4]),  # seq_len mismatch
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])],
            initializers,
        )
        onnx_opset = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
        model = helper.make_model(graph, opset_imports=(onnx_opset,))
        model_name = "skip_layer_norm_incompatible.onnx"
        onnx.save(model, model_name)

        options = FusionOptions("bert")
        optimized_model = optimize_model(model_name, optimization_options=options, opt_level=0)
        self.assertEqual(len(optimized_model.get_nodes_by_op_type("SkipLayerNormalization")), 0)
        self.assertEqual(len(optimized_model.get_nodes_by_op_type("Add")), 1)
        os.remove(model_name)

    def test_skip_simplified_layer_norm_broadcast(self):
        """SimplifiedLayerNormalization (RMS LayerNorm) with broadcast skip should fuse."""
        model = self.create_broadcast_test_model(skip_shape="2d", skip_on_input=1, simplified=True)
        model_name = "skip_simplified_layer_norm_broadcast.onnx"
        onnx.save(model, model_name)

        options = FusionOptions("bert")
        optimized_model = optimize_model(model_name, optimization_options=options, opt_level=0)

        sln_nodes = optimized_model.get_nodes_by_op_type("SkipSimplifiedLayerNormalization")
        self.assertEqual(len(sln_nodes), 1)
        self.assertEqual(len(optimized_model.get_nodes_by_op_type("Add")), 0)
        self.assertEqual(len(optimized_model.get_nodes_by_op_type("SimplifiedLayerNormalization")), 0)
        # SimplifiedLayerNorm has no bias, so only 3 inputs: input, skip, weight
        self.assertEqual(list(sln_nodes[0].input), ["input_1", "input_2", "layer_norm_weight"])
        os.remove(model_name)


if __name__ == "__main__":
    unittest.main()
