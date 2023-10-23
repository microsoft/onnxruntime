# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


def float_tensor(name: str, shape: List[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [np.random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


class TestSimplifiedLayerNormFusion(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 5
        self.batch_size = 2
        self.sequence_length = 8
        self.hidden_size = 16
        self.epsilon = 0.000009999999747378752

    def verify_fusion(self, expected_model_path, original_model_path):
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        options = FusionOptions("gpt2")
        optimized_model = optimize_model(original_model_path, optimization_options=options)
        optimized_model.topological_sort(is_deterministic=True)

        self.assertTrue(str(expected_model.model.graph), str(optimized_model.model.graph))

    def create_initializers(self, use_embed_weight: bool = False):
        initializers = [
            helper.make_tensor("Two", TensorProto.FLOAT, [1], np.array([2], dtype=np.float32)),
            helper.make_tensor("epsilon", TensorProto.FLOAT, [1], np.array([self.epsilon], dtype=np.float32)),
            helper.make_tensor("One", TensorProto.FLOAT, [1], np.array([1], dtype=np.float32)),
            float_tensor("scale", [self.hidden_size]),
        ]
        if use_embed_weight:
            initializers = [  # noqa: RUF005
                float_tensor("embed_weight", [self.vocab_size, self.hidden_size])
            ] + initializers
        return initializers

    def create_inputs_and_outputs(self, start_node_type: str):
        inputs, start_node = None, None
        if start_node_type == "Add":
            start_node = helper.make_node(
                "Add",
                inputs=["input_0", "input_1"],
                outputs=["D"],
                name="Add_0",
            )
            input_0 = helper.make_tensor_value_info(
                "input_0",
                TensorProto.FLOAT,
                [self.batch_size, self.sequence_length, self.hidden_size],
            )
            input_1 = helper.make_tensor_value_info(
                "input_1",
                TensorProto.FLOAT,
                [self.batch_size, self.sequence_length, self.hidden_size],
            )
            inputs = [input_0, input_1]
        elif start_node_type == "Gather":
            start_node = helper.make_node(
                "Gather",
                inputs=["embed_weight", "input_0"],
                outputs=["D"],
                name="Gather_0",
            )
            input_0 = helper.make_tensor_value_info(
                "input_0",
                TensorProto.INT64,
                [self.batch_size, self.sequence_length],
            )
            inputs = [input_0]
        else:
            # start_node_type is a graph input
            assert start_node_type == "GraphInput"
            input_0 = helper.make_tensor_value_info(
                "D",
                TensorProto.FLOAT,
                [self.batch_size, self.sequence_length, self.hidden_size],
            )
            inputs = [input_0]

        outputs = [
            helper.make_tensor_value_info(
                "output_0",
                TensorProto.FLOAT,
                [self.batch_size, self.sequence_length, self.hidden_size],
            )
        ]
        return inputs, outputs, start_node

    def create_fused_model(self, start_node_type: str, initializers: List[TensorProto]):
        inputs, outputs, start_node = self.create_inputs_and_outputs(start_node_type)

        sln_node = helper.make_node(
            "SimplifiedLayerNormalization",
            inputs=[start_node.output[0] if start_node is not None else "D", initializers[0].name],
            outputs=[outputs[0].name],
            axis=-1,
            epsilon=initializers[2].float_data[0],
            stash_type=1,
        )

        graph = helper.make_graph(
            nodes=[sln_node] + ([] if start_node is None else [start_node]),
            name="SimplifiedLayerNorm_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    # Notation follows https://onnx.ai/onnx/operators/onnx__LayerNormalization.html#summary
    def create_test_model(self, start_node_type: str, first_parent_idx: int, initializers: List[TensorProto]):
        end_node = helper.make_node(
            "Mul",
            inputs=["scale", "Normalized"] if first_parent_idx == 1 else ["Normalized", "scale"],
            outputs=["output_0"],
            name="Mul_1",
        )
        mul_node = helper.make_node(
            "Mul",
            inputs=["D", "InvStdDev"],
            outputs=["Normalized"],
            name="Mul_0",
        )
        div_node = helper.make_node(
            "Div",
            inputs=["One", "StdDev"],
            outputs=["InvStdDev"],
            name="Div_0",
        )
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=["VarEps"],
            outputs=["StdDev"],
            name="Sqrt_0",
        )
        add_node = helper.make_node(
            "Add",
            inputs=["Var", "epsilon"],
            outputs=["VarEps"],
            name="Add_1",
        )
        reducemean_node = helper.make_node(
            "ReduceMean",
            inputs=["DD"],
            outputs=["Var"],
            name="ReduceMean_0",
        )
        pow_node = helper.make_node(
            "Pow",
            inputs=["D", "Two"],
            outputs=["DD"],
            name="Pow_0",
        )

        inputs, outputs, start_node = self.create_inputs_and_outputs(start_node_type)

        main_nodes = [pow_node, reducemean_node, add_node, sqrt_node, div_node, mul_node, end_node]
        graph = helper.make_graph(
            nodes=main_nodes + ([] if start_node is None else [start_node]),
            name="SimplifiedLayerNorm_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model

    def check_models(self, start_node_type: str, first_parent_idx: int, initializers: List[TensorProto]):
        expected_model_filename = "expected_model.onnx"
        expected_model = self.create_fused_model(start_node_type, initializers)
        onnx.save(expected_model, expected_model_filename)

        original_model_filename = "original_model.onnx"
        original_model = self.create_test_model(start_node_type, first_parent_idx, initializers)
        onnx.save(original_model, original_model_filename)

        self.verify_fusion(expected_model_filename, original_model_filename)
        os.remove(expected_model_filename)
        os.remove(original_model_filename)

    # sim_ln_nodes_1
    def test_simplified_layernorm_add_idx1(self):
        start_node_type = "Add"
        first_parent_idx = 1
        initializers = self.create_initializers()
        self.check_models(start_node_type, first_parent_idx, initializers)

    # sim_ln_nodes_2
    def test_simplified_layernorm_gather_idx1(self):
        start_node_type = "Gather"
        first_parent_idx = 1
        initializers = self.create_initializers(use_embed_weight=True)
        self.check_models(start_node_type, first_parent_idx, initializers)

    # sim_ln_nodes_3
    def test_simplified_layernorm_add_idx0(self):
        start_node_type = "Add"
        first_parent_idx = 0
        initializers = self.create_initializers()
        self.check_models(start_node_type, first_parent_idx, initializers)

    # sim_ln_nodes_4
    def test_simplified_layernorm_gather_graph_input(self):
        start_node_type = "GraphInput"
        first_parent_idx = 0
        initializers = self.create_initializers()
        self.check_models(start_node_type, first_parent_idx, initializers)


if __name__ == "__main__":
    unittest.main()
