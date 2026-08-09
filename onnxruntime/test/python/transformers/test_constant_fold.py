# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
from bert_model_generator import float_tensor
from onnx import TensorProto, helper, numpy_helper
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


class TestConstantFoldFusion(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 4
        self.inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch_size", self.hidden_size])]
        self.outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", self.hidden_size])]

    def verify_fusion(self, expected_model_path, original_model_path):
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        options = FusionOptions("")
        optimized_model = optimize_model(original_model_path, optimization_options=options, opt_level=0)
        optimized_model.topological_sort(is_deterministic=True)

        nodes = optimized_model.model.graph.node
        self.assertEqual(len(nodes), len(expected_model.model.graph.node))

        for i in range(len(nodes)):
            self.assertEqual(nodes[i], expected_model.model.graph.node[i])

        for expected_initializer in expected_model.model.graph.initializer:
            self.assertTrue(
                OnnxModel.has_same_value(
                    optimized_model.get_initializer(expected_initializer.name), expected_initializer
                )
            )

    def tearDown(self):
        names = [
            "test_folding_initializer_data_original",
            "test_folding_initializer_data_expected",
            "test_folding_transpose_transpose_original",
            "test_folding_transpose_transpose_expected",
        ]
        for name in names:
            path = os.path.join(os.path.dirname(__file__), f"{name}.onnx")
            if os.path.exists(path):
                os.remove(path)

    def make_model(self, name, nodes, initializers):
        graph = helper.make_graph(nodes, "graph", self.inputs, self.outputs, initializers)
        opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
        model = helper.make_model(graph, opset_imports=(opsetid,))
        onnx.save_model(model, os.path.join(os.path.dirname(__file__), f"{name}.onnx"))

    def test_folding_initializer_data(self):
        # Create original model
        initializers = [float_tensor("w", [self.hidden_size, self.hidden_size], random=True)]
        nodes = [
            helper.make_node("Transpose", inputs=["w"], outputs=["w.T"], name="Transpose_0", perm=[0, 1]),
            helper.make_node("MatMul", inputs=["x", "w.T"], outputs=["y"], name="MatMul_0"),
        ]
        original_name = "test_folding_initializer_data_original"
        self.make_model(original_name, nodes, initializers)

        # Create expected model
        data = numpy_helper.to_array(initializers[0])
        data = data.T
        initializers = [
            helper.make_tensor("w", initializers[0].data_type, dims=[data.shape[0], data.shape[1]], vals=data)
        ]
        nodes = [
            helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="MatMul_0"),
        ]
        expected_name = "test_folding_initializer_data_expected"
        self.make_model(expected_name, nodes, initializers)

        # Compare models
        self.verify_fusion(
            os.path.join(os.path.dirname(__file__), f"{expected_name}.onnx"),
            os.path.join(os.path.dirname(__file__), f"{original_name}.onnx"),
        )

    def test_folding_transpose_transpose(self):
        # Create original model
        initializers = [helper.make_tensor("unit_offset", TensorProto.FLOAT, dims=[1], vals=[1.0])]
        nodes = [
            helper.make_node("Transpose", inputs=["x"], outputs=["x.T"], name="Transpose_0", perm=[0, 1]),
            helper.make_node("Transpose", inputs=["x.T"], outputs=["x.T.T"], name="Transpose_1", perm=[0, 1]),
            helper.make_node("Add", inputs=["x.T.T", "unit_offset"], outputs=["y"], name="Add_0"),
        ]
        original_name = "test_folding_transpose_transpose_original"
        self.make_model(original_name, nodes, initializers)

        # Create expected model
        nodes = [
            helper.make_node("Add", inputs=["x", "unit_offset"], outputs=["y"], name="Add_0"),
        ]
        expected_name = "test_folding_transpose_transpose_expected"
        self.make_model(expected_name, nodes, initializers)

        # Compare models
        self.verify_fusion(
            os.path.join(os.path.dirname(__file__), f"{expected_name}.onnx"),
            os.path.join(os.path.dirname(__file__), f"{original_name}.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
