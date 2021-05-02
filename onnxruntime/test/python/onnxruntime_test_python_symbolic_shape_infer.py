# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import onnx
from onnx import AttributeProto, TensorProto, GraphProto
import os
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from pathlib import Path
import unittest

def unique_element(lst):
    assert len(lst) == 1
    return lst[0]

class TestSymbolicShapeInference(unittest.TestCase):
    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, '..', 'models')
        for filename in Path(test_model_dir).rglob('*.onnx'):
            if filename.name.startswith('.'):
                continue  # skip some bad model files
            print("Running symbolic shape inference on : " + str(filename))
            SymbolicShapeInference.infer_shapes(
                in_mp=onnx.load(str(filename)),
                auto_merge=True,
                int_max=100000,
                guess_output_rank=True)

class TestSymbolicShapeInferenceForSlice(unittest.TestCase):
    def test_flip(self):
        starts = onnx.helper.make_tensor("starts", TensorProto.INT64, [1], [-1])
        ends = onnx.helper.make_tensor("ends", TensorProto.INT64, [1], [-2**32])
        axes = onnx.helper.make_tensor("axes", TensorProto.INT64, [1], [0])
        steps = onnx.helper.make_tensor("steps", TensorProto.INT64, [1], [-1])
        node_def = onnx.helper.make_node(
            "Slice",
            ["data", "starts", "ends", "axes", "steps"], ["output"]
        )
        data = onnx.helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N"])
        output = onnx.helper.make_tensor_value_info("data", TensorProto.FLOAT, ["output_dim"])
        graph_def = onnx.helper.make_graph([node_def], "graph", [data], [output], initializer=[starts, ends, axes, steps])
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output = unique_element(model.graph.output)
        dim = unique_element(output.type.tensor_type.shape.dim)
        self.assertEqual(dim.dim_param, "N")


if __name__ == '__main__':
    unittest.main()
