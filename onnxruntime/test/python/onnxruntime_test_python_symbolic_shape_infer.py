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
    def check_slice_of_concat(self, input_dims, start, end, step, expected_output_dim):
        _dimstrmap = {dim: f"dim{i}" for i, dim in enumerate(input_dims)}
        def dimstrmap(dim):
            return _dimstrmap.get(dim, dim)
        zero = onnx.helper.make_tensor("zero", TensorProto.INT64, [1], [0])
        one = onnx.helper.make_tensor("one", TensorProto.INT64, [1], [1])
        two = onnx.helper.make_tensor("two", TensorProto.INT64, [1], [2])
        intmax = onnx.helper.make_tensor("intmax", TensorProto.INT64, [1], [2**32])
        neg_one = onnx.helper.make_tensor("neg_one", TensorProto.INT64, [1], [-1])
        neg_intmax = onnx.helper.make_tensor("neg_intmax", TensorProto.INT64, [1], [-2**32])
        inputs = []
        nodes = []
        for i, dim in enumerate(input_dims):
            inputs.append(onnx.helper.make_tensor_value_info(f"t{i}", TensorProto.FLOAT, ["B", dim]))
            nodes.extend(
                [
                    onnx.helper.make_node("Shape", [f"t{i}"], [f"shape{i}"]),
                    onnx.helper.make_node(
                        "Slice",
                        [f"shape{i}", "one", "two", "zero", "one"],
                        [f"dim{i}"]
                    ),
                    onnx.helper.make_node("Neg", [f"dim{i}"], [f"neg_dim{i}"])
                ]
            )

        def make_concat_dims(concat_name, dims):
            dims = [
                f"neg_{dimstrmap(dim[1:])}" if dim.startswith("-") else dimstrmap(dim) for dim in dims
            ]
            return onnx.helper.make_node("Concat", dims, [concat_name], axis=0)

        nodes.extend(
            [
                onnx.helper.make_node("Concat", [inp.name for inp in inputs], ["concat"], axis=1),
                make_concat_dims("starts", ["zero", start]),
                make_concat_dims("ends", ["intmax", end]),
                make_concat_dims("axes", ["zero", "one"]),
                make_concat_dims("steps", ["one", step]),
                onnx.helper.make_node("Slice", ["concat", "starts", "ends", "axes", "steps"], ["output"])
            ]
        )
        output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, ["d1", "d2"])
        graph_def = onnx.helper.make_graph(
            nodes,
            "graph",
            inputs,
            [output],
            initializer=[zero, one, two, intmax, neg_one, neg_intmax]
        )
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output = unique_element(model.graph.output)
        shape = [d.dim_param for d in output.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["B", expected_output_dim])

    def test_symbolic_negative_start_index(self):
        self.check_slice_of_concat(["M", "N"], "-N", "intmax", "one", "N")

    def test_non_unit_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "two", "N")

    def test_symbolic_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "N", "floor((3*N - 1)/N)")

    def test_symbolic_negative_step(self):
        self.check_slice_of_concat(["N", "N"], "-one", "-intmax", "-N", "floor(-(1 - 3*N)/N)")

    def test_flip(self):
        self.check_slice_of_concat(["N"], "-one", "-intmax", "-one", "N")

    def test_flip_of_concat(self):
        self.check_slice_of_concat(["N", "N", "N"], "-one", "-intmax", "-one", "3*N")


if __name__ == '__main__':
    unittest.main()
