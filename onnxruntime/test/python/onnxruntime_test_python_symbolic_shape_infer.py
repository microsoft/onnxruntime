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
        def get_initializer(name):
            valuemap = {"zero": 0, "one": 1, "two": 2, "ten": 10, "intmax": 2**32}
            value = -valuemap[name[4:]] if name.startswith("neg_") else valuemap[name]
            return onnx.helper.make_tensor(name, TensorProto.INT64, [1], [value])

        initializers = [get_initializer(name) for name in  ["zero", "one", "two", "ten", "intmax", "neg_intmax", "neg_one", "neg_ten"]]
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
            initializer=initializers
        )
        model = SymbolicShapeInference.infer_shapes(onnx.helper.make_model(graph_def))
        output = unique_element(model.graph.output)
        shape = [d.dim_param if d.dim_param else d.dim_value for d in output.type.tensor_type.shape.dim]
        self.assertEqual(shape, ["B", expected_output_dim])

    def test_numeric_negative_indices_forward(self):
        self.check_slice_of_concat(["M"], "-ten", "-one", "one", 9)

    def test_numeric_negative_indices_backward(self):
        self.check_slice_of_concat(["M"], "-one", "-ten", "-one", 9)

    def test_symbolic_end_index(self):
        self.check_slice_of_concat(["M", "N"], "zero", "M", "one", "M")

    def test_symbolic_negative_start_index(self):
        self.check_slice_of_concat(["M", "N"], "-N", "intmax", "one", "N")

    def test_non_unit_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "two", "N")

    def test_symbolic_step(self):
        self.check_slice_of_concat(["N", "N"], "zero", "intmax", "N", "floor(-1/N) + 3")

    def test_symbolic_negative_step(self):
        self.check_slice_of_concat(["N", "N"], "-one", "-intmax", "-N", "floor(-1/N) + 3")

    def test_flip(self):
        self.check_slice_of_concat(["N"], "-one", "-intmax", "-one", "N")

    def test_flip_of_concat(self):
        self.check_slice_of_concat(["N", "N", "N"], "-one", "-intmax", "-one", "3*N")


if __name__ == '__main__':
    unittest.main()
