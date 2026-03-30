# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Test that saving an ORT-optimized model preserves value_info for
intermediate tensors created during function inlining.

ORT inlines ONNX model-local functions during optimization.  Prior to
the fix, the intermediate NodeArgs produced by inlined nodes were not
included in the saved model's ``value_info``, even though ORT had
inferred their types and shapes.  This test verifies that the saved
optimized model contains value_info entries for those intermediates.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def _make_model_with_function() -> onnx.ModelProto:
    """Build a small model whose graph calls a model-local function.

    The function ``MyAdd`` computes ``Y = A + B`` via an intermediate
    tensor.  After ORT inlines the function, the intermediate should
    get a value_info entry in the saved optimized model.
    """
    # --- Define a simple function: MyAdd(A, B) -> Y ---
    # Body: tmp = Add(A, B); Y = Identity(tmp)
    # The Identity creates a named intermediate that, after inlining,
    # should appear in value_info.
    func_nodes = [
        helper.make_node("Add", ["A", "B"], ["tmp"]),
        helper.make_node("Identity", ["tmp"], ["Y"]),
    ]
    my_func = helper.make_function(
        domain="test.domain",
        fname="MyAdd",
        inputs=["A", "B"],
        outputs=["Y"],
        nodes=func_nodes,
        opset_imports=[helper.make_opsetid("", 18)],
    )

    # --- Main graph calls the function ---
    X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [2, 3])
    X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [2, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    # Call MyAdd to produce mid, then Relu to produce Z
    call_node = helper.make_node("MyAdd", ["X1", "X2"], ["mid"], domain="test.domain")
    relu_node = helper.make_node("Relu", ["mid"], ["Z"])

    graph = helper.make_graph([call_node, relu_node], "main", [X1, X2], [Z])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("test.domain", 1),
        ],
        functions=[my_func],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class TestOptimizedModelValueInfo(unittest.TestCase):
    def test_inlined_function_intermediates_have_value_info(self):
        """Verify that function-inlined intermediates appear in value_info."""
        model = _make_model_with_function()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input_model.onnx")
            optimized_path = os.path.join(tmpdir, "optimized_model.onnx")

            onnx.save(model, input_path)

            # Load with ORT — this inlines the function — and save optimized
            so = ort.SessionOptions()
            so.optimized_model_filepath = optimized_path
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ = ort.InferenceSession(input_path, so, providers=["CPUExecutionProvider"])

            self.assertTrue(os.path.isfile(optimized_path))

            # Load the saved optimized model and inspect value_info
            opt_model = onnx.load(optimized_path)

            # The original model had a function "MyAdd" with an
            # intermediate "tmp".  After inlining, all functions should
            # be gone and the intermediate should exist as a node output
            # in the main graph.
            self.assertEqual(len(opt_model.functions), 0, "Functions should be inlined")

            # Collect all intermediate node output names (not graph
            # inputs/outputs).
            graph_input_names = {inp.name for inp in opt_model.graph.input}
            graph_output_names = {out.name for out in opt_model.graph.output}
            intermediate_names = set()
            for node in opt_model.graph.node:
                for out_name in node.output:
                    if out_name and out_name not in graph_input_names and out_name not in graph_output_names:
                        intermediate_names.add(out_name)

            # There must be at least one intermediate (from the inlined
            # function).
            self.assertGreater(len(intermediate_names), 0, "Expected inlined intermediates")

            # Verify that every intermediate has a value_info entry with
            # type information.
            vi_names = {vi.name for vi in opt_model.graph.value_info}
            missing = intermediate_names - vi_names
            self.assertEqual(
                missing,
                set(),
                f"Intermediate node outputs missing from value_info: {missing}",
            )

            # Verify that value_info entries have type information
            for vi in opt_model.graph.value_info:
                if vi.name in intermediate_names:
                    self.assertTrue(
                        vi.type.HasField("tensor_type"),
                        f"value_info '{vi.name}' should have tensor type",
                    )


if __name__ == "__main__":
    unittest.main()
