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

import onnx

import onnxruntime as ort

# A minimal model with a model-local function ``MyAdd``.
# The function body has an intermediate tensor ``tmp`` (output of Add,
# input to Identity).  After ORT inlines the function, ``tmp`` becomes
# a main-graph intermediate (renamed ``_inlfunc_MyAdd_tmp``) and should
# appear in the saved model's value_info.
_MODEL_TEXT = """\
<ir_version: 8, opset_import: ["" : 18, "test.domain" : 1]>
agraph (float[2, 3] X1, float[2, 3] X2) => (float[2, 3] Z) {
    mid = test.domain.MyAdd(X1, X2)
    Z = Relu(mid)
}

<domain: "test.domain", opset_import: ["" : 18]>
MyAdd (A, B) => (Y) {
    tmp = Add(A, B)
    Y = Identity(tmp)
}
"""


class TestOptimizedModelValueInfo(unittest.TestCase):
    def test_inlined_function_intermediates_have_value_info(self):
        """Verify that function-inlined intermediates appear in value_info."""
        model = onnx.parser.parse_model(_MODEL_TEXT)
        onnx.checker.check_model(model)

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
