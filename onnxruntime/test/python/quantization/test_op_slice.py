#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from op_test_utils import (
    TestDataFeeds,
    check_model_correctness,
    check_op_type_count,
    get_tensor_consumers_and_producers,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestQDQSlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.slice_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_slice_model(
        self,
        input_shape: list[int],
        input_tensor_type: onnx.TensorProto.DataType,
        starts: list[int],
        ends: list[int],
        axes: list[int] | None = None,
        steps: list[int] | None = None,
    ) -> onnx.ModelProto:
        """
        Returns an onnx.ModelProto with a single Slice operator.
        """
        input_0 = onnx.helper.make_tensor_value_info("input_0", input_tensor_type, input_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", input_tensor_type, None)

        initializers = [
            onnx.numpy_helper.from_array(np.array(starts, dtype=np.int64), "starts"),
            onnx.numpy_helper.from_array(np.array(ends, dtype=np.int64), "ends"),
        ]
        slice_input_names = ["input_0", "starts", "ends"]

        if axes:
            initializers.append(onnx.numpy_helper.from_array(np.array(axes, dtype=np.int64), "axes"))
            slice_input_names.append("axes")

        if steps:
            if not axes:
                slice_input_names.append("")  # Empty axes input.
            initializers.append(onnx.numpy_helper.from_array(np.array(steps, dtype=np.int64), "steps"))
            slice_input_names.append("steps")

        slice_node = onnx.helper.make_node("Slice", slice_input_names, ["output_0"], name="Slice0")

        graph = onnx.helper.make_graph(
            [slice_node],
            "SliceGraph",
            [input_0],
            [output_0],
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_qdq_slice_qparams(self):
        """
        Test that QDQ Slice has equal scale/zero-point for its input and output.
        """
        test_configs = [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]

        for onnx_tensor_type in test_configs:
            with self.subTest(onnx_tensor_type=onnx_tensor_type):
                label = f"{onnx.TensorProto.DataType.Name(onnx_tensor_type)}"
                float_model_path = os.path.join(self._tmp_dir_path, f"slice.{label}.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"slice.{label}.qdq.onnx")

                input_shape = [2, 4]
                float_model = self.build_slice_model(
                    input_shape=input_shape,
                    input_tensor_type=onnx_tensor_type,
                    starts=[1, 0],
                    ends=[2, 3],
                    axes=None,
                    steps=[1, 2],
                )
                onnx.save_model(float_model, float_model_path)

                # Create a data reader
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_tensor_type)
                input_data_list = [
                    {"input_0": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np_dtype)},
                    {"input_0": np.array([[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]], dtype=np_dtype)},
                ]
                data_reader = TestDataFeeds(input_data_list)

                # quantize model to QDQ
                quantize_static(
                    float_model_path,
                    qdq_model_path,
                    data_reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    extra_options={"ForceQuantizeNoInputCheck": True},
                )
                expected_op_counts = {"DequantizeLinear": 2, "QuantizeLinear": 2, "Slice": 1}
                check_op_type_count(self, qdq_model_path, **expected_op_counts)

                data_reader.rewind()
                check_model_correctness(self, float_model_path, qdq_model_path, data_reader.get_next())

                qdq_model = onnx.load_model(qdq_model_path)

                slice_node = next((node for node in qdq_model.graph.node if node.op_type == "Slice"), None)
                self.assertNotEqual(slice_node, None)
                self.assertEqual(slice_node.op_type, "Slice")

                # Get the parent and child nodes of the Slice and check that they are DQ/Q.
                consumers, producers = get_tensor_consumers_and_producers(qdq_model)
                input_dq_node = producers.get(slice_node.input[0], None)
                self.assertNotEqual(input_dq_node, None)
                self.assertEqual(input_dq_node.op_type, "DequantizeLinear")

                output_q_node = consumers.get(slice_node.output[0], [None])[0]
                self.assertNotEqual(output_q_node, None)
                self.assertEqual(output_q_node.op_type, "QuantizeLinear")

                # Check that the Slice's input DQ uses the same scale/zp as the Slice's output Q.
                self.assertEqual(input_dq_node.input[1], output_q_node.input[1])
                self.assertEqual(input_dq_node.input[2], output_q_node.input[2])


if __name__ == "__main__":
    unittest.main()
