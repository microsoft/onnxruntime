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
from op_test_utils import TestDataFeeds, check_op_type_count

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestQDQGatherElements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.gatherelems_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_test_model_static_indices(
        self,
        inp_shape: list[int],
        indices_data: np.ndarray,
        axis: int = 0,
    ):
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, inp_shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, None)
        indices = onnx.numpy_helper.from_array(indices_data, "indices")

        gatherelems_node = onnx.helper.make_node(
            "GatherElements", ["input_0", "indices"], ["output_0"], axis=axis, name="GatherElems0"
        )
        graph = onnx.helper.make_graph(
            [gatherelems_node],
            "GatherElemsf32",
            [input_0],
            [output_0],
            initializer=[indices],
        )
        opset_imports = [onnx.helper.make_opsetid("", 21)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        return onnx.shape_inference.infer_shapes(model)

    def test_qdq_gatherelems_static_indices(self):
        """
        Test quantization of GatherElements with static indices.
        """
        float_model_path = os.path.join(self._tmp_dir_path, "gather_elems.f32.onnx")
        qdq_model_path = os.path.join(self._tmp_dir_path, "gather_elems.qdq.onnx")

        inp_shape = [3, 3]
        indices_data = np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int64)
        float_model = self.build_test_model_static_indices(inp_shape, indices_data, axis=0)

        onnx.checker.check_model(float_model, True)
        onnx.save_model(float_model, float_model_path)

        input_data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        input_data2 = np.array([[8, 2, 9], [4, 5, 6], [7, 8, 1]], dtype=np.float32)
        data_reader = TestDataFeeds([{"input_0": input_data1}, {"input_0": input_data2}])

        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=[node.op_type for node in float_model.graph.node],
            extra_options={
                "ForceQuantizeNoInputCheck": True,
            },
        )

        qdq_node_counts = {"QuantizeLinear": 2, "DequantizeLinear": 2}
        check_op_type_count(self, qdq_model_path, **qdq_node_counts)

        qdq_model = onnx.load_model(qdq_model_path)
        onnx.checker.check_model(qdq_model, True)

        initializers = {init.name: init for init in qdq_model.graph.initializer}

        zp_name = "input_0_zero_point"
        scale_name = "input_0_scale"
        self.assertIn(zp_name, initializers)
        self.assertIn(scale_name, initializers)

        # Check that all Q/DQ nodes use the same scale and zero-point
        for node in qdq_model.graph.node:
            if node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear":
                self.assertEqual(node.input[1], scale_name)
                self.assertEqual(node.input[2], zp_name)


if __name__ == "__main__":
    unittest.main()
