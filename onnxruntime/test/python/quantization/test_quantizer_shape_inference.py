#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import QuantizationMode, QuantType


class TestQuantizerShapeInference(unittest.TestCase):
    def test_com_microsoft(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MatMul", ["X", "W1"], ["T1"]),
                    oh.make_node("FusedMatMul", ["T1", "W2"], ["T2"], domain="com.microsoft"),
                    oh.make_node("MatMul", ["T2", "W3"], ["T3"]),
                    oh.make_node("MatMul", ["T3", "W4"], ["Y"]),
                ],
                "name",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])],
                [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])],
                [
                    onh.from_array(np.random.randn(4, 4).astype(np.float32), "W1"),
                    onh.from_array(np.random.randn(4, 4).astype(np.float32), "W2"),
                    onh.from_array(np.random.randn(4, 4).astype(np.float32), "W3"),
                    onh.from_array(np.random.randn(4, 4).astype(np.float32), "W4"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
        )
        model_shaped = onnx.shape_inference.infer_shapes(model)
        shaped_results = set(t.name for t in model_shaped.graph.value_info)
        # every result after T1 depends on T2 coming from a node com.microsoft,
        # shape_inference cannot go beyond this point
        self.assertEqual(shaped_results, {"T1"})

        # first try: checks it raises an exception
        quantizer = ONNXQuantizer(
            model,
            False,  # per_channel
            False,  # reduce_range
            QuantizationMode.IntegerOps,  # mode
            False,  # static
            QuantType.QInt8,  #  weight_type,
            QuantType.QUInt8,  # dynamic activation only supports uint8
            None,
            [],  # nodes_to_quantize,
            [],  # nodes_to_exclude
            ["MatMul"],  # op_types_to_quantize,
            {"MatMulConstBOnly": True},  # extra_options,
            # {'DefaultTensorType': 1, }
        )

        with self.assertRaises(RuntimeError) as e:
            quantizer.quantize_model()
            self.assertIn("Unable to find data type for weight_name=", str(e))

        # second try: checks it works
        quantizer = ONNXQuantizer(
            model,
            False,  # per_channel
            False,  # reduce_range
            QuantizationMode.IntegerOps,  # mode
            False,  # static
            QuantType.QInt8,  #  weight_type,
            QuantType.QUInt8,  # dynamic activation only supports uint8
            None,
            [],  # nodes_to_quantize,
            [],  # nodes_to_exclude
            ["MatMul"],  # op_types_to_quantize,
            {
                "MatMulConstBOnly": True,
                "DefaultTensorType": 1,
            },
        )

        model = quantizer.quantize_model()
        ops = {n.op_type for n in model.graph.node}
        self.assertEqual(ops, {"Cast", "FusedMatMul", "MatMulInteger", "DynamicQuantizeLinear", "Mul"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
