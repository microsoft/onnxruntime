#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxruntime import quantization


class TestQuantMinRealRangeOption(unittest.TestCase):
    def setUp(self):
        self.qdq_model_name = "model_qdq_u8.onnx"

        # Set up activations/weights with zero value ranges (i.e., rmax - rmax == 0).
        self.zero_range_activations = [
            np.zeros([1, 2, 32, 32], dtype="float32"),
        ]

        self.zero_range_weights = np.zeros([1, 2, 2, 2], dtype="float32")

    def perform_quantization(self, activations, weight, quant_min_rrange):
        # One-layer convolution model to be quantized with uint8 activations and uint8 weights.
        act = helper.make_tensor_value_info("ACT", TensorProto.FLOAT, activations[0].shape)
        helper.make_tensor_value_info("WGT", TensorProto.FLOAT, weight.shape)
        res = helper.make_tensor_value_info("RES", TensorProto.FLOAT, [None, None, None, None])
        wgt_init = numpy_helper.from_array(weight, "WGT")
        conv_node = onnx.helper.make_node("Conv", ["ACT", "WGT"], ["RES"])
        graph = helper.make_graph([conv_node], "test", [act], [res], initializer=[wgt_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        onnx.save(model, "model.onnx")

        # Quantize model
        class DummyDataReader(quantization.CalibrationDataReader):
            def __init__(self):
                self.iterator = ({"ACT": act} for act in activations)

            def get_next(self):
                return next(self.iterator, None)

        quantization.quantize_static(
            model_input="model.onnx",
            model_output=self.qdq_model_name,
            calibration_data_reader=DummyDataReader(),
            quant_format=quantization.QuantFormat.QDQ,
            activation_type=quantization.QuantType.QUInt8,
            weight_type=quantization.QuantType.QUInt8,
            op_types_to_quantize=["Conv"],
            extra_options={"QuantMinRealRange": quant_min_rrange},
        )

        # Extract quantization parameters: scales and zero points for activations and weights.
        model = onnx.load(self.qdq_model_name)
        act_zp = next(init for init in model.graph.initializer if init.name == "ACT_zero_point").int32_data[0]
        act_sc = next(init for init in model.graph.initializer if init.name == "ACT_scale").float_data[0]
        wgt_zp = next(init for init in model.graph.initializer if init.name == "WGT_zero_point").int32_data[0]
        wgt_sc = next(init for init in model.graph.initializer if init.name == "WGT_scale").float_data[0]

        # Return quantization parameters
        return act_zp, act_sc, wgt_zp, wgt_sc

    def test_default(self):
        """
        Test default behavior without specifing the QuantMinRealRange option.
        """
        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.zero_range_activations,
            self.zero_range_weights,
            quant_min_rrange=None,  # default behavior
        )

        # No minimum real range is set. Expect default behavior (scale = 1.0, zp = 0)
        self.assertEqual(act_zp, 0)
        self.assertEqual(act_sc, 1.0)
        self.assertEqual(wgt_zp, 0)
        self.assertEqual(wgt_sc, 1.0)

    def test_min_real_range(self):
        """
        Test a QuantMinRealRange value of 0.0001.
        """
        quant_min_rrange = 0.0001

        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.zero_range_activations,
            self.zero_range_weights,
            quant_min_rrange=quant_min_rrange,
        )

        expected_scale = np.float32(quant_min_rrange / 255)

        # Minimum floating-point range is set. Expect small scale values.
        self.assertEqual(act_zp, 0)
        self.assertEqual(act_sc, expected_scale)
        self.assertEqual(wgt_zp, 0)
        self.assertEqual(wgt_sc, expected_scale)


if __name__ == "__main__":
    unittest.main()
