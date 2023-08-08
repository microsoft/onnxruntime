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


class TestSymmetricFlag(unittest.TestCase):
    def setUp(self):
        # Set up symmetrically and asymmetrically disributed values for activations
        self.symmetric_activations = [
            -1 * np.ones([1, 2, 32, 32], dtype="float32"),
            +1 * np.ones([1, 2, 32, 32], dtype="float32"),
        ]
        self.asymmetric_activations = [
            -1 * np.ones([1, 2, 32, 32], dtype="float32"),
            +2 * np.ones([1, 2, 32, 32], dtype="float32"),
        ]

        # Set up symmetrically and asymmetrically disributed values for weights
        self.symmetric_weights = np.concatenate(
            (
                -1 * np.ones([1, 1, 2, 2], dtype="float32"),
                +1 * np.ones([1, 1, 2, 2], dtype="float32"),
            ),
            axis=1,
        )
        self.asymmetric_weights = np.concatenate(
            (
                -1 * np.ones([1, 1, 2, 2], dtype="float32"),
                +2 * np.ones([1, 1, 2, 2], dtype="float32"),
            ),
            axis=1,
        )

    def perform_quantization(self, activations, weight, act_sym, wgt_sym):
        # One-layer convolution model
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
            model_output="quantized-model.onnx",
            calibration_data_reader=DummyDataReader(),
            quant_format=quantization.QuantFormat.QOperator,
            activation_type=quantization.QuantType.QInt8,
            weight_type=quantization.QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul"],
            extra_options={"WeightSymmetric": wgt_sym, "ActivationSymmetric": act_sym},
        )

        # Extract quantization parameters: scales and zero points for activations, weights, and results
        model = onnx.load("quantized-model.onnx")
        act_zp = [init for init in model.graph.initializer if init.name == "ACT_zero_point"][0].int32_data[0]
        act_sc = [init for init in model.graph.initializer if init.name == "ACT_scale"][0].float_data[0]
        wgt_zp = [init for init in model.graph.initializer if init.name == "WGT_zero_point"][0].int32_data[0]
        wgt_sc = [init for init in model.graph.initializer if init.name == "WGT_scale"][0].float_data[0]

        # Return quantization parameters
        return act_zp, act_sc, wgt_zp, wgt_sc

    def test_0(self):
        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.asymmetric_activations,
            self.asymmetric_weights,
            act_sym=True,
            wgt_sym=True,
        )

        # Calibration activations are asymmetric, but activation
        # symmetrization flag is set to True, hence expect activation zero
        # point = 0
        self.assertEqual(act_zp, 0)

        # Weights are asymmetric, but weight symmetrization flag is set to
        # True, hence expect weight zero point = 0
        self.assertEqual(wgt_zp, 0)

    def test_1(self):
        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.asymmetric_activations,
            self.asymmetric_weights,
            act_sym=False,
            wgt_sym=False,
        )

        # Calibration activations are asymmetric, symmetrization flag not
        # set, hence expect activation zero point != 0
        self.assertNotEqual(act_zp, 0)

        # Weights are asymmetric, weight symmetrization flag is set to
        # False, hence expect weight zero point != 0
        self.assertNotEqual(wgt_zp, 0)

    def test_2(self):
        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.symmetric_activations,
            self.symmetric_weights,
            act_sym=True,
            wgt_sym=True,
        )

        # Calibration activations are symmetric, hence expect activation
        # zero point == 0 (regardless of flag)
        self.assertEqual(act_zp, 0)

        # Weights are symmetric, hence expect weight
        # zero point == 0 (regardless of flag)
        self.assertEqual(wgt_zp, 0)

    def test_3(self):
        act_zp, act_sc, wgt_zp, wgt_sc = self.perform_quantization(
            self.symmetric_activations,
            self.symmetric_weights,
            act_sym=False,
            wgt_sym=False,
        )

        # Calibration activations are symmetric, hence expect activation
        # zero point == 0 (regardless of flag)
        self.assertEqual(act_zp, 0)

        # Weights are symmetric, hence expect weight
        # zero point == 0 (regardless of flag)
        self.assertEqual(wgt_zp, 0)


if __name__ == "__main__":
    unittest.main()
