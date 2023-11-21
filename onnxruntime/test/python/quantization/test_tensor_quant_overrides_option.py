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
from onnxruntime.quantization.quant_utils import compute_scale_zp, get_qmin_qmax_for_qType


class TestTensorQuantOverridesOption(unittest.TestCase):
    def setUp(self):
        self.activations = [
            np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype="float32"),
        ]

        self.weight = np.array([[[-1.0, -2.0], [1.0, 2.0]]], dtype="float32")
        self.default_act_qtype = quantization.QuantType.QUInt8
        self.default_wgt_qtype = quantization.QuantType.QUInt8

        self.expected_zp_scales = {
            "INP": (0, np.float32(0.0235294122248888)),
            "SIG_OUT": (0, np.float32(0.003911871928721666)),
            "OUT": (0, np.float32(0.001866568811237812)),
            "WGT": (128, np.float32(0.01568627543747425)),
        }

    def perform_qdq_quantization(self, output_model_name, tensor_quant_overrides=None):
        #    (input)
        #       |
        #    Sigmoid
        #       |
        #     Conv
        #       |
        #    (output)

        inp = helper.make_tensor_value_info("INP", TensorProto.FLOAT, self.activations[0].shape)
        sigmoid_node = onnx.helper.make_node("Sigmoid", ["INP"], ["SIG_OUT"])

        out = helper.make_tensor_value_info("OUT", TensorProto.FLOAT, [None, None, None])
        wgt_init = numpy_helper.from_array(self.weight, "WGT")
        conv_node = onnx.helper.make_node("Conv", ["SIG_OUT", "WGT"], ["OUT"])

        graph = helper.make_graph([sigmoid_node, conv_node], "test", [inp], [out], initializer=[wgt_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        onnx.save(model, "model.onnx")

        # Quantize model
        class DummyDataReader(quantization.CalibrationDataReader):
            def __init__(self, activations):
                self.iterator = ({"INP": act} for act in activations)

            def get_next(self):
                return next(self.iterator, None)

        extra_options = {}
        if tensor_quant_overrides is not None:
            extra_options["TensorQuantOverrides"] = tensor_quant_overrides

        quantization.quantize_static(
            model_input="model.onnx",
            model_output=output_model_name,
            calibration_data_reader=DummyDataReader(self.activations),
            quant_format=quantization.QuantFormat.QDQ,
            activation_type=self.default_act_qtype,
            weight_type=self.default_wgt_qtype,
            op_types_to_quantize=["Conv", "Sigmoid"],
            extra_options=extra_options,
        )

        # Extract quantization parameters: scales and zero points for activations and weights.
        model = onnx.load(output_model_name)
        inp_zp = next(init for init in model.graph.initializer if init.name == "INP_zero_point")
        inp_sc = next(init for init in model.graph.initializer if init.name == "INP_scale")
        sig_out_zp = next(init for init in model.graph.initializer if init.name == "SIG_OUT_zero_point")
        sig_out_sc = next(init for init in model.graph.initializer if init.name == "SIG_OUT_scale")
        wgt_zp = next(init for init in model.graph.initializer if init.name == "WGT_zero_point")
        wgt_sc = next(init for init in model.graph.initializer if init.name == "WGT_scale")
        out_zp = next(init for init in model.graph.initializer if init.name == "OUT_zero_point")
        out_sc = next(init for init in model.graph.initializer if init.name == "OUT_scale")

        # Return quantization parameters
        return inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, out_zp, out_sc

    def test_qdq_default(self):
        """
        Test default behavior without specifying the TensorQuantOverrides option.
        """
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, out_zp, out_sc = self.perform_qdq_quantization(
            "model_default_quant_overrides.onnx",
            tensor_quant_overrides=None,  # default behavior
        )

        # No overrides set. Expect default values
        self.assertEqual(inp_zp.int32_data[0], self.expected_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(inp_sc.float_data[0], self.expected_zp_scales["INP"][1])

        self.assertEqual(sig_out_zp.int32_data[0], self.expected_zp_scales["SIG_OUT"][0])
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(sig_out_sc.float_data[0], self.expected_zp_scales["SIG_OUT"][1])

        self.assertEqual(wgt_zp.int32_data[0], self.expected_zp_scales["WGT"][0])
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype.tensor_type)
        self.assertEqual(wgt_sc.float_data[0], self.expected_zp_scales["WGT"][1])

        self.assertEqual(out_zp.int32_data[0], self.expected_zp_scales["OUT"][0])
        self.assertEqual(out_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(out_sc.float_data[0], self.expected_zp_scales["OUT"][1])

    def test_qdq_overrides1(self):
        """
        Test overriding scale/zp for Sigmoid output, and quant_type, symmetric, reduce_range for weight.
        """
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides1.onnx",
            tensor_quant_overrides={
                "SIG_OUT": {"scale": 1.0, "zero_point": 127},
                "WGT": {"quant_type": quantization.QuantType.QInt8, "symmetric": True, "reduce_range": True},
            },
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.expected_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(inp_sc.float_data[0], self.expected_zp_scales["INP"][1])

        # Sigmoid output should have overridden scale/zp
        self.assertEqual(sig_out_zp.int32_data[0], 127)
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(1.0))

        # Weight should have different type, zero_point, and scale
        self.assertEqual(wgt_zp.data_type, quantization.QuantType.QInt8.tensor_type)

        wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type, reduce_range=True, symmetric=True)
        wgt_rmin, wgt_rmax = np.min(self.weight), np.max(self.weight)
        new_wgt_zp, new_wgt_sc = compute_scale_zp(wgt_rmin, wgt_rmax, wgt_qmin, wgt_qmax, symmetric=True)
        self.assertEqual(wgt_zp.int32_data[0], new_wgt_zp)
        self.assertEqual(wgt_sc.float_data[0], np.float32(new_wgt_sc))

    def test_qdq_overrides2(self):
        """
        Test overriding rmin/rmax for Sigmoid output.
        """
        sigmoid_rmin, sigmoid_rmax = 0.0, 0.5
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides2.onnx",
            tensor_quant_overrides={"SIG_OUT": {"rmin": 0.0, "rmax": 0.5}},
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.expected_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype.tensor_type)
        self.assertEqual(inp_sc.float_data[0], self.expected_zp_scales["INP"][1])

        # Sigmoid output should have different scale/zp due to overridden rmin/rmax
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype.tensor_type)

        sigmoid_qmin, sigmoid_qmax = get_qmin_qmax_for_qType(sig_out_zp.data_type)
        new_sigmoid_zp, new_sigmoid_sc = compute_scale_zp(sigmoid_rmin, sigmoid_rmax, sigmoid_qmin, sigmoid_qmax)
        self.assertEqual(sig_out_zp.int32_data[0], new_sigmoid_zp)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(new_sigmoid_sc))

    def test_override_validation_nonexisting_tensor(self):
        """
        Test that specifying a non-existing tensor should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"NON_EXISTING": {"rmin": 0.0, "rmax": 0.5}},
            )

        self.assertTrue("is not present in the model" in str(context.exception))

    def test_override_validation_scale_missing_zp(self):
        """
        Test that specifying a scale without zero_point should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": {"scale": 0.0}},
            )

        self.assertTrue("Must provide both 'scale' and 'zero_point'" in str(context.exception))

    def test_override_validation_bad_combination(self):
        """
        Test that specifying a scale/zero_point with rmax should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": {"scale": 0.0, "zero_point": 0, "rmax": 10.0}},
            )

        self.assertTrue("option 'rmax' is invalid with 'scale' and 'zero_point'" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
