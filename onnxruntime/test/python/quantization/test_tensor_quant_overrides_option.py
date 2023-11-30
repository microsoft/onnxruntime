#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import struct
import unittest

import numpy as np
import onnx

from onnxruntime import quantization
from onnxruntime.quantization.quant_utils import compute_scale_zp, get_qmin_qmax_for_qType


class TestTensorQuantOverridesOption(unittest.TestCase):
    def setUp(self):
        self.activations = [
            np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype="float32"),
        ]

        self.weight = np.array([[[-1.0, -2.0], [1.0, 2.0]], [[-0.5, -1.5], [0.5, 1.5]]], dtype=np.float32)
        self.bias = np.array([0.0, 1.0], dtype=np.float32)
        self.default_act_qtype = onnx.TensorProto.UINT8
        self.default_wgt_qtype = onnx.TensorProto.UINT8
        self.default_wgt_qtype_per_channel = onnx.TensorProto.INT8
        self.default_bias_qtype = onnx.TensorProto.INT32

        self.default_zp_scales = {
            "INP": (0, np.float32(0.0235294122248888)),
            "SIG_OUT": (0, np.float32(0.003911871928721666)),
            "WGT": (128, np.float32(0.01568627543747425)),
            "BIAS": (0, np.float32(0.0000613626980339177)),  # zp == 0, scale = weight_scale * sig_out_scale
            "OUT": (0, np.float32(0.005075461231172085)),
        }
        self.default_zp_scales_per_channel = {
            "INP": (0, np.float32(0.0235294122248888)),
            "SIG_OUT": (0, np.float32(0.003911871928721666)),
            "WGT": ([0, 0], [np.float32(0.015748031437397003), np.float32(0.011811023578047752)]),
            "BIAS": ([0, 0], [np.float32(0.00006160428165458143), np.float32(0.00004620321124093607)]),
            "OUT": (0, np.float32(0.005075461231172085)),
        }

    def perform_qdq_quantization(self, output_model_name, tensor_quant_overrides=None, per_channel=False):
        #    (input)
        #       |
        #    Sigmoid
        #       |
        #     Conv
        #       |
        #    (output)

        inp = onnx.helper.make_tensor_value_info("INP", onnx.TensorProto.FLOAT, self.activations[0].shape)
        sigmoid_node = onnx.helper.make_node("Sigmoid", ["INP"], ["SIG_OUT"])

        out = onnx.helper.make_tensor_value_info("OUT", onnx.TensorProto.FLOAT, [None, None, None])
        wgt_init = onnx.numpy_helper.from_array(self.weight, "WGT")
        bias_init = onnx.numpy_helper.from_array(self.bias, "BIAS")
        conv_node = onnx.helper.make_node("Conv", ["SIG_OUT", "WGT", "BIAS"], ["OUT"])

        graph = onnx.helper.make_graph(
            [sigmoid_node, conv_node], "test", [inp], [out], initializer=[wgt_init, bias_init]
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
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
            per_channel=per_channel,
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
        bias_zp = next(
            init
            for init in model.graph.initializer
            if init.name == "BIAS_quantized_zero_point" or init.name == "BIAS_zero_point"
        )
        bias_sc = next(
            init for init in model.graph.initializer if init.name == "BIAS_quantized_scale" or init.name == "BIAS_scale"
        )
        out_zp = next(init for init in model.graph.initializer if init.name == "OUT_zero_point")
        out_sc = next(init for init in model.graph.initializer if init.name == "OUT_scale")

        # Return quantization parameters
        return inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, bias_zp, bias_sc, out_zp, out_sc

    def test_qdq_default(self):
        """
        Test default behavior without specifying the TensorQuantOverrides option.
        """
        (
            inp_zp,
            inp_sc,
            sig_out_zp,
            sig_out_sc,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            out_zp,
            out_sc,
        ) = self.perform_qdq_quantization(
            "model_default_quant_overrides.onnx",
            tensor_quant_overrides=None,  # default behavior
        )

        # No overrides set. Expect default values
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        self.assertEqual(sig_out_zp.int32_data[0], self.default_zp_scales["SIG_OUT"][0])
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], self.default_zp_scales["SIG_OUT"][1])

        self.assertEqual(wgt_zp.int32_data[0], self.default_zp_scales["WGT"][0])
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertEqual(wgt_sc.float_data[0], self.default_zp_scales["WGT"][1])

        self.assertEqual(bias_zp.int32_data[0], self.default_zp_scales["BIAS"][0])
        self.assertEqual(bias_zp.data_type, self.default_bias_qtype)
        self.assertEqual(bias_sc.float_data[0], self.default_zp_scales["BIAS"][1])

        self.assertEqual(out_zp.int32_data[0], self.default_zp_scales["OUT"][0])
        self.assertEqual(out_zp.data_type, self.default_act_qtype)
        self.assertEqual(out_sc.float_data[0], self.default_zp_scales["OUT"][1])

    def test_qdq_default_per_channel(self):
        """
        Test default per-channel behavior without specifying the TensorQuantOverrides option.
        """
        (
            inp_zp,
            inp_sc,
            sig_out_zp,
            sig_out_sc,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            out_zp,
            out_sc,
        ) = self.perform_qdq_quantization(
            "model_default_per_channel_quant_overrides.onnx",
            tensor_quant_overrides=None,  # default behavior
            per_channel=True,
        )

        # No overrides set. Expect default values
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        self.assertEqual(sig_out_zp.int32_data[0], self.default_zp_scales["SIG_OUT"][0])
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], self.default_zp_scales["SIG_OUT"][1])

        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(self.default_zp_scales_per_channel["WGT"][0]):
            self.assertEqual(wgt_zp.int32_data[index], zp)
        for index, scale in enumerate(self.default_zp_scales_per_channel["WGT"][1]):
            self.assertEqual(wgt_sc.float_data[index], scale)

        self.assertEqual(bias_zp.data_type, self.default_bias_qtype)

        num_bias_zps = len(self.default_zp_scales_per_channel["BIAS"][0])
        actual_bias_zps = struct.unpack(f"<{num_bias_zps}i", bias_zp.raw_data)
        for index, zp in enumerate(self.default_zp_scales_per_channel["BIAS"][0]):
            self.assertEqual(actual_bias_zps[index], zp)

        num_bias_scales = len(self.default_zp_scales_per_channel["BIAS"][1])
        actual_bias_scales = struct.unpack(f"<{num_bias_scales}f", bias_sc.raw_data)
        for index, scale in enumerate(self.default_zp_scales_per_channel["BIAS"][1]):
            self.assertEqual(actual_bias_scales[index], scale)

        self.assertEqual(out_zp.int32_data[0], self.default_zp_scales["OUT"][0])
        self.assertEqual(out_zp.data_type, self.default_act_qtype)
        self.assertEqual(out_sc.float_data[0], self.default_zp_scales["OUT"][1])

    def test_qdq_overrides1(self):
        """
        Test overriding:
          - scale/zp for Sigmoid output
          - quant_type, symmetric, reduce_range for Conv weight
          - quant_type, symmetric, reduce_range for Conv bias
        """
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, bias_zp, bias_sc, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides1.onnx",
            tensor_quant_overrides={
                "SIG_OUT": [{"scale": 1.0, "zero_point": 127}],
                "WGT": [{"quant_type": quantization.QuantType.QInt8, "symmetric": True, "reduce_range": True}],
                "BIAS": [{"quant_type": quantization.QuantType.QInt8, "symmetric": True, "reduce_range": True}],
            },
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        # Sigmoid output should have overridden scale/zp
        self.assertEqual(sig_out_zp.int32_data[0], 127)
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(1.0))

        # Weight should have different type, zero_point, and scale
        self.assertEqual(wgt_zp.data_type, quantization.QuantType.QInt8.tensor_type)

        wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type, reduce_range=True, symmetric=True)
        wgt_rmin, wgt_rmax = np.min(self.weight), np.max(self.weight)
        new_wgt_zp, new_wgt_sc = compute_scale_zp(wgt_rmin, wgt_rmax, wgt_qmin, wgt_qmax, symmetric=True)
        self.assertEqual(wgt_zp.int32_data[0], new_wgt_zp)
        self.assertEqual(wgt_sc.float_data[0], np.float32(new_wgt_sc))

        # Bias should now be treated as a weight and should have different type, zero_point, and scale
        self.assertEqual(bias_zp.data_type, quantization.QuantType.QInt8.tensor_type)

        bias_qmin, bias_qmax = get_qmin_qmax_for_qType(bias_zp.data_type, reduce_range=True, symmetric=True)
        bias_rmin, bias_rmax = np.min(self.bias), np.max(self.bias)
        new_bias_zp, new_bias_sc = compute_scale_zp(bias_rmin, bias_rmax, bias_qmin, bias_qmax, symmetric=True)
        self.assertEqual(bias_zp.int32_data[0], new_bias_zp)
        self.assertEqual(bias_sc.float_data[0], np.float32(new_bias_sc))

    def test_qdq_overrides2(self):
        """
        Test overriding rmin/rmax for Sigmoid output.
        """
        sigmoid_rmin, sigmoid_rmax = 0.0, 0.5
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, _, _, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides2.onnx",
            tensor_quant_overrides={"SIG_OUT": [{"rmin": sigmoid_rmin, "rmax": sigmoid_rmax}]},
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        # Sigmoid output should have different scale/zp due to overridden rmin/rmax
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)

        sigmoid_qmin, sigmoid_qmax = get_qmin_qmax_for_qType(sig_out_zp.data_type)
        new_sigmoid_zp, new_sigmoid_sc = compute_scale_zp(sigmoid_rmin, sigmoid_rmax, sigmoid_qmin, sigmoid_qmax)
        self.assertEqual(sig_out_zp.int32_data[0], new_sigmoid_zp)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(new_sigmoid_sc))

    def test_qdq_overrides3(self):
        """
        Test overriding rmin and rmax for Conv weight
        """
        wgt_rmin, wgt_rmax = 0.0, 1.0
        _, _, _, _, wgt_zp, wgt_sc, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides3.onnx",
            tensor_quant_overrides={
                "WGT": [{"rmin": wgt_rmin, "rmax": wgt_rmax}],
            },
        )

        # Weight should have different zero_point and scale
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertNotEqual(wgt_rmin, np.min(self.weight))
        self.assertNotEqual(wgt_rmax, np.max(self.weight))

        wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type)
        new_wgt_zp, new_wgt_sc = compute_scale_zp(wgt_rmin, wgt_rmax, wgt_qmin, wgt_qmax)
        self.assertEqual(wgt_zp.int32_data[0], new_wgt_zp)
        self.assertEqual(wgt_sc.float_data[0], np.float32(new_wgt_sc))

    def test_qdq_overrides4(self):
        """
        Test overriding scale and zero_point for Conv weight
        """
        wgt_zp_val, wgt_scale_val = 4, 0.5
        _, _, _, _, wgt_zp, wgt_sc, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides4.onnx",
            tensor_quant_overrides={
                "WGT": [{"zero_point": wgt_zp_val, "scale": wgt_scale_val}],
            },
        )

        # Weight should have have the expected zero_point and scale
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertEqual(wgt_zp.int32_data[0], wgt_zp_val)
        self.assertEqual(wgt_sc.float_data[0], np.float32(wgt_scale_val))

    def test_qdq_overrides_per_channel1(self):
        """
        Test per-channel overriding of scale/zero_point for Conv weight and bias.
        """
        zp_vals, scale_vals = [2, 4], [0.5, 0.2]
        (
            _,
            _,
            _,
            _,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            _,
            _,
        ) = self.perform_qdq_quantization(
            "model_per_channel_quant_overrides1.onnx",
            tensor_quant_overrides={
                "WGT": [
                    {"zero_point": zp_vals[0], "scale": scale_vals[0]},
                    {"zero_point": zp_vals[1], "scale": scale_vals[1]},
                ],
                "BIAS": [
                    {"zero_point": zp_vals[0], "scale": scale_vals[0]},
                    {"zero_point": zp_vals[1], "scale": scale_vals[1]},
                ],
            },
            per_channel=True,
        )

        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(zp_vals):
            self.assertEqual(wgt_zp.int32_data[index], zp)
        for index, scale in enumerate(scale_vals):
            self.assertEqual(wgt_sc.float_data[index], np.float32(scale))

        # NOTE: Bias with overrides is treated as a weight.
        self.assertEqual(bias_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(zp_vals):
            self.assertEqual(bias_zp.int32_data[index], zp)
        for index, scale in enumerate(scale_vals):
            self.assertEqual(bias_sc.float_data[index], np.float32(scale))

    def test_qdq_overrides_per_channel2(self):
        """
        Test per-channel overriding of rmin, rmax, reduce_range, and quant_type for Conv weight.
        """
        rmin_vals = [0.0, 0.2]
        rmax_vals = [1.0, 0.8]
        quant_type = quantization.QuantType.QUInt8
        reduce_ranges = [True, False]
        (
            _,
            _,
            _,
            _,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            _,
            _,
        ) = self.perform_qdq_quantization(
            "model_per_channel_quant_overrides2.onnx",
            tensor_quant_overrides={
                "WGT": [
                    {
                        "quant_type": quant_type,
                        "rmin": rmin_vals[0],
                        "rmax": rmax_vals[0],
                        "reduce_range": reduce_ranges[0],
                    },
                    {
                        "quant_type": quant_type,
                        "rmin": rmin_vals[1],
                        "rmax": rmax_vals[1],
                        "reduce_range": reduce_ranges[1],
                    },
                ],
            },
            per_channel=True,
        )

        self.assertEqual(wgt_zp.data_type, quant_type.tensor_type)
        for index, (zp, scale) in enumerate(zip(wgt_zp.int32_data, wgt_sc.float_data)):
            wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type, reduce_range=reduce_ranges[index])
            expected_zp, expected_scale = compute_scale_zp(rmin_vals[index], rmax_vals[index], wgt_qmin, wgt_qmax)
            self.assertEqual(zp, expected_zp)
            self.assertEqual(scale, np.float32(expected_scale))

    def test_override_validation_nonexisting_tensor(self):
        """
        Test that specifying a non-existing tensor should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"NON_EXISTING": [{"rmin": 0.0, "rmax": 0.5}]},
            )

        self.assertIn("is not present in the model", str(context.exception))

    def test_override_validation_scale_missing_zp(self):
        """
        Test that specifying a scale without zero_point should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": [{"scale": 0.0}]},
            )

        self.assertIn("Must provide both 'scale' and 'zero_point'", str(context.exception))

    def test_override_validation_bad_combination(self):
        """
        Test that specifying a scale/zero_point with rmax/rmin/symmetric/reduce_range should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": [{"scale": 0.0, "zero_point": 0, "rmax": 10.0}]},
            )

        self.assertIn("option 'rmax' is invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": [{"scale": 0.0, "zero_point": 0, "rmin": 10.0}]},
            )

        self.assertIn("option 'rmin' is invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": [{"scale": 0.0, "zero_point": 0, "symmetric": True}]},
            )

        self.assertIn("option 'symmetric' is invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                tensor_quant_overrides={"SIG_OUT": [{"scale": 0.0, "zero_point": 0, "reduce_range": True}]},
            )

        self.assertIn("option 'reduce_range' is invalid with 'scale' and 'zero_point'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
