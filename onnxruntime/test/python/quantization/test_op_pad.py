#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import itertools
import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import (
    TestDataFeeds,
    check_model_correctness,
    check_op_type_count,
    check_qtype_by_node_type,
    get_tensor_consumers_and_producers,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestOpQuatizerPad(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_pad(
        self,
        output_model_path,
        pad_mode,
        pad_input_shape,
        pad_dims,
        constant_value=None,
    ):
        #    (input)
        #      |
        #     Pad
        #      |
        #    (output)
        rank = len(pad_input_shape)
        self.assertEqual(rank * 2, len(pad_dims))

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, pad_input_shape)
        pad_dims_initializer = helper.make_tensor("pad_dims", TensorProto.INT64, [2 * rank], pad_dims)
        output_shape = [sum(e) for e in list(zip(pad_input_shape, pad_dims[:rank], pad_dims[rank:]))]
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        inputs = ["input", "pad_dims"]
        initializers = [pad_dims_initializer]
        if (constant_value is not None) and (pad_mode is None or pad_mode == "constant"):
            constant_value_tensor = helper.make_tensor("padding_value", TensorProto.FLOAT, [], [constant_value])
            inputs.extend(["padding_value"])
            initializers.extend([constant_value_tensor])
        kwargs = {"mode": pad_mode} if pad_mode is not None else {}
        pad_node = helper.make_node("Pad", inputs, ["output"], name="PadNode", **kwargs)

        graph = helper.make_graph(
            [pad_node],
            "TestOpQuantizerPad_test_model",
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def construct_model_conv_pad(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        pad_input_shape,
        pad_mode,
        pad_dims,
        constant_value=None,
    ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity   Pad
        #    /            \
        # (identity_out)  (output)
        rank = len(pad_input_shape)
        self.assertEqual(rank * 2, len(pad_dims))

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, pad_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        pad_dims_initializer = helper.make_tensor("pad_dims", TensorProto.INT64, [2 * rank], pad_dims)
        output_shape = [sum(e) for e in list(zip(pad_input_shape, pad_dims[:rank], pad_dims[rank:]))]
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        pad_inputs = ["conv_output", "pad_dims"]
        initializers = [conv_weight_initializer, pad_dims_initializer]
        if (constant_value is not None) and (pad_mode is None or pad_mode == "constant"):
            constant_value_tensor = helper.make_tensor("padding_value", TensorProto.FLOAT, [], [constant_value])
            pad_inputs.extend(["padding_value"])
            initializers.extend([constant_value_tensor])
        kwargs = {"mode": pad_mode} if pad_mode is not None else {}
        pad_node = helper.make_node("Pad", pad_inputs, ["output"], name="pad_node", **kwargs)

        graph = helper.make_graph(
            [conv_node, identity_node, pad_node],
            "TestOpQuantizerPad_test_model",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_model(
        self,
        model_fp32_path,
        model_i8_path,
        data_reader=None,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        extra_options={},  # noqa: B006
    ):
        if data_reader is not None:
            quantize_static(
                model_fp32_path,
                model_i8_path,
                data_reader,
                reduce_range=True,
                quant_format=QuantFormat.QOperator,
                activation_type=activation_type,
                weight_type=weight_type,
                extra_options=extra_options,
            )
        else:
            quantize_dynamic(
                model_fp32_path,
                model_i8_path,
                reduce_range=True,
                weight_type=weight_type,
                extra_options=extra_options,
            )

    def verify_should_not_trigger(self, quantize_mode="static"):
        np.random.seed(108)
        model_fp32_path = f"qop_pad_notrigger_fp32_{quantize_mode}.onnx"
        model_i8_path = f"qop_pad_notrigger_i8_{quantize_mode}.onnx"
        data_reader = self.input_feeds(1, {"input": [1, 16, 31, 31]})
        self.construct_model_pad(model_fp32_path, "constant", [1, 16, 31, 31], [0, 0, 1, 2, 0, 0, 3, 4])
        self.quantize_model(
            model_fp32_path,
            model_i8_path,
            None if quantize_mode != "static" else data_reader,
        )
        data_reader.rewind()
        # DequantizeLinear=0 pad node is not been quantized as input is not quantized.
        check_op_type_count(
            self,
            model_i8_path,
            DynamicQuantizeLinear=0,
            QuantizeLinear=0,
            DequantizeLinear=0,
        )
        check_model_correctness(self, model_fp32_path, model_i8_path, data_reader.get_next())

    def test_static_quantize_no_trigger(self):
        self.verify_should_not_trigger(quantize_mode="static")

    def test_dynamic_quantize_no_trigger(self):
        self.verify_should_not_trigger(quantize_mode="dynamic")

    def verify_quantize_with_pad_mode(
        self,
        pad_mode,
        constant_value=None,
        quantize_mode="static",
        rtol=0.01,
        atol=0.05,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        extra_options=None,
    ):
        if extra_options is None:
            extra_options = {}
        np.random.seed(108)
        tag_pad_mode = pad_mode if pad_mode is not None else "none"
        tag_constant_value = "" if constant_value is None else "_value"
        model_fp32_path = f"qop_pad_{quantize_mode}_fp32_{tag_pad_mode}{tag_constant_value}.onnx"
        edge_case = "dual_feed" in extra_options and extra_options["dual_feed"] and constant_value is not None
        if edge_case:
            data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33], "padding_value": [1]})
            self.construct_edge_case_model(
                model_fp32_path,
                [1, 8, 33, 33],
                [16, 8, 3, 3],
                [1, 16, 31, 31],
                pad_mode,
                [0, 0, 1, 2, 0, 0, 3, 4],
                constant_value=constant_value,
            )
        else:
            data_reader = self.input_feeds(1, {"input": [1, 8, 33, 33]})
            self.construct_model_conv_pad(
                model_fp32_path,
                [1, 8, 33, 33],
                [16, 8, 3, 3],
                [1, 16, 31, 31],
                pad_mode,
                [0, 0, 1, 2, 0, 0, 3, 4],
                constant_value=constant_value,
            )

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_i8_path = (
            f"qop_pad_{quantize_mode}_i8_{tag_pad_mode}{tag_constant_value}_{activation_type_str}{weight_type_str}.onnx"
        )
        data_reader.rewind()
        self.quantize_model(
            model_fp32_path,
            model_i8_path,
            None if quantize_mode != "static" else data_reader,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        # DequantizeLinear=2 means there are one DequantizeLinear Node aftr both conv and pad,
        # which means pad node is running in quantized semantic.
        # In dynamic quantize mode, pad operator in fact not quantized as input is fp32.
        if quantize_mode != "static":
            kwargs = {"DynamicQuantizeLinear": 1} if activation_type == QuantType.QUInt8 else {"QuantizeLinear": 1}
        else:
            # edge case will have 2 graph inputs
            kwargs = {"DequantizeLinear": 2, "QuantizeLinear": 2 if edge_case else 1}
        check_op_type_count(self, model_i8_path, **kwargs)
        # check node input/output type if such node exists in the graph
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        qnode_io_qtypes.update({"ConvInteger": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_i8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self,
            model_fp32_path,
            model_i8_path,
            data_reader.get_next(),
            rtol=rtol,
            atol=atol,
        )

    def test_static_mode_edge(self):
        self.verify_quantize_with_pad_mode("edge", constant_value=None)

    def test_static_mode_reflect(self):
        self.verify_quantize_with_pad_mode("reflect", constant_value=None)

    def test_static_mode_constant_default(self):
        self.verify_quantize_with_pad_mode("constant", constant_value=None)

    def test_static_mode_constant_value(self):
        self.verify_quantize_with_pad_mode("constant", constant_value=3.75)

    def test_static_mode_edge_s8s8(self):
        self.verify_quantize_with_pad_mode(
            "edge",
            constant_value=None,
            rtol=0.1,
            atol=0.1,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_static_mode_reflect_s8s8(self):
        self.verify_quantize_with_pad_mode(
            "reflect",
            constant_value=None,
            rtol=0.1,
            atol=0.1,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_static_mode_constant_default_s8s8(self):
        self.verify_quantize_with_pad_mode(
            "constant",
            constant_value=None,
            rtol=0.1,
            atol=0.1,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_static_mode_constant_value_s8s8(self):
        self.verify_quantize_with_pad_mode(
            "constant",
            constant_value=3.75,
            rtol=0.1,
            atol=0.1,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_dynamic_mode_edge(self):
        self.verify_quantize_with_pad_mode("edge", constant_value=None, quantize_mode="dynamic")

    def test_dynamic_mode_reflect(self):
        self.verify_quantize_with_pad_mode("reflect", constant_value=None, quantize_mode="dynamic")

    def test_dynamic_mode_constant_default(self):
        self.verify_quantize_with_pad_mode("constant", constant_value=None, quantize_mode="dynamic")

    def test_dynamic_mode_constant_value(self):
        self.verify_quantize_with_pad_mode("constant", constant_value=3.75, quantize_mode="dynamic")

    # TODO: uncomment following after ConvInteger s8 supported
    # def test_dynamic_mode_edge_s8s8(self):
    #     self.verify_quantize_with_pad_mode('edge', constant_value=None, quantize_mode='dynamic', activation_type=QuantType.QInt8,
    #                                        weight_type=QuantType.QInt8, extra_options={'ActivationSymmetric': True})

    # def test_dynamic_mode_reflect_s8s8(self):
    #     self.verify_quantize_with_pad_mode('reflect', constant_value=None, quantize_mode='dynamic', activation_type=QuantType.QInt8,
    #                                        weight_type=QuantType.QInt8, extra_options={'ActivationSymmetric': True})

    # def test_dynamic_mode_constant_default_s8s8(self):
    #     self.verify_quantize_with_pad_mode('constant', constant_value=None, quantize_mode='dynamic', activation_type=QuantType.QInt8,
    #                                        weight_type=QuantType.QInt8, extra_options={'ActivationSymmetric': True})

    # def test_dynamic_mode_constant_value_s8s8(self):
    #     self.verify_quantize_with_pad_mode('constant', constant_value=3.75, quantize_mode='dynamic', activation_type=QuantType.QInt8,
    #                                        weight_type=QuantType.QInt8, extra_options={'ActivationSymmetric': True})
    def construct_edge_case_model(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        pad_input_shape,
        pad_mode,
        pad_dims,
        constant_value=None,
    ):
        #      (input)
        #          \
        #         Conv   (padding_value)
        #        /    \   /
        #   Identity   Pad
        #    /            \
        # (identity_out)  (output)
        rank = len(pad_input_shape)
        self.assertEqual(rank * 2, len(pad_dims))

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)
        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, pad_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        pad_dims_initializer = helper.make_tensor("pad_dims", TensorProto.INT64, [2 * rank], pad_dims)
        output_shape = [sum(e) for e in list(zip(pad_input_shape, pad_dims[:rank], pad_dims[rank:]))]
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        pad_inputs = ["conv_output", "pad_dims"]
        initializers = [conv_weight_initializer, pad_dims_initializer]
        constant_value_tensor = helper.make_tensor_value_info("padding_value", TensorProto.FLOAT, [1])
        pad_inputs.extend(["padding_value"])
        kwargs = {"mode": pad_mode} if pad_mode is not None else {}
        pad_node = helper.make_node("Pad", pad_inputs, ["output"], name="pad_node", **kwargs)

        graph = helper.make_graph(
            [conv_node, identity_node, pad_node],
            "TestOpQuantizerPad_test_model",
            [input_tensor, constant_value_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def test_static_mode_constant_value_edge_case(self):
        self.verify_quantize_with_pad_mode(
            "constant", constant_value=0.1, quantize_mode="static", extra_options={"dual_feed": True}
        )

    @classmethod
    def construct_model_add_pad_add(
        cls,
        # Name of model input, i.e., "input" in the illustration graph below.
        name,
        # Name of model output.
        final_name,
        # model input shape.
        shape,
    ):
        # Graph implemented below is
        #  `name`, `name` -> Add -> "first_add_output"
        #  "first_add_output", "pads" -> Pad -> "pad_output"
        #  "pad_output", "pad_output" -> Add -> `final_name`
        # where `name` is the 2nd argument of this function,
        # `final_name` is the 3rd argument of this function,
        # and the rest lowercase strings are tensor names in the graph.

        input_name = name
        first_add_output_name = "first_add_output"
        pads_name = "pads"
        pad_output_name = "pad_output"
        second_add_output_name = final_name

        input_shape = shape
        input_rank = len(input_shape)

        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)

        first_add_node = helper.make_node(
            "Add",
            [input_name, input_name],
            [first_add_output_name],
            name="FirstAdd",
        )

        pads = [1, 2] * input_rank
        pads_initializer = helper.make_tensor(
            pads_name,
            TensorProto.INT64,
            # 1-D tensor of shape [2 * input_rank].
            [len(pads)],
            pads,
        )
        pad_node = helper.make_node(
            "Pad",
            [first_add_output_name, pads_name, ""],
            [pad_output_name],
            name="PadNode",
            mode="constant",
        )
        pad_output_shape = tuple(input_shape[i] + pads[i] + pads[i + input_rank] for i in range(input_rank))

        second_add_node = helper.make_node(
            "Add",
            [pad_output_name, pad_output_name],
            [second_add_output_name],
            name="SecondAdd",
        )

        output_tensor = helper.make_tensor_value_info(second_add_output_name, TensorProto.FLOAT, pad_output_shape)

        graph = helper.make_graph(
            [first_add_node, pad_node, second_add_node],
            "TestPadWithEmptyStringInput",
            [input_tensor],
            [output_tensor],
            initializer=[pads_initializer],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        return model

    def test_pad_with_empty_string_input_name(self):
        np.random.seed(108)
        model_fp32_path = "pad_with_empty_string_input_name_fp32.onnx"
        model_i8_path = "pad_with_empty_string_input_name_i8.onnx"

        shape = [
            3,
        ]
        name = "input"
        data_reader = self.input_feeds(
            1,
            {
                name: shape,
            },
        )

        model_fp32 = TestOpQuatizerPad.construct_model_add_pad_add(name=name, shape=shape, final_name="output")
        op_types = [n.op_type for n in model_fp32.graph.node]
        self.assertEqual(["Add", "Pad", "Add"], op_types)

        onnx.save(model_fp32, model_fp32_path)

        self.quantize_model(
            model_fp32_path,
            model_i8_path,
            data_reader=data_reader,
        )

        model_i8 = onnx.load(model_i8_path)
        print(model_i8)

        # Assert quantization really happens.
        op_types = [n.op_type for n in model_i8.graph.node]
        self.assertEqual(["QuantizeLinear", "QLinearAdd", "Pad", "QLinearAdd", "DequantizeLinear"], op_types)

        for node in model_i8.graph.node:
            # Examine no empty string flows to quantization process.
            # Previously, optional input specified by `""` in NodeProto.input
            # may cause phantom node to generate `"_quantized"` in quantization process.
            for name in itertools.chain(node.input, node.output):
                self.assertNotEqual(name, "")
                self.assertNotEqual(name, "_quantized")


class TestQDQPad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.qdq.pad_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_pad_model(
        self,
        mode: str,
        constant_value: float | None = None,
        opset: int = 21,
        float_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
    ) -> onnx.ModelProto:
        num_pads_start = 1
        input_0 = onnx.helper.make_tensor_value_info("input_0", float_type, (3, 2))
        output_0 = onnx.helper.make_tensor_value_info("output_0", float_type, (3, 2 + num_pads_start))

        initializers = []
        pad_input_names = ["input_0"]
        attrs = {"mode": mode}

        pads_data = np.array([0, num_pads_start, 0, 0], dtype=np.int64)  # Pad one val at beginning of axis 1.
        if opset >= 11:
            initializers.append(onnx.numpy_helper.from_array(pads_data, "pads"))
            pad_input_names.append("pads")
        else:
            attrs["pads"] = pads_data.tolist()

        if mode == "constant" and constant_value is not None:
            if opset >= 11:
                initializers.append(onnx.helper.make_tensor("constant_value", float_type, [], [constant_value]))
                pad_input_names.append("constant_value")
            else:
                attrs["value"] = float(constant_value)

        pad_node = onnx.helper.make_node("Pad", pad_input_names, ["output_0"], name="Pad0", **attrs)

        graph = onnx.helper.make_graph(
            [pad_node],
            "PadFloat",
            [input_0],
            [output_0],
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", opset)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_qdq_pad_qparams(self):
        """
        Test that QDQ Pad has equal scale/zero-point for its input and output for certain configurations.
        """
        test_configs = [
            # Opset 21
            ("constant", None, 21, onnx.TensorProto.FLOAT),
            ("constant", None, 21, onnx.TensorProto.FLOAT16),
            ("constant", 0, 21, onnx.TensorProto.FLOAT),
            ("constant", 0, 21, onnx.TensorProto.FLOAT16),
            ("constant", 10.0, 21, onnx.TensorProto.FLOAT),
            ("constant", 10.0, 21, onnx.TensorProto.FLOAT16),
            ("reflect", None, 21, onnx.TensorProto.FLOAT),
            ("reflect", None, 21, onnx.TensorProto.FLOAT16),
            ("edge", None, 21, onnx.TensorProto.FLOAT),
            ("edge", None, 21, onnx.TensorProto.FLOAT16),
            ("wrap", None, 21, onnx.TensorProto.FLOAT),
            ("wrap", None, 21, onnx.TensorProto.FLOAT16),
            # Model with opset 10 will use pad of opset 2, which uses attributes instead of inputs.
            # Opset 10 Q/DQ ops don't support float16.
            ("constant", None, 10, onnx.TensorProto.FLOAT),
            ("constant", 0, 10, onnx.TensorProto.FLOAT),
            ("constant", 10.0, 10, onnx.TensorProto.FLOAT),
            ("reflect", None, 10, onnx.TensorProto.FLOAT),
            ("edge", None, 10, onnx.TensorProto.FLOAT),
        ]

        for pad_mode, constant_value, opset, float_type in test_configs:
            with self.subTest(pad_mode=pad_mode, constant_value=constant_value, opset=opset, float_type=float_type):
                label = f"_{pad_mode}_{constant_value}_opset{opset}_{onnx.TensorProto.DataType.Name(float_type)}"
                float_model_path = os.path.join(self._tmp_dir_path, f"pad{label}.float.onnx")
                qdq_model_path = os.path.join(self._tmp_dir_path, f"pad{label}.qdq.onnx")

                float_model = self.build_pad_model(pad_mode, constant_value, opset=opset, float_type=float_type)
                onnx.save_model(float_model, float_model_path)

                # Create a data reader
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(float_type)
                input_data_list = [
                    {"input_0": np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], dtype=np_dtype)},
                    {"input_0": np.array([[2.3, 3.4], [4.5, 5.7], [1.0, 1.2]], dtype=np_dtype)},
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
                )

                expected_op_counts = {"DequantizeLinear": 2, "QuantizeLinear": 2, "Pad": 1}
                if constant_value is not None and opset >= 11:
                    expected_op_counts["DequantizeLinear"] += 1  # The constant padding value is quantized.
                check_op_type_count(self, qdq_model_path, **expected_op_counts)

                if pad_mode != "reflect":
                    # Do not check model correctness for 'reflect' mode because ONNX Runtime implementation does
                    # not match the ONNX reference implementation. See the following issue:
                    # https://github.com/microsoft/onnxruntime/issues/20801
                    data_reader.rewind()
                    check_model_correctness(self, float_model_path, qdq_model_path, data_reader.get_next())

                qdq_model = onnx.load_model(qdq_model_path)
                quant_output_same_as_input = False

                if pad_mode in ("reflect", "edge", "wrap"):
                    quant_output_same_as_input = True

                if pad_mode == "constant" and constant_value in (None, 0):
                    quant_output_same_as_input = True

                pad_node = next((node for node in qdq_model.graph.node if node.op_type == "Pad"), None)
                self.assertNotEqual(pad_node, None)
                self.assertEqual(pad_node.op_type, "Pad")

                # Get the parent and child nodes of the Pad and check that they are DQ/Q.
                consumers, producers = get_tensor_consumers_and_producers(qdq_model)
                input_dq_node = producers.get(pad_node.input[0], None)
                self.assertNotEqual(input_dq_node, None)
                self.assertEqual(input_dq_node.op_type, "DequantizeLinear")

                output_q_node = consumers.get(pad_node.output[0], [None])[0]
                self.assertNotEqual(output_q_node, None)
                self.assertEqual(output_q_node.op_type, "QuantizeLinear")

                # Check that the Pad's input DQ uses the same scale/zp as the Pad's output Q.
                if quant_output_same_as_input:
                    self.assertEqual(input_dq_node.input[1], output_q_node.input[1])  # Same scale
                    self.assertEqual(input_dq_node.input[2], output_q_node.input[2])  # Same zero-point
                else:
                    self.assertNotEqual(input_dq_node.input[1], output_q_node.input[1])
                    self.assertNotEqual(input_dq_node.input[2], output_q_node.input[2])


if __name__ == "__main__":
    unittest.main()
