# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import struct
from pathlib import Path
import numpy as np
import logging

import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto
from onnx import TensorProto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg, type_to_name, quantize_nparray
from .quant_utils import QuantType, onnx_domain, __producer__, __version__

from .registry import CreateQDQQuantizer

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer


class QDQQuantizer(ONNXQuantizer):
    def __init__(self, model, per_channel, reduce_range, mode, static, weight_qType, input_qType, tensors_range,
                 nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, extra_options={}):
        ONNXQuantizer.__init__(self, model, per_channel, reduce_range, mode, static, weight_qType, input_qType,
                               tensors_range, nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, extra_options)
        self.tensors_to_quantize = []
        self.tensors_to_quantize_per_channel = []
        self.bias_to_quantize = []
        self.nodes_to_remove = []

    def quantize_tensor(self, tensor_name):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_name)
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_name)
        else:
            logging.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name))

    def quantize_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize_per_channel.append((tensor_name, axis))
        else:
            logging.warning(
                "only support per-channel quantization on weight. Quantize tensor: {} with per-tensor instead.".format(
                    tensor_name))
            self.quantize_tensor(tensor_name)

    def quantize_bias_tensor(self, bias_name, input_name, weight_name):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append((bias_name, input_name, weight_name))
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

        self.quantize_tensors()
        self.quantize_weights_per_channel()
        self.quantize_bias_tensors()
        self.remove_nodes()
        self.model.remove_unused_constant()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if output_name in self.quantization_params.keys() and \
           len(self.model.input_name_to_nodes()[upstream_output_name]) == 1 and \
           not self.model.is_graph_output(output_name):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            return True
        return False

    def quantize_tensors(self):
        for tensor_name in self.tensors_to_quantize:
            if tensor_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                q_weight_name, zp_name, scale_name = self.quantize_weight(initializer, self.weight_qType)

                inputs = [q_weight_name, scale_name, zp_name]
                output_name = tensor_name + '_DequantizeLinear'
                node = onnx.helper.make_node("DequantizeLinear", inputs, [output_name],
                                             tensor_name + '_DequantizeLinear')
                self.model.add_node(node)
                self.model.replace_input_of_all_nodes(tensor_name, tensor_name + "_DequantizeLinear")
            else:
                data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

                if data_found == False:
                    raise ValueError(
                        "Quantization parameters are not specified for param {}."
                        "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        .format(tensor_name))

                qlinear_node = onnx.helper.make_node("QuantizeLinear", [tensor_name, scale_name, zp_name],
                                                     [tensor_name + "_QuantizeLinear"], tensor_name + "_QuantizeLinear")
                dequant_node = onnx.helper.make_node("DequantizeLinear",
                                                     [tensor_name + "_QuantizeLinear", scale_name, zp_name],
                                                     [tensor_name + "_DequantizeLinear"],
                                                     tensor_name + "_DequantizeLinear")
                self.model.replace_input_of_all_nodes(tensor_name, tensor_name + "_DequantizeLinear")

                self.model.add_nodes([qlinear_node, dequant_node])

                quantized_value = QuantizedValue(tensor_name, tensor_name + "_QuantizeLinear", scale_name, zp_name,
                                                 QuantizedValueType.Input)
                self.quantized_value_map[tensor_name] = quantized_value

    def quantize_bias_tensors(self):
        for bias_name, input_name, weight_name in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node("DequantizeLinear",
                                                     inputs, [bias_name],
                                                     bias_name + '_DequantizeLinear',
                                                     axis=quant_value.axis)
            else:
                dequant_node = onnx.helper.make_node("DequantizeLinear", inputs, [bias_name],
                                                     bias_name + '_DequantizeLinear')
            self.model.add_node(dequant_node)

    def quantize_weights_per_channel(self):
        if self.opset_version < 13 and len(self.tensors_to_quantize_per_channel) > 0:
            raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
        for weight_name, axis in self.tensors_to_quantize_per_channel:
            #q_name, zp_name, scale_name = self.quantize_weight_per_channel(weight_name, self.weight_qType, axis)
            q_name, zp_name, scale_name = self.quantize_weight_per_channel(weight_name, onnx_proto.TensorProto.INT8,
                                                                           axis)

            inputs = [q_name, scale_name, zp_name]
            output_name = weight_name + "_DequantizeLinear"
            node = onnx.helper.make_node("DequantizeLinear",
                                         inputs, [output_name],
                                         weight_name + '_DequantizeLinear',
                                         axis=axis)
            self.model.add_node(node)

            # Replace weight_name with output of DequantizeLinear
            self.model.replace_input_of_all_nodes(weight_name, output_name)
