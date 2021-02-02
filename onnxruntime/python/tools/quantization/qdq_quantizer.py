# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct
from pathlib import Path

import numpy as np

from onnx import onnx_pb as onnx_proto
from onnx import TensorProto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue, quantization_modes
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg, type_to_name, quantize_nparray
from .quant_utils import QuantType, onnx_domain, __producer__, __version__

from .registry import CreateOpQuantizer, CreateDefaultOpQuantizer

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer


class QDQQuantizer(ONNXQuantizer):
    def __init__(self, model, per_channel, reduce_range, mode, static, weight_qType, input_qType, quantization_params,
                 nodes_to_quantize, nodes_to_exclude, op_types_to_quantize):
        ONNXQuantizer.__init__(self, model, per_channel, reduce_range, mode, static, weight_qType, input_qType,
                               quantization_params, nodes_to_quantize, nodes_to_exclude, op_types_to_quantize)
        self.tensors_to_quantize = []

    def is_qdq_format(self):
        return True

    def add_tensor_to_quantize(self, tensor_to_quantize):
        tensor_name = tensor_to_quantize.tensor_name
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_to_quantize)
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type is TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_to_quantize)
        else:
            print(
                "Warning: failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".
                format(tensor_name))

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = CreateOpQuantizer(self, node)
                op_quantizer.quantize()

        self.quantize_tensors()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def quantize_tensors(self):
        for tensor_to_quantize in self.tensors_to_quantize:
            tensor_name = tensor_to_quantize.tensor_name
            if tensor_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                weight = self._get_quantized_weight(initializer, self.weight_qType)

                # Update graph
                self._update_weight(weight)

                self.model.remove_initializer(initializer)
                inputs = [weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point"]
                node = onnx.helper.make_node("DequantizeLinear", inputs, [tensor_to_quantize.tensor_name],
                                             tensor_name + '_DequantizeLinear')
                self.model.add_node(node)
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
                                                 QuantizedValueType.Input, None, self.input_qType)
                self.quantized_value_map[tensor_name] = quantized_value
