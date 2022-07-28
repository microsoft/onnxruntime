# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
import os
from pickletools import uint8
import struct
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer
from .quant_utils import (
    QuantizationMode,
    QuantizedInitializer,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    __producer__,
    __version__,
    attribute_to_kwarg,
    find_by_name,
    generate_identified_filename,
    get_elem_index,
    get_mul_node,
    get_qmin_qmax_for_qType,
    get_qrange_for_qType,
    onnx_domain,
    quantize_nparray,
    type_to_name,
)
from .registry import CreateQDQQuantizer


class QDQQuantizer(ONNXQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        input_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options={},
    ):
        ONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            input_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = []
        self.tensors_to_quantize_per_channel = []
        self.bias_to_quantize = []
        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = (
            []
            if "OpTypesToExcludeOutputQuantizatioin" not in extra_options
            else extra_options["OpTypesToExcludeOutputQuantizatioin"]
        )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (
            False if "DedicatedQDQPair" not in extra_options else extra_options["DedicatedQDQPair"]
        )
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {}
            if "QDQOpTypePerChannelSupportToAxis" not in extra_options
            else extra_options["QDQOpTypePerChannelSupportToAxis"]
        )

        # Name of initializer with minimum quantization value for qint8 or quint8
        self.fixed_qmin_name = "fixed_qmin"

    def quantize_tensor(self, tensor_name):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_name)
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                self.tensors_to_quantize.append(tensor_name)
        else:
            logging.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name
                )
            )

    def quantize_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize_per_channel.append((tensor_name, axis))
        else:
            logging.warning(
                "only support per-channel quantization on weight. Quantize tensor: {} with per-tensor instead.".format(
                    tensor_name
                )
            )
            self.quantize_tensor(tensor_name)

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                if op_quantizer == None: continue # Skip quantize if no quantizer returned
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)
        self.quantize_tensors()
        self.quantize_weights_per_channel()
        self.quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            ONNXQuantizer.CleanGraphInitializers(self.model.graph(), self.model.model)

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params.keys()
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            self.tensors_to_quantize.remove(upstream_output_name)
            return True
        return False

    def create_dynamic_subgraph(self, input_name, nodes_list, qType, symmetric=False):
        """
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        if symmetric:
            return self.create_dynamic_subgraph_symmetric(input_name, nodes_list, qType)

        return self.create_dynamic_subgraph_asymmetric(input_name, nodes_list,qType)

    def create_dynamic_subgraph_symmetric(self, input_name, nodes_list, qType):
        """
        Create nodes for dynamic symmetric quantization of input and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """

        qrange_name = self.fixed_qrange_int8_name if qType == onnx_proto.TensorProto.INT8 else self.fixed_qrange_uint8_name
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(
            qrange_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [get_qrange_for_qType(qType, reduce_range=self.reduce_range, symmetric=True) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], qrange_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)

        # Zero point
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=self.reduce_range, symmetric=True)
        zp = int((qmin + qmax) / 2)
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [zp])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def create_dynamic_subgraph_asymmetric(self, input_name, nodes_list, qType):
        """
        Create nodes for asymmetric dynamic quantization of input and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"
        qrange_name = self.fixed_qrange_int8_name if qType == onnx_proto.TensorProto.INT8 else self.fixed_qrange_uint8_name

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(
            qrange_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [get_qrange_for_qType(qType, reduce_range=self.reduce_range, symmetric=False)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx_proto.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=self.reduce_range, symmetric=False)
        initializer_qmin = onnx.helper.make_tensor(self.fixed_qmin_name, onnx_proto.TensorProto.FLOAT, [], [qmin])
        self.model.add_initializer(initializer_qmin)

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        nodes_list.append(reduce_min_node)
        
        zero_min_name = input_name + "_Min"
        zero_min_node = onnx.helper.make_node(
            "Min",
            [reduce_min_name + ":0", self.fixed_zero_name],
            [zero_min_name+":0"],
            zero_min_name
        )
        nodes_list.append(zero_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        nodes_list.append(reduce_max_node)

        zero_max_name = input_name + "_Max"
        zero_max_node = onnx.helper.make_node(
            "Max",
            [reduce_max_name + ":0", self.fixed_zero_name],
            [zero_max_name+":0"],
            zero_max_name
        )
        nodes_list.append(zero_max_node)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [zero_max_node.output[0], zero_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], qrange_name],
            [input_scale_name],
            scale_div_name,
        )
        nodes_list.append(scale_div_node)
        
        # Divide rmin by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zero_min_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        nodes_list.append(zp_div_node)
        # Compute zero point
        #   Subtract zero and rmin/scale
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_qmin_name, zp_div_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        nodes_list.append(zp_sub_node)
        
        # Compute round
        zp_round_name = input_name + "_zero_point_Round"
        zp_round_node = onnx.helper.make_node("Round", zp_sub_node.output, [zp_round_name + ":0"], zp_round_name)
        nodes_list.append(zp_round_node)
        # Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_round_node.output, [input_zp_name], zp_cast_name, to=qType) # TODO recast zp as int32 to avoid underflow...
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def quantize_tensors(self):
        for tensor_name in self.tensors_to_quantize:
            if tensor_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:

                if self.add_qdq_pair_to_weight:
                    q_weight_name, zp_name, scale_name = self.quantize_weight(
                        initializer, self.weight_qType, keep_float_weight=True
                    )
                    qlinear_node = onnx.helper.make_node(
                        "QuantizeLinear",
                        [tensor_name, scale_name, zp_name],
                        [tensor_name + "_QuantizeLinear"],
                        tensor_name + "_QuantizeLinear",
                    )
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        [tensor_name + "_QuantizeLinear", scale_name, zp_name],
                        [tensor_name + "_DequantizeLinear"],
                        tensor_name + "_DequantizeLinear",
                    )
                    self.model.replace_input_of_all_nodes(tensor_name, tensor_name + "_DequantizeLinear")

                    self.model.add_nodes([qlinear_node, dequant_node])
                else:
                    q_weight_name, zp_name, scale_name = self.quantize_weight(initializer, self.weight_qType)
                    inputs = [q_weight_name, scale_name, zp_name]
                    output_name = tensor_name + "_DequantizeLinear"
                    node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [output_name],
                        tensor_name + "_DequantizeLinear",
                    )
                    self.model.add_node(node)
                    self.model.replace_input_of_all_nodes(tensor_name, tensor_name + "_DequantizeLinear")
            else:
                data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)
                nodes = []

                if data_found == False:
                    if self.static:
                        raise ValueError(
                            "Quantization parameters are not specified for param {}."
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required.".format(
                                tensor_name
                            )
                        )
                    # Here we add dynamic subgraph, if we found no static params
                    # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
                    qType = self.input_qType
                    if self.model.is_graph_output(tensor_name): # Changes name to quantize output correctly
                        (
                            scale_name,
                            zp_name,
                            scale_shape,
                            zp_shape,
                        ) = self.create_dynamic_subgraph(tensor_name + "_QuantizeLinearInput", nodes, qType, symmetric=self.is_activation_symmetric)
                    else:
                        (
                            scale_name,
                            zp_name,
                            scale_shape,
                            zp_shape,
                        ) = self.create_dynamic_subgraph(tensor_name, nodes, qType, symmetric=self.is_activation_symmetric)
                if (
                    self.dedicated_qdq_pair
                    and tensor_name in self.tensor_to_its_receiving_nodes
                    and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
                ):
                    # TODO: This if block should be tested
                    num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
                    for i in range(num_dedicated_qdq_pair):
                        postfix = str(i + 1)
                        q_input = tensor_name
                        q_output = tensor_name + "_QuantizeLinear_" + postfix
                        dq_input = q_output
                        dq_output = tensor_name + "_DequantizeLinear_" + postfix
                        quant_node_name = tensor_name + "_QuantizeLinear_" + postfix
                        dequant_node_name = tensor_name + "_DequantizeLinear_" + postfix
                        qlinear_node = onnx.helper.make_node(
                            "QuantizeLinear",
                            [q_input, scale_name, zp_name],
                            [q_output],
                            quant_node_name,
                        )
                        dequant_node = onnx.helper.make_node(
                            "DequantizeLinear",
                            [dq_input, scale_name, zp_name],
                            [dq_output],
                            dequant_node_name,
                        )
                        self.model.add_nodes([qlinear_node, dequant_node] + nodes)

                        node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                        self.model.replace_node_input(node, tensor_name, dq_output)

                    quantized_value = QuantizedValue(
                        tensor_name,
                        dq_output,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
                else:
                    q_input = tensor_name
                    q_output = tensor_name + "_QuantizeLinear"
                    dq_input = q_output
                    dq_output = tensor_name + "_DequantizeLinear"
                    if self.model.is_graph_output(tensor_name):
                        q_input = tensor_name + "_QuantizeLinearInput"
                        dq_output = tensor_name
                        self.model.replace_output_of_all_nodes(tensor_name, q_input)
                    else:
                        self.model.replace_input_of_all_nodes(tensor_name, dq_output)

                    quant_node_name = tensor_name + "_QuantizeLinear"
                    dequant_node_name = tensor_name + "_DequantizeLinear"
                    qlinear_node = onnx.helper.make_node(
                        "QuantizeLinear",
                        [q_input, scale_name, zp_name],
                        [q_output],
                        quant_node_name,
                    )
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        [dq_input, scale_name, zp_name],
                        [dq_output],
                        dequant_node_name,
                    )
                    self.model.add_nodes(nodes + [qlinear_node, dequant_node])

                    quantized_value = QuantizedValue(
                        tensor_name,
                        dq_output,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value

    def quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            # TODO: check if we have an input_scale initializer and decide whether to quantize bias static based off of that
            # get scale for input
            if input_name in self.quantized_value_map:
                input_scale_name = self.quantized_value_map[input_name].scale_name
            elif input_name in self.quantization_params:
                _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
            inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
            if inputscale_initializer  is None:
                # self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
                continue
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    bias_name + "_DequantizeLinear",
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    bias_name + "_DequantizeLinear",
                )
            self.model.add_node(dequant_node)

    def quantize_weights_per_channel(self):
        if self.opset_version < 13 and len(self.tensors_to_quantize_per_channel) > 0:
            raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
        for weight_name, axis in self.tensors_to_quantize_per_channel:
            if self.add_qdq_pair_to_weight:
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(
                    weight_name,
                    onnx_proto.TensorProto.INT8,
                    axis,
                    keep_float_weight=True,
                )
                qlinear_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    [weight_name, scale_name, zp_name],
                    [weight_name + "_QuantizeLinear"],
                    weight_name + "_QuantizeLinear",
                    axis=axis,
                )
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    [weight_name + "_QuantizeLinear", scale_name, zp_name],
                    [weight_name + "_DequantizeLinear"],
                    weight_name + "_DequantizeLinear",
                    axis=axis,
                )
                self.model.replace_input_of_all_nodes(weight_name, weight_name + "_DequantizeLinear")

                self.model.add_nodes([qlinear_node, dequant_node])
            else:
                # q_name, zp_name, scale_name = self.quantize_weight_per_channel(weight_name, self.weight_qType, axis)
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(
                    weight_name, onnx_proto.TensorProto.INT8, axis
                )

                inputs = [q_name, scale_name, zp_name]
                output_name = weight_name + "_DequantizeLinear"
                node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [output_name],
                    weight_name + "_DequantizeLinear",
                    axis=axis,
                )
                self.model.add_node(node)

                # Replace weight_name with output of DequantizeLinear
                self.model.replace_input_of_all_nodes(weight_name, output_name)
