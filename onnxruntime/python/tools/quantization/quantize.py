# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct

import numpy as np
from onnx import onnx_pb as onnx_proto

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
onnx_op_set_version = 10

type_to_name = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
}

# Quantization mode
# IntegerOps: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
# QLinearOps: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.
class QuantizationMode():
    IntegerOps = 0
    QLinearOps = 1

# Data Quantization mode
# Linear_NonScaled: Quantize data using linear, non scaled tranformation.
# Linear_Scaled: Quantize data using linear, scaled transformation.
class DataQuantizationMode():
    Linear_NonScaled = 0
    Linear_Scaled = 1

    @staticmethod
    def mode_for_data_type(data_type):
        return DataQuantizationMode.Linear_Scaled if data_type == onnx_proto.TensorProto.INT8\
            else DataQuantizationMode.Linear_NonScaled


quantization_modes = [getattr(QuantizationMode, attr) for attr in dir(QuantizationMode)\
    if not callable(getattr(QuantizationMode, attr)) and not attr.startswith("__")]
data_quantization_modes = [getattr(DataQuantizationMode, attr) for attr in dir(DataQuantizationMode)\
    if not callable(getattr(DataQuantizationMode, attr)) and not attr.startswith("__")]


class Weight:
    '''
        Represents a linearly quantized weight input from ONNX operators
    '''
    def __init__(self, name, initializer, rmins, rmaxs, zero_points, scales, data=[], quantized_data=[], axis=None,
                 qType=onnx_proto.TensorProto.UINT8):
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        self.zero_points = zero_points  # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        self.axis = axis  # Scalar to specify which dimension in the initializer to weight pack.
                          # If empty, single zero point and scales computed from a single rmin and rmax
        self.qType = qType # type of quantized data.


def quantize_data(data, quantize_range, mode=DataQuantizationMode.Linear_NonScaled):
    '''
        :parameter quantize_range: list of data to weight pack.
        :parameter mode: mode to quantize data of type DataQuantizationMode
        :return: minimum, maximum, zero point, scale, and quantized weights

        To pack weights, we compute a linear transformation
            - in non-scaled mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - in scaled mode, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
                m = max(abs(rmin), abs(rmax))

        and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation
        r = S(q-z), where
            r: real original value
            q: quantized value
            S: scale
            z: zero point
    '''
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    if mode == DataQuantizationMode.Linear_Scaled:
        max_range = max(abs(rmin), abs(rmax))
        scale = (float(max_range)*2) / quantize_range
        zero_point = 0
        quantized_data = (np.asarray(data) / scale).round().astype('b') #signed byte type
    else:
        scale = (float(rmax) - rmin) / quantize_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale) # round to nearest integer
        quantized_data = ((np.asarray(data) / scale).round() + zero_point).astype('B') # unsigned byte type
    return rmin, rmax, zero_point, scale, quantized_data


def _attribute_to_kwarg(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    '''
    if (attribute.type == 0):
        raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if (attribute.type == 1):
        value = attribute.f
    elif (attribute.type == 2):
        value = attribute.i
    elif (attribute.type == 3):
        value = attribute.s
    elif (attribute.type == 4):
        value = attribute.t
    elif (attribute.type == 5):
        value = attribute.g
    elif (attribute.type == 6):
        value = attribute.floats
    elif (attribute.type == 7):
        value = attribute.ints
    elif (attribute.type == 8):
        value = attribute.strings
    elif (attribute.type == 9):
        value = attribute.tensors
    elif (attribute.type == 10):
        value = attribute.graphs
    else:
        raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

    return {attribute.name: value}

def _find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None

def _get_mul_node(inputs, output, name):
    '''
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    '''
    return onnx.helper.make_node("Mul", inputs, [output], name)

def _find_node_by_name(node_name, graph, new_nodes_list):
    '''
    Helper function to check if a node exists in a graph or
    new set of nodes created during quantization.
        parameter node_name: name of the node.
        parameter graph: GraphProto.
        parameter new_nodes_list: list of nodes added during quantization.
        return: NodeProto if found. None otherwise.
    '''
    graph_nodes_list = list(graph.node) # deep copy
    graph_nodes_list.extend(new_nodes_list)
    node = _find_by_name(node_name, graph_nodes_list)
    return node

def _add_initializer_if_not_present(graph, name, value, shape, type):
    '''
    Helper function to add an initializer if it is not present in the graph.
        parameter graph: GraphProto.
        parameter name: Initializer's name.
        parameter value: Initializer's value.
        parameter shape: Initializer's shape.
        parameter type: Initializer's type.
    '''
    if _find_by_name(name, graph.initializer) is None:
        initializer = onnx.helper.make_tensor(name, type, shape, value)
        value_info = onnx.helper.make_tensor_value_info(name, type, shape)
        graph.initializer.extend([initializer])
        graph.input.extend([value_info])

def _get_qrange_for_qType(qType):
    '''
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    '''
    if qType == onnx_proto.TensorProto.UINT8:
        return 255  # 2^b - 1
    elif qType == onnx_proto.TensorProto.INT8:
        return 254  # [-(2^{b-1}-1), 2^{b-1}-1]: [-127, 127] for 8 bits.
    else:
        raise ValueError('unsupported quantization data type')

def _find_nodes_using_initializer(graph, initializer):
    '''
    Helper function to find all nodes with an initializer as a input.
        parameter graph: GraphProto.
        parameter initializer: Initializer in TensorProto format.
        return: List of nodes.
    '''
    result = []
    for node in graph.node:
        for node_input in node.input:
            if node_input == initializer.name:
                result.append(node)
    return result

class ONNXQuantizer:
    def __init__(self, model, per_channel, mode, static, weight_qType, input_qType,
            input_quantization_params, output_quantization_params, nodes_to_quantize):
        self.model = model
        self.per_channel = per_channel # weight-pack per channel
        self.weight_qType = weight_qType  # quantize data type
        self.mode = mode # QuantizationMode.Value
        self.static = static # use static quantization for inputs.
        self.input_qType = input_qType # quantize input type
        self.input_quantization_params = input_quantization_params # zero point and scale values for node inputs.
        self.output_quantization_params = output_quantization_params # zero point and scale values for node outputs.
        self.nodes_to_quantize = nodes_to_quantize # specific nodes to quantize

        if not self.mode in quantization_modes:
            raise ValueError('unsupported quantization mode {}'.format(self.mode))

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_non_scaled_name = "fixed_quantization_range_non_scaled"
        self.fixed_qrange_scaled_name = "fixed_quantization_range_scaled"
        # In non scaled mode, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # In scaled mode, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # List of weights quantized.
        self._quantized_weights = []
    
    def quantize_model(self):
        # Create a new topologically sorted list for quantizing a model
        new_list = []
        for node in self.model.graph.node:
            if self.nodes_to_quantize is not None and node.name not in self.nodes_to_quantize:
                new_list.append(node)
            else:
                if node.op_type == 'Conv' and len(node.input) == 2:
                    new_list += self._quantize_convolution(node, new_list)
                elif node.op_type == 'MatMul':
                    new_list += self._quantize_matmul(node, new_list)
                elif node.op_type == 'Gather':
                    new_list += self._quantize_gather_ops(node, new_list)
                else:
                    new_list.append(node)

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph.ClearField('node')
        self.model.graph.node.extend(new_list)

        # Remove weights which are already quantized from graph.
        self._remove_quantized_weights()

        # update opset.
        opset_info = next((opset for opset in self.model.opset_import if opset.domain == '' or opset.domain == onnx_domain), None)
        if opset_info is not None:
            self.model.opset_import.remove(opset_info)
        self.model.opset_import.extend([onnx.helper.make_opsetid(onnx_domain, onnx_op_set_version)])

        return self.model

    def find_weight_data(self, initializer):
        '''
            :param initializer: TensorProto initializer object from a graph
            :return: a list of initialized data in a given initializer object
        '''
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer)
        else:
            raise ValueError('Model contains conv operator weights in {}. Only float type quantization is supported.'.format(
                type_to_name[initializer.data_type]))
        return weights

    def _remove_quantized_weights(self):
        ''' Remove the weights which are already quantized from graph initializer list.
            This function assumes that after quantization, all nodes that previously use a weight:
                - use output from DequantizeLinear as input if they do not support quantization.
                - use quantized weight if they support quantization.
        '''
        for weight in self._quantized_weights:
            # Remove existing weight initializer
            self.model.graph.initializer.remove(weight.initializer)

            # Removing input weight to a convolution
            try:
                weight_input = next(val for val in self.model.graph.input if val.name == weight.name)
                self.model.graph.input.remove(weight_input)
            except StopIteration:
                if self.model.ir_version < 4:
                    raise ValueError('invalid weight name {} found in the graph (not a graph input) '.format(weight.name))


    def _update_graph(self, weight):
        '''
            Given a weight object, update the graph by doing the following:
             - remove old initializer, update new initializers for quantized weight, zero point, and scale
             - remove old weight input, update with new inputs for quantized weight, zero point, and scale
            This function does NOT update the nodes in the graph, just initializers and inputs
        '''
        packed_weight_name = weight.name + '_quantized'
        scale_name = weight.name + '_scale'
        zero_point_name = weight.name + '_zero_point'

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(weight.quantized_data,
            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight.qType]).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else: # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, weight.scales)
        zero_initializer = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points)

        self.model.graph.initializer.extend([packed_weight_initializer, scale_initializer, zero_initializer])

        # Create input for initialized scale and zeros
        packed_weight_value_info = onnx.helper.make_tensor_value_info(packed_weight_name, weight.qType,
                                        weight.initializer.dims)
        scale_value_info = onnx.helper.make_tensor_value_info(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape)
        zero_point_value_info = onnx.helper.make_tensor_value_info(zero_point_name,
            zero_point_type, zero_scale_shape) # zero_point is int for dequantize operator

        self.model.graph.input.extend([packed_weight_value_info, scale_value_info, zero_point_value_info])

        self._quantized_weights.append(weight)

    def _get_quantized_weight(self, initializer, qType):
        '''
            :param initializer: TensorProto initializer
            :param qType: type to quantize to
            :return: Weight class with quantization information
        '''
        weights_data = self.find_weight_data(initializer)
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(weights_data.flatten().tolist(),
            _get_qrange_for_qType(qType), mode=DataQuantizationMode.mode_for_data_type(qType))
        weight = Weight(initializer.name, initializer, [rmin], [rmax], [zero_point], [scale],
                        weights_data, quantized_weights_data, axis=None, qType=qType)
        return weight

    def _get_quantized_weight_convolution(self, initializer, qType):
        '''
            :param initializer: initializer TypeProto to quantize
            :param qType: type to quantize to
            :return: Weight class object with quantization information for a given initializer
        '''
        if not self.per_channel:
            return self._get_quantized_weight(initializer, qType)

        weights = self.find_weight_data(initializer)
        # Quantize per output channel
        # Assuming (M x C/group x kH x kW) format where M is number of output channels.
        channel_count = initializer.dims[0]
        np_data = np.reshape(weights, initializer.dims)
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            # for each channel, compute quantization data. Assuming (M x C/group x kH x kW)
            per_channel_data = np_data[i,:,:,:].flatten()
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data(per_channel_data.flatten().tolist(),
                _get_qrange_for_qType(qType), mode=DataQuantizationMode.mode_for_data_type(qType))
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)
        channel_index = 0 # (M x C/group x kH x kW)
        # combine per_channel_data into one
        reshape_dims = list(initializer.dims)  # deep copy
        reshape_dims[channel_index] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), axis=0)

        weight = Weight(initializer.name, initializer, rmin_list, rmax_list, zero_point_list,
                        scale_list, weights, quantized_weights.flatten().tolist(), channel_index, qType)
        return weight

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        mode = DataQuantizationMode.mode_for_data_type(qType)
        if mode == DataQuantizationMode.Linear_Scaled:
            return self._get_dynamic_input_quantization_params_scaled(input_name, nodes_list)

        return self._get_dynamic_input_quantization_params_non_scaled(input_name, nodes_list)

    def _get_dynamic_input_quantization_params_scaled(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list
        in DataQuantizationMode.Linear_Scaled
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name],
            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name],
            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node("Abs", [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"], reduce_min_abs_name)
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node("Abs", [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"], reduce_max_abs_name)
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node("Max", [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"], abs_max_name)
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        _add_initializer_if_not_present(self.model.graph, self.fixed_qrange_scaled_name,
            [_get_qrange_for_qType(qType)/2.0], [], onnx_proto.TensorProto.FLOAT)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [abs_max_node.output[0], self.fixed_qrange_scaled_name],
            [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Zero point
        _add_initializer_if_not_present(self.model.graph, self.fixed_zero_zp_name,
            [0], [], qType)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_non_scaled(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list
        in DataQuantizationMode.Linear_NonScaled
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name],
            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name],
            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        _add_initializer_if_not_present(self.model.graph, self.fixed_qrange_non_scaled_name,
            [_get_qrange_for_qType(qType)], [], onnx_proto.TensorProto.FLOAT)
        _add_initializer_if_not_present(self.model.graph, self.fixed_zero_name,
            [0.0], [], onnx_proto.TensorProto.FLOAT)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node("Sub", [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"], scale_sub_name)
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [scale_sub_node.output[0], self.fixed_qrange_non_scaled_name],
            [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node("Sub", [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"], zp_sub_name)
        nodes_list.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node("Div", [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"], zp_div_name)
        nodes_list.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output,
            [zp_floor_name + ":0"], zp_floor_name)
        nodes_list.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output,
            [input_zp_name], zp_cast_name, to=qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_static_input_quantization_params(self, input_name, qType):
        '''
        Create initializers and inputs in the graph for static quantization of input.

        Zero point and scale values are obtained from self.input_quantization_params if specified.
        ValueError is thrown otherwise.

            parameter input_name: Name of the input.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if self.input_quantization_params is None or input_name not in self.input_quantization_params:
            raise ValueError("Quantization parameters are not specified for input {}.".format(input_name))
        params = self.input_quantization_params[input_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. "
                "Specified values for input {}: {}".format(input_name, params))

        if not np.isscalar(params[0]):
            raise ValueError("Zero point for input {} should be a scalar value. Value specified: {}".format(
                input_name, params[0]))
        if not np.isscalar(params[1]):
            raise ValueError("Scale for input {} should be a scalar value. Value specified: {}".format(
                input_name, params[1]))

        zero_point_values = [params[0].item()]
        zero_point_shape = []
        zero_point_name = input_name + "_zero_point"

        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]
        if zero_point_type != qType:
            raise ValueError("Zero point and input data types should be the same. "
                "Zero point for input {} is specified as {}, but input is being quantized to {}."
                .format(input_name, params[0].dtype, onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[qType]))

        scale_values = [params[1].item()]
        scale_shape = []
        scale_name = input_name + "_scale"

        # Add initializers
        _add_initializer_if_not_present(self.model.graph, zero_point_name, zero_point_values,
            zero_point_shape, qType)
        _add_initializer_if_not_present(self.model.graph, scale_name, scale_values,
            scale_shape, onnx_proto.TensorProto.FLOAT)

        return scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_output_quantization_params(self, output_name):
        '''
        Create initializers and inputs in the graph for zero point and scale of output.
        Used when mode is QuantizationMode.QLinearOps.

        Zero point and scale values are obtained from self.output_quantization_params if specified.
        ValueError is thrown otherwise.

            parameter output_name: Name of the output.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if self.output_quantization_params is None or output_name not in self.output_quantization_params:
            raise ValueError("Quantization parameters are not specified for output {}.".format(output_name))
        params = self.output_quantization_params[output_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. "
                "Specified values for output {}: {}".format(output_name, params))

        if not np.isscalar(params[0]):
            raise ValueError("Zero point for output {} should be a scalar value. Value specified: {}".format(
                output_name, params[0]))
        if not np.isscalar(params[1]):
            raise ValueError("Scale for output {} should be a scalar value. Value specified: {}".format(
                output_name, params[1]))

        zero_point_values = [params[0].item()]
        zero_point_shape = []
        zero_point_name = output_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1].item()]
        scale_shape = []
        scale_name = output_name + "_scale"

        # Add initializers
        _add_initializer_if_not_present(self.model.graph, zero_point_name, zero_point_values, zero_point_shape,
            zero_point_type)
        _add_initializer_if_not_present(self.model.graph, scale_name, scale_values, scale_shape,
            onnx_proto.TensorProto.FLOAT)

        return scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(self, node, input_index, qType):
        '''
        Given a input for a node (which is not a initializer), this function
            - add elements to graph to compute zero point and scale for this input.
            - add new QuantizeLinear nodes to quantize the input.

            parameter node: node being quantized in NodeProto format.
            parameter input_index: index of input in node.input.
            parameter qType: type to quantize to.
            return: List of newly created nodes in NodeProto format.
        '''
        input_name = node.input[input_index]

        nodes = []
        if self.static:
            scale_name, zp_name, scale_shape, zp_shape = \
                self._get_static_input_quantization_params(input_name, qType)
        else:
            scale_name, zp_name, scale_shape, zp_shape = \
                self._get_dynamic_input_quantization_params(input_name, nodes, qType)

        # Add QuantizeLinear Node
        output_name = input_name + "_quantized"
        qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name],
            [output_name], input_name + "_QuantizeLinear")
        return nodes + [qlinear_node]

    def _update_unsupported_nodes_using_weight(self, weight, new_nodes_list):        
        '''Find all nodes using a weight that do not support quantization and
        add a DequantizeLinear node before those nodes. This includes all nodes except Conv, MatMul.

            parameter weight: Weight object
            parameter new_nodes_list: List of new nodes created before processing current node.
            return: List of new nodes created.
        '''
        nodes_using_weight = _find_nodes_using_initializer(self.model.graph, weight.initializer)
        unsupported_nodes = [node for node in nodes_using_weight if node.op_type not in ["Conv", "MatMul", "Gather"]]

        nodes_list = []
        dequantize_linear_name = weight.name + "_DequantizeLinear"
        output_name = weight.name + "_dequantized"

        # Check if DequantizeLinear node needs to be added to graph.
        if len(unsupported_nodes) != 0 and \
            _find_node_by_name(dequantize_linear_name, self.model.graph, new_nodes_list) is None:
            inputs = [weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point"]
            node = onnx.helper.make_node("DequantizeLinear", inputs, [output_name],
                                         dequantize_linear_name)
            nodes_list.append(node)

        # Update unsupported nodes to take dequantized weight as input.
        for node in unsupported_nodes:
            for i, node_input in enumerate(node.input):
                if node_input == weight.name:
                    node.input[i] = output_name

        return nodes_list

    def _is_quantized(self, weight):
        '''
        Check if this weight is already quantized to the expected type and quantization axis.
        If it is already quantized to (type, axis) different from expected values,
        this function will throw an exception and stop the quantization.

            parameter weight: Weight object.
            return: Boolean indicating if quantized weight is already added to graph.
        '''
        quantized_initializer_name = weight.name + "_quantized"
        quantized_initializer = _find_by_name(quantized_initializer_name, self.model.graph.initializer)
        zero_point = _find_by_name(weight.name + "_zero_point", self.model.graph.initializer)
        if quantized_initializer is None:
            return False

        # Compare type
        if quantized_initializer.data_type != weight.qType:
            raise ValueError("{} is being used by multiple nodes which are being quantized to different types. "
                "Please use different initializers for these nodes.", weight.name)

        expected_dims = [] if weight.axis is None else [len(weight.zero_points)]
        # Compare quantization axis
        if zero_point.dims != expected_dims:
            raise ValueError("{} is being used by multiple nodes which are being quantized to different shapes. "
                "Please use different initializers for these nodes.", weight.name)

        return True

    def _quantize_inputs(self, node, indices, weight_index, new_nodes_list):
        '''
        Given a node, this function quantizes the inputs as follows:
            - If input is a initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization

            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            parameter weight_index: index of weight input.
                                    In Asymmetric mode, this input is quantized into signed integer.
            parameter new_nodes_list: List of new nodes created before processing this node. This is used to
                                      check that two QuantizeLinear nodes are not being added for same input.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        '''
        assert (node.op_type == "Conv" or node.op_type == "MatMul" or node.op_type == "Gather")

        quantized_input_names = []
        zero_point_names = []
        scale_names = []
        nodes = []

        for input_index in indices:
            qType = self.weight_qType if input_index == weight_index else self.input_qType
            node_input = node.input[input_index]
            initializer = _find_by_name(node_input, self.model.graph.initializer)
            if initializer is not None:
                # Quantize the data
                if node.op_type == "Conv" and input_index == weight_index:
                    weight = self._get_quantized_weight_convolution(initializer, qType)
                else:
                    weight = self._get_quantized_weight(initializer, qType)

                if not self._is_quantized(weight):
                    nodes.extend(self._update_unsupported_nodes_using_weight(weight, new_nodes_list))
                    self._update_graph(weight)

                quantized_input_names.append(weight.name + "_quantized")
                zero_point_names.append(weight.name + "_zero_point")
                scale_names.append(weight.name + "_scale")
            else:
                # Not an initializer input. Add QuantizeLinear node.
                # Find if there is already a quantizeLinear node for this input
                qlinear_node = _find_node_by_name(node_input + "_QuantizeLinear", self.model.graph, new_nodes_list)
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, qType)
                    nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                quantized_input_names.extend(qlinear_node.output)
                scale_names.append(qlinear_node.input[1])
                zero_point_names.append(qlinear_node.input[2])

        return (quantized_input_names, zero_point_names, scale_names, nodes)

    def _quantize_gather_ops(self, node, new_nodes_list):
        assert (node.op_type == "Gather")
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0], 0, new_nodes_list)
        
        gather_new_output = node.output[0] + "_quantized"
        gather_original_output = node.output[0]
        node.output[0] = gather_new_output
        node.input[0] = quantized_input_names[0]
        nodes.append(node)

        # Add DequantizeLinear node.
        dqlinear_name = node.output[0] + "_DequantizeLinear"
        dqlinear_inputs = [gather_new_output, scale_names[0], zero_point_names[0]]
        dqlinear_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [gather_original_output], dqlinear_name)
        print(dqlinear_node.name)
        nodes.append(dqlinear_node)
        return nodes        

    def _quantize_convolution_integer_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.IntegerOps.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], 1, new_nodes_list)

        conv_integer_output = node.output[0] + "_quantized"
        conv_integer_name = ""
        if node.name != "":
            conv_integer_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(_attribute_to_kwarg(attribute))
        conv_integer_node = onnx.helper.make_node("ConvInteger", quantized_input_names + zero_point_names,
            [conv_integer_output], conv_integer_name, **kwargs)
        nodes.append(conv_integer_node)

        # Add cast operation to cast convInteger output to float.
        cast_op_output = conv_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [conv_integer_output], [cast_op_output],
            conv_integer_output + "_cast", to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        if conv_integer_name != "":
            scales_mul_op = conv_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = _find_node_by_name(scales_mul_op, self.model.graph, new_nodes_list)
        if scales_mul_node is None:
            scales_mul_node = _get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of ConvInteger
        # and make the output of this node the same as output of original conv node.
        output_scale_mul_op = ""
        if conv_integer_name != "":
            output_scale_mul_op = conv_integer_name + "_output_scale_mul"
        nodes.append(_get_mul_node([cast_op_output, scales_mul_op_output], node.output[0], output_scale_mul_op))
        return nodes

    def _quantize_matmul_integer_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.IntegerOps.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized MatMul node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], 1, new_nodes_list)

        matmul_integer_output = node.output[0] + "_quantized"
        matmul_integer_name = ""
        if node.name != "":
            matmul_integer_name = node.name + "_quant"
        matmul_integer_node = onnx.helper.make_node("MatMulInteger", quantized_input_names + zero_point_names,
            [matmul_integer_output], matmul_integer_name)
        nodes.append(matmul_integer_node)

        # Add cast operation to cast matmulInteger output to float.
        cast_op_output = matmul_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [matmul_integer_output], [cast_op_output],
            matmul_integer_output + "_cast", to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        if matmul_integer_name != "":
            scales_mul_op = matmul_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = _find_node_by_name(scales_mul_op, self.model.graph, new_nodes_list)
        if scales_mul_node is None:
            scales_mul_node = _get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of MatMulInteger
        # and make the output of this node the same as output of original matmul node.
        output_scale_mul_op = ""
        if matmul_integer_name != "":
            output_scale_mul_op = matmul_integer_name + "_output_scale_mul"
        nodes.append(_get_mul_node([cast_op_output, scales_mul_op_output], node.output[0],
            output_scale_mul_op))
        return nodes

    def _quantize_convolution_qlinear_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.QLinearOps.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], 1, new_nodes_list)

        output_scale_name, output_zp_name, output_scale_shape, output_zp_shape = \
            self._get_output_quantization_params(node.output[0])

        qlinear_conv_output = node.output[0] + "_quantized"
        qlinear_conv_name = ""
        if node.name != "":
            qlinear_conv_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(_attribute_to_kwarg(attribute))
        qlinear_conv_inputs = []
        # Input 0
        qlinear_conv_inputs.append(quantized_input_names[0])
        qlinear_conv_inputs.append(scale_names[0])
        qlinear_conv_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_conv_inputs.append(quantized_input_names[1])
        qlinear_conv_inputs.append(scale_names[1])
        qlinear_conv_inputs.append(zero_point_names[1])
        # Output
        qlinear_conv_inputs.append(output_scale_name)
        qlinear_conv_inputs.append(output_zp_name)

        qlinear_conv_node = onnx.helper.make_node("QLinearConv", qlinear_conv_inputs,
            [qlinear_conv_output], qlinear_conv_name, **kwargs)
        nodes.append(qlinear_conv_node)

        # Add DequantizeLinear node.
        dqlinear_name = node.output[0] + "_DequantizeLinear"
        dqlinear_inputs = [qlinear_conv_output, output_scale_name, output_zp_name]
        dqlinear_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [node.output[0]], dqlinear_name)
        nodes.append(dqlinear_node)
        return nodes

    def _quantize_matmul_qlinear_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.QLinearOps.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], 1, new_nodes_list)

        output_scale_name, output_zp_name, output_scale_shape, output_zp_shape = \
            self._get_output_quantization_params(node.output[0])

        qlinear_matmul_output = node.output[0] + "_quantized"
        qlinear_matmul_name = ""
        if node.name != "":
            qlinear_matmul_name = node.name + "_quant"

        qlinear_matmul_inputs = []
        # Input 0
        qlinear_matmul_inputs.append(quantized_input_names[0])
        qlinear_matmul_inputs.append(scale_names[0])
        qlinear_matmul_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_matmul_inputs.append(quantized_input_names[1])
        qlinear_matmul_inputs.append(scale_names[1])
        qlinear_matmul_inputs.append(zero_point_names[1])
        # Output
        qlinear_matmul_inputs.append(output_scale_name)
        qlinear_matmul_inputs.append(output_zp_name)

        qlinear_matmul_node = onnx.helper.make_node("QLinearMatMul", qlinear_matmul_inputs,
            [qlinear_matmul_output], qlinear_matmul_name)
        nodes.append(qlinear_matmul_node)

        # Add DequantizeLinear node.
        dqlinear_name = node.output[0] + "_DequantizeLinear"
        dqlinear_inputs = [qlinear_matmul_output, output_scale_name, output_zp_name]
        dqlinear_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [node.output[0]], dqlinear_name)
        nodes.append(dqlinear_node)
        return nodes

    def _quantize_convolution(self, node, new_nodes_list):
        '''
            https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
            :param node: Conv node
            :param new_nodes_list: List of new nodes created before processing this node.
            :return: a list of nodes in topological order that represents quantized Conv node
        '''
        assert (node.op_type == "Conv")

        if self.mode == QuantizationMode.IntegerOps:
            return self._quantize_convolution_integer_ops(node, new_nodes_list)

        if self.mode == QuantizationMode.QLinearOps:
            return self._quantize_convolution_qlinear_ops(node, new_nodes_list)

        return [node]

    def _quantize_matmul(self, node, new_nodes_list):
        '''
            https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
            :param node: MatMul node
            :param new_nodes_list: List of new nodes created before processing this node.
            :return: a list of nodes in topological order that represents quantized MatMul node
        '''
        assert(node.op_type == 'MatMul')

        if self.mode == QuantizationMode.IntegerOps:
            return self._quantize_matmul_integer_ops(node, new_nodes_list)

        if self.mode == QuantizationMode.QLinearOps:
            return self._quantize_matmul_qlinear_ops(node, new_nodes_list)

        return [node]


def quantize(model, per_channel=False, nbits=8, quantization_mode=QuantizationMode.IntegerOps,
    static=False, asymmetric_input_types=False, input_quantization_params=None, output_quantization_params=None, nodes_to_quantize=None):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file

    :param model: ModelProto to quantize
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param quantization_mode: Can be one of the QuantizationMode types.
        IntegerOps:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
        QLinearOps:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
    :param static:
        True: The inputs/activations are quantized using static scale and zero point values
              specified through input_quantization_params.
        False: The inputs/activations are quantized using dynamic scale and zero point values
               computed while running the model.
    :param asymmetric_input_types:
        True: Weights are quantized into signed integers and inputs/activations into unsigned integers.
        False: Weights and inputs/activations are quantized into unsigned integers.
    :param input_quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The input_quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :param output_quantization_params:
        Dictionary to specify the zero point and scale values for outputs of conv and matmul nodes.
        Should be specified in QuantizationMode.QLinearOps mode.
        The output_quantization_params should be specified in the following format:
            {
                "output_name": [zero_point, scale]
            }
        zero_point can be of type np.uint8/np.int8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_3:0': [np.int8(0), np.float32(0.011359662748873234)],
                'resnet_model/Relu_4:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :return: ModelProto with quantization
    :param nodes_to quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        exmaple:
        [
            'Cov__224',
            'Conv__252'
        ]
    '''
    if nbits == 8:
        input_qType = onnx_proto.TensorProto.UINT8
        weight_qType = onnx_proto.TensorProto.INT8 if asymmetric_input_types else onnx_proto.TensorProto.UINT8
        mode = quantization_mode
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)
        quantizer = ONNXQuantizer(copy_model, per_channel, mode, static, weight_qType, input_qType,
                        input_quantization_params, output_quantization_params, nodes_to_quantize)
        quantizer.quantize_model()
        quantizer.model.producer_name = __producer__
        quantizer.model.producer_version = __version__
        return quantizer.model
    else:
        raise ValueError('Unknown value for nbits. only 8 bit quantization is currently supported')