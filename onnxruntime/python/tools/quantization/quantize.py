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
onnx_version = 10

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
# IntegerOps_Static: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
#               Use static quantization for weights and inputs/activations.
# IntegerOps_Dynamic: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
#               Use static quantization for weights and dynamic quantization for inputs/activations.
# QLinearOps_Static: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.
#               Use static quantization for weights and inputs/activations.
# QLinearOps_Dynamic: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.
#               Use static quantization for weights and dynamic quantization for inputs/activations.
class QuantizationMode():
    IntegerOps_Static = 0
    IntegerOps_Dynamic = 1
    QLinearOps_Static = 2
    QLinearOps_Dynamic = 3

    @staticmethod
    def is_integer_ops_mode(mode):
        return mode in [QuantizationMode.IntegerOps_Static, QuantizationMode.IntegerOps_Dynamic]

    @staticmethod
    def is_qlinear_ops_mode(mode):
        return mode in [QuantizationMode.QLinearOps_Static, QuantizationMode.QLinearOps_Dynamic]

    @staticmethod
    def is_static_mode(mode):
        return mode in [QuantizationMode.IntegerOps_Static, QuantizationMode.QLinearOps_Static]

# Data Quantization mode
# Linear_NonScaled: Quantize data using linear, non scaled tranformation.
# Linear_Scaled: Quantize data using linear, scaled transformation.
class DataQuantizationMode():
    Linear_NonScaled = 0
    Linear_Scaled = 1

quantization_modes = [getattr(QuantizationMode, attr) for attr in dir(QuantizationMode)\
    if not callable(getattr(QuantizationMode, attr)) and not attr.startswith("__")]
data_quantization_modes = [getattr(DataQuantizationMode, attr) for attr in dir(DataQuantizationMode)\
    if not callable(getattr(DataQuantizationMode, attr)) and not attr.startswith("__")]


class Weight:
    '''
        Represents a linearly quantized weight input from ONNX operators
    '''
    def __init__(self, name, initializer, rmins, rmaxs, zero_points, scales, data=[], quantized_data=[], axis=None):
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
    rmin = min(data)
    rmax = max(data)

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


class ONNXQuantizer:
    def __init__(self, model, per_channel, mode, weight_qType, input_qType, data_quantization_mode,
            input_quantization_params, output_quantization_params):
        self.model = model
        self.per_channel = per_channel # weight-pack per channel
        self.weight_qType = weight_qType  # quantize data type
        self.mode = mode # QuantizationMode.Value
        self.input_qType = input_qType # quantize input type
        self.data_quantization_mode = data_quantization_mode # DataQuantizationMode.Value
        self.input_quantization_params = input_quantization_params # zero point and scale values for node inputs.
        self.output_quantization_params = output_quantization_params # zero point and scale values for node outputs.

        if not self.mode in quantization_modes:
            raise ValueError('unsupported quantization mode {}'.format(self.mode))
        if not self.data_quantization_mode in data_quantization_modes:
            raise ValueError('unsupported data quantization mode {}'.format(self.data_quantization_mode))

        if self.weight_qType == onnx_proto.TensorProto.UINT8:
            self.weight_qrange = 255  # 2^b - 1
        elif self.weight_qType == onnx_proto.TensorProto.INT8:
            assert(self.data_quantization_mode == DataQuantizationMode.Linear_Scaled)
            self.weight_qrange = 254  # [-(2^{b-1}-1), 2^{b-1}-1]: [-127, 127] for 8 bits.
        else:
            raise ValueError('unsupported quantization data type')

        if self.input_qType == onnx_proto.TensorProto.UINT8:
            self.input_qrange = 255 # 2^b - 1
        else:
            raise ValueError('unsupported quantization data type')

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when QuantizationMode.is_static_mode() is False
        self.fixed_qrange_name = "fixed_quantization_range"
        self.fixed_zero_name = "fixed_zero"

    def quantize_model(self):
        # Create a new topologically sorted list for quantizing a model
        new_list = []
        for node in self.model.graph.node:
            if node.op_type == 'Conv':
                new_list += self._quantize_convolution(node, new_list)
            elif node.op_type == 'MatMul':
                new_list += self._quantize_matmul(node, new_list)
            else:
                new_list.append(node)

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph.ClearField('node')
        self.model.graph.node.extend(new_list)

        # update opset.
        opset_info = next((opset for opset in self.model.opset_import if opset.domain == '' or opset.domain == onnx_domain), None)
        if opset_info is not None:
            self.model.opset_import.remove(opset_info)
        self.model.opset_import.extend([onnx.helper.make_opsetid(onnx_domain, onnx_version)])

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

        # Remove existing weight initializer
        self.model.graph.initializer.remove(weight.initializer)
        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(weight.quantized_data,
            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.weight_qType]).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else: # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = self.weight_qType
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, weight.scales)
        zero_initializer = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points)

        self.model.graph.initializer.extend([packed_weight_initializer, scale_initializer, zero_initializer])

        # Removing input weight to a convolution
        try:
            weight_input = next(val for val in self.model.graph.input if val.name == weight.name)
        except StopIteration:
            raise ValueError('invalid weight name {} found in the graph '.format(weight.name))
        self.model.graph.input.remove(weight_input)

        # Create input for initialized scale and zeros
        packed_weight_value_info = onnx.helper.make_tensor_value_info(packed_weight_name, self.weight_qType,
                                        weight.initializer.dims)
        scale_value_info = onnx.helper.make_tensor_value_info(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape)
        zero_point_value_info = onnx.helper.make_tensor_value_info(zero_point_name,
            zero_point_type, zero_scale_shape) # zero_point is int for dequantize operator

        self.model.graph.input.extend([packed_weight_value_info, scale_value_info, zero_point_value_info])

    def _get_quantized_weight(self, initializer):
        '''
            :param initializer: TensorProto initializer
            :return: Weight class with quantization information
        '''
        weights_data = self.find_weight_data(initializer)
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(weights_data.flatten().tolist(),
            self.weight_qrange, mode=self.data_quantization_mode)
        weight = Weight(initializer.name, initializer, [rmin], [rmax], [zero_point], [scale], weights_data, quantized_weights_data)
        return weight

    def _get_quantized_weight_convolution(self, initializer):
        '''
            :param initializer: initializer TypeProto to quantize
            :return: Weight class object with quantization information for a given initializer
        '''
        if not self.per_channel:
            return self._get_quantized_weight(initializer)

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
                self.weight_qrange, mode=self.data_quantization_mode)
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

        weight = Weight(initializer.name, initializer, rmin_list, rmax_list,
                        zero_point_list, scale_list, weights, quantized_weights.flatten().tolist(), channel_index)
        return weight

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
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
        _add_initializer_if_not_present(self.model.graph, self.fixed_qrange_name,
            [self.input_qrange], [], onnx_proto.TensorProto.FLOAT)
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
        scale_div_node = onnx.helper.make_node("Div", [scale_sub_node.output[0], self.fixed_qrange_name],
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
            [input_zp_name], zp_cast_name, to=self.input_qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_static_input_quantization_params(self, input_name):
        '''
        Create initializers and inputs in the graph for static quantization of input.

        Zero point and scale values are obtained from self.input_quantization_params if specified.
        ValueError is thrown otherwise.

            parameter input_name: Name of the input.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if self.input_quantization_params is None or input_name not in self.input_quantization_params:
            raise ValueError("Quantization parameters are not specified for input {}.".format(input_name))
        params = self.input_quantization_params[input_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. \
                Specified values for input {}: {}".format(input_name, params))

        if not np.isscalar(params[0]):
            raise ValueError("Zero point for input {} should be a scalar value. Value specified: {}".format(
                input_name, params[0]))
        if not np.isscalar(params[1]):
            raise ValueError("Scale for input {} should be a scalar value. Value specified: {}".format(
                input_name, params[1]))

        zero_point_values = [params[0].item()]
        zero_point_shape = []
        zero_point_name = input_name + "_zero_point"

        scale_values = [params[1].item()]
        scale_shape = []
        scale_name = input_name + "_scale"

        # Add initializers
        _add_initializer_if_not_present(self.model.graph, zero_point_name, zero_point_values,
            zero_point_shape, self.input_qType)
        _add_initializer_if_not_present(self.model.graph, scale_name, scale_values,
            scale_shape, onnx_proto.TensorProto.FLOAT)

        return scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_output_quantization_params(self, output_name):
        '''
        Create initializers and inputs in the graph for zero point and scale of output.
        Used when QuantizationMode.is_qlinear_ops_mode() is true.

        Zero point and scale values are obtained from self.output_quantization_params if specified.
        ValueError is thrown otherwise.

            parameter output_name: Name of the output.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if self.output_quantization_params is None or output_name not in self.output_quantization_params:
            raise ValueError("Quantization parameters are not specified for output {}.".format(output_name))
        params = self.output_quantization_params[output_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. \
                Specified values for output {}: {}".format(output_name, params))

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

    def _get_quantize_input_nodes(self, node, input_index):
        '''
        Given a input for a node (which is not a initializer), this function
            - add elements to graph to compute zero point and scale for this input.
            - add new QuantizeLinear nodes to quantize the input.

            parameter node: node being quantized in NodeProto format.
            parameter input_index: index of input in node.input.
            return: List of newly created nodes in NodeProto format.
        '''
        input_name = node.input[input_index]

        nodes = []
        if QuantizationMode.is_static_mode(self.mode):
            scale_name, zp_name, scale_shape, zp_shape = \
                self._get_static_input_quantization_params(input_name)
        else:
            scale_name, zp_name, scale_shape, zp_shape = \
                self._get_dynamic_input_quantization_params(input_name, nodes)

        # Add QuantizeLinear Node
        output_name = input_name + "_quantized"
        qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name],
            [output_name], input_name + "_QuantizeLinear")
        return nodes + [qlinear_node]

    def _quantize_inputs(self, node, indices, new_nodes_list):
        '''
        Given a node, this function quantizes the inputs as follows:
            - If input is a initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization

            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            parameter new_nodes_list: List of new nodes created before processing this node. This is used to
                                      check that two QuantizeLinear nodes are not being added for same input.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        '''
        assert (node.op_type == "Conv" or node.op_type == "MatMul")

        quantized_input_names = []
        zero_point_names = []
        scale_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]
            initializer = _find_by_name(node_input, self.model.graph.initializer)
            if initializer is not None:
                # Quantize the data
                if node.op_type == "Conv" and input_index == 1: # Weight index
                    weight = self._get_quantized_weight_convolution(initializer)
                else:
                    weight = self._get_quantized_weight(initializer)
                self._update_graph(weight)

                quantized_input_names.append(weight.name + "_quantized")
                zero_point_names.append(weight.name + "_zero_point")
                scale_names.append(weight.name + "_scale")
            else:
                # Not an initializer input. Add QuantizeLinear node.
                # Find if there is already a quantizeLinear node for this input
                qlinear_node = _find_node_by_name(node_input + "_QuantizeLinear", self.model.graph, new_nodes_list)
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index)
                    nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                quantized_input_names.extend(qlinear_node.output)
                scale_names.append(qlinear_node.input[1])
                zero_point_names.append(qlinear_node.input[2])

        return (quantized_input_names, zero_point_names, scale_names, nodes)

    def _quantize_convolution_integer_ops(self, node, new_nodes_list):
        '''
        Used when QuantizationMode.is_integer_ops_mode(self.mode) is true.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

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
        Used when QuantizationMode.is_integer_ops_mode(self.mode) is true.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized MatMul node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

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
        Used when QuantizationMode.is_qlinear_ops_mode(self.mode) is true.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

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
        Used when QuantizationMode.is_qlinear_ops_mode(self.mode) is true.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

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

        if QuantizationMode.is_integer_ops_mode(self.mode):
            return self._quantize_convolution_integer_ops(node, new_nodes_list)

        if QuantizationMode.is_qlinear_ops_mode(self.mode):
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

        if QuantizationMode.is_integer_ops_mode(self.mode):
            return self._quantize_matmul_integer_ops(node, new_nodes_list)

        if QuantizationMode.is_qlinear_ops_mode(self.mode):
            return self._quantize_matmul_qlinear_ops(node, new_nodes_list)

        return [node]


def quantize(model, per_channel=True, nbits=8, quantization_mode=QuantizationMode.IntegerOps_Dynamic,
    asymmetric_input_types=False, input_quantization_params=None, output_quantization_params=None):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file

    :param model: ModelProto to quantize
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param quantization_mode: Can be one of the QuantizationMode types.
        IntegerOps_Static:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
            The inputs/activations are quantized using static scale and zero point values
            specified through input_quantization_params.
        IntegerOps_Dynamic:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
            The inputs/activations are quantized using dynamic scale and zero point values
            computed while running the model.
        QLinearOps_Static:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
            The inputs/activations are quantized using static scale and zero point values
            specified through input_quantization_params.
        QLinearOps_Dynamic:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
            The inputs/activations are quantized using dynamic scale and zero point values
            computed while running the model.
    :param asymmetric_input_types:
        True: Weights are quantized into signed integers and inputs/activations into unsigned integers.
        False: Weights and inputs/activations are quantized into unsigned integers.
    :param input_quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Used in QuantizationMode.IntegerOps_Static or QuantizationMode.QLinearOps_Static mode.
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
        Used in QuantizationMode.QLinearOps_Static or QuantizationMode.QLinearOps_Dynamic mode.
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
    '''
    if nbits == 8:
        input_qType = onnx_proto.TensorProto.UINT8
        weight_qType = onnx_proto.TensorProto.INT8 if asymmetric_input_types else onnx_proto.TensorProto.UINT8
        mode = quantization_mode
        data_quantization_mode = DataQuantizationMode.Linear_Scaled if asymmetric_input_types \
                                    else DataQuantizationMode.Linear_NonScaled
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)
        quantizer = ONNXQuantizer(copy_model, per_channel, mode, weight_qType, input_qType, data_quantization_mode,
                        input_quantization_params, output_quantization_params)
        quantizer.quantize_model()
        quantizer.model.producer_name = __producer__
        quantizer.model.producer_version = __version__
        return quantizer.model
    else:
        raise ValueError('Unknown value for nbits. only 8 bit quantization is currently supported')