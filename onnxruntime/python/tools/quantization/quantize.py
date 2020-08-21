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
from onnx import shape_inference
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode,QuantizedValueType,QuantizedInitializer,QuantizedValue,quantization_modes
from .quant_utils import _find_by_name,_get_elem_index,_get_mul_node,_generate_identified_filename,_attribute_to_kwarg

from .registry import CreateOpQuantizer, CreateDefaultOpQuantizer

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"
onnx_op_set_version = 11


def quantize_data(data, quantize_range, qType):
    '''
        :parameter data: data to quantize
        :parameter quantize_range: list of data to weight pack.
        :parameter qType: data type to quantize to. Supported types UINT8 and INT8
        :return: minimum, maximum, zero point, scale, and quantized weights
        To pack weights, we compute a linear transformation
            - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
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

    if qType == onnx_proto.TensorProto.INT8:
        max_range = max(abs(rmin), abs(rmax))
        scale = (float(max_range) * 2) / quantize_range
        zero_point = 0
        # signed byte type
        quantized_data = (np.asarray(data) / scale).round().astype('b')
    elif qType == onnx_proto.TensorProto.UINT8:
        scale = (float(rmax) - rmin) / quantize_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale)  # round to nearest integer
        quantized_data = ((np.asarray(data) / scale).round() + zero_point).astype('B')  # unsigned byte type
    else:
        raise ValueError("Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType))

    return rmin, rmax, zero_point, scale, quantized_data


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


def optimize_model(model_path:Path):
        '''
            Generate model that applies graph optimization (constant folding,etc.)
            parameter model_path: path to the original onnx model
            return: optimized onnx model
        '''
        opt_model_path = _generate_identified_filename(model_path,"-opt")
        sess_option = SessionOptions()
        sess_option.optimized_model_filepath = opt_model_path.as_posix()
        sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = InferenceSession(model_path.as_posix(),sess_option)
        optimized_model = onnx.load(opt_model_path.as_posix())
        return optimized_model


def check_opset_version(org_model, force_fusions):         
    '''
        Check opset version of original model and set opset version and fuse_dynamic_quant accordingly.
        If opset version < 10, set quantized model opset version to 10.
        If opset version == 10, do quantization without using dynamicQuantizeLinear operator.
        If opset version == 11, do quantization using dynamicQuantizeLinear operator.
        :return: fuse_dynamic_quant boolean value.
    '''
    global onnx_op_set_version
    ai_onnx_domain = [opset for opset in org_model.opset_import if not opset.domain or opset.domain == "ai.onnx"]
    if 1 != len(ai_onnx_domain):
        raise ValueError('Failed to find proper ai.onnx domain')
    opset_version = ai_onnx_domain[0].version

    fuse_dynamic_quant = False

    if opset_version < 11 and force_fusions == True:
        print("Warning: The original model opset version is {}, which does not support node fusions.\n\
            Forcing fusions can break other nodes in the model.".format(opset_version))
        onnx_op_set_version = 11
        fuse_dynamic_quant = True
        return fuse_dynamic_quant

    if opset_version < 10:
        print("Warning: The original model opset version is {}, which does not support quantized operators.\n\
            The opset version of quantized model will be set to 10. Use onnx model checker to verify model after quantization."
              .format(opset_version))
        onnx_op_set_version = 10
    elif opset_version == 10:
        onnx_op_set_version = 10
    else:
        fuse_dynamic_quant = True
    return fuse_dynamic_quant



class ONNXModel:
    def __init__(self, model):
        self.model = model
        self.node_name_counter = {}

    def nodes(self):
        return self.model.graph.node

    def initializer(self):
        return self.model.graph.initializer

    def graph(self):
        return self.model.graph

    def ir_version(self):
        return self.model.ir_version

    def opset_import(self):
        return self.model.opset_import
    
    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self.model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        if _find_by_name(tensor.name, self.model.graph.initializer) is None:
            self.model.graph.initializer.extend([tensor])

    def get_initializer(self, name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None   

    def remove_initializer(self,tensor):
        if tensor in self.model.graph.initializer:
            self.model.graph.initializer.remove(tensor)
        
    def remove_initializers(self,init_to_remove):
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                output_name_to_node[output_name] = node      
        return output_name_to_node

    def get_children(self, node, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for node in input_name_to_nodes[output]:
                    children.append(node)
        return children
        
    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents
    
    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]
    
    def find_node_by_name(self,node_name,new_nodes_list,graph):
        '''
        Find out if a node exists in a graph or a node is in the 
        new set of nodes created during quantization. Return the node found.
        '''
        graph_nodes_list = list(graph.node)  #deep copy
        graph_nodes_list.extend(new_nodes_list)
        node = _find_by_name(node_name, graph_nodes_list)
        return node

    def find_nodes_by_initializer(self,graph,initializer):
        '''
        Find all nodes with given initializer as an input.
        '''
        nodes = []
        for node in graph.node:
            for node_input in node.input:
                if node_input == initializer.name:
                    nodes.append(node)
        return nodes


class ONNXQuantizer:
    def __init__(self, model:ONNXModel, value_infos, per_channel, mode, static, fuse_dynamic_quant, weight_qType, input_qType,
                 quantization_params, nodes_to_quantize, nodes_to_exclude):
        self.model = model    
        self.value_infos = value_infos
        self.per_channel = per_channel  
        self.mode = mode  
        self.static = static  
        self.fuse_dynamic_quant = fuse_dynamic_quant
        self.input_qType = input_qType  
        self.weight_qType = weight_qType 
        self.quantization_params = quantization_params
        self.nodes_to_quantize = nodes_to_quantize  
        self.nodes_to_exclude = nodes_to_exclude  
        self.new_nodes = []

        if not self.mode in quantization_modes:
            raise ValueError('unsupported quantization mode {}'.format(self.mode))

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # List of quantized weights
        self._quantized_weights = []
        # Map of all original value names to quantized value names
        self.quantized_value_map = {}

    def replace_gemm_with_matmul(self):
        nodes_to_remove = []
        nodes_to_add = []
       
        for node in self.model.nodes():
            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0 and transB == 0:
                    matmul_node = onnx.helper.make_node(
                        'MatMul',
                        [node.input[0], node.input[1]],                                                      
                        [node.output[0]+'_MatMul'],
                        name=node.output[0]+'_MatMul')

                    add_node = onnx.helper.make_node(
                        'Add',
                        inputs=[node.output[0]+'_MatMul', node.input[2]],
                        outputs=node.output,
                        name=node.output[0]+'_Add')
                    
                    nodes_to_add.extend([matmul_node, add_node])
                    nodes_to_remove.extend([node])

        self.model.add_nodes(nodes_to_add)
        self.model.remove_nodes(nodes_to_remove)
     

    def remove_fake_quantized_nodes(self):
        '''
            Detect and remove the quantize/dequantizelinear node pairs(fake quantized nodes in Quantization-Aware training) 
            and reconnect and update the nodes.
        '''
        nodes_to_remove = []
        initializers_to_remove = []
  
        for curr_node in self.model.nodes():
            if curr_node.op_type == 'QuantizeLinear':
                next_node,prev_node,succ_node = None,None,None
                for child_node in self.model.get_children(curr_node):
                    if child_node.op_type == 'DequantizeLinear':
                            next_node = child_node
                if next_node is None:
                    raise ValueError("Remove fake-quantized node pair Error: DequantizeLinear node is not found for {}."
                                    .format(curr_node.name))

                prev_node = self.model.get_parent(curr_node,0)
                if prev_node is None:
                    raise ValueError("Remove fake-quantized node pair Error: Parent node is not found for {}." 
                                    .format(curr_node.name))
                
                succ_nodes = self.model.get_children(next_node)
                if len(succ_nodes) == 0:
                    raise ValueError("Remove fake-quantized node pair Error: No successive nodes found for {}." 
                                    .format(next_node.name))

                scale_tensor_name = curr_node.input[1]
                zp_tensor_name = curr_node.input[2]
                initializer_scale = _find_by_name(scale_tensor_name,self.model.initializer())
                initializer_zp = _find_by_name(zp_tensor_name,self.model.initializer())
                zp_and_scale = [onnx.numpy_helper.to_array(initializer_zp),onnx.numpy_helper.to_array(initializer_scale)]

                #connect the previous and successive node input and output
                for succ_node in succ_nodes:
                    succ_idx = _get_elem_index(next_node.output[0],succ_node.input)
                    if succ_idx != -1:
                        succ_node.input[succ_idx] = curr_node.input[0]
                    else:
                        raise ValueError("Remove fake-quantized node pair Error: Connection failed. No matched successive node input found for {}."
                                        .format(next_node.name))

                param_name = curr_node.input[0]
                if self.quantization_params is None:
                    self.quantization_params = {}
                self.quantization_params[param_name] = zp_and_scale
                
                #remove fake-quantized nodes
                nodes_to_remove.extend([curr_node])
                nodes_to_remove.extend([next_node])

                #remove unused initializers in graph
                initializers_to_remove.extend([initializer_scale])
                initializers_to_remove.extend([initializer_zp])

        self.model.remove_nodes(nodes_to_remove)
        self.model.remove_initializers(initializers_to_remove)

        return self.model.model


    def quantize_model(self):

        self.replace_gemm_with_matmul()
    
        self.remove_fake_quantized_nodes()
       
        for node in self.model.nodes():
            # if a list of ops to be quantized is provided then only quantize those ops
            if self.nodes_to_quantize is not None and node.name not in self.nodes_to_quantize:
                op_quantizer = CreateDefaultOpQuantizer(self, node)
            elif self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
                op_quantizer = CreateDefaultOpQuantizer(self, node)
            else:
                op_quantizer = CreateOpQuantizer(self, node)
            op_quantizer.quantize()

        self._dequantize_outputs()

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph().ClearField('node')
        self.model.graph().node.extend(self.new_nodes)

        # Remove weights which are already quantized from graph.
        self._remove_quantized_weights()

        # update opset.
        opset_info = next(
            (opset for opset in self.model.opset_import() if opset.domain == '' or opset.domain == onnx_domain), None)
        if opset_info is not None:
            self.model.opset_import().remove(opset_info)
        self.model.opset_import().extend([onnx.helper.make_opsetid(onnx_domain, onnx_op_set_version)])

        return self.model.model

    def find_weight_data(self, initializer):
        '''
            :param initializer: TensorProto initializer object from a graph
            :return: a list of initialized data in a given initializer object
        '''
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer)
        else:
            raise ValueError('Only float type quantization is supported. Weights {} is {}. '.format(
                initializer.name, type_to_name[initializer.data_type]))
        return weights

    def _is_valid_quantize_value(self, value_name):
        if value_name in self.value_infos:
            value_info = self.value_infos[value_name]
            return value_info.type.HasField(
                'tensor_type') and value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT
        return self._is_valid_initializer_value(value_name)

    def _is_valid_initializer_value(self, value_name):
        weight = _find_by_name(value_name, self.model.initializer())
        return weight is not None and weight.data_type == onnx_proto.TensorProto.FLOAT

    def _is_valid_quantize_weight(self, weight_name):
        weight = _find_by_name(weight_name, self.model.initializer())
        return weight is not None and weight.data_type == onnx_proto.TensorProto.FLOAT

    def _remove_quantized_weights(self):
        ''' Remove the weights which are already quantized from graph initializer list.
            This function assumes that after quantization, all nodes that previously use a weight:
                - use output from DequantizeLinear as input if they do not support quantization.
                - use quantized weight if they support quantization.
        '''
        for weight in self._quantized_weights:
            # Remove existing weight initializer
            self.model.initializer().remove(weight.initializer)

            # Removing input weight to a convolution
            try:
                weight_input = next(val for val in self.model.graph().input if val.name == weight.name)
                self.model.graph().input.remove(weight_input)
            except StopIteration:
                if self.model.ir_version() < 4:
                    print("Warning: invalid weight name {} found in the graph (not a graph input)".format(weight.name))

    def _update_graph(self, weight):
        '''
            Given a weight object, update the graph by doing the following:
             - remove old initializer, update new initializers for quantized weight, zero point, and scale
             - remove old weight input, update with new inputs for quantized weight, zero point, and scale
            This function does NOT update the nodes in the graph, just initializers and inputs
        '''
        quantized_value = self.quantized_value_map[weight.name]
        assert (quantized_value is not None)
        packed_weight_name = quantized_value.q_name
        scale_name = quantized_value.scale_name
        zero_point_name = quantized_value.zp_name

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(weight.quantized_data,
                                           dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight.qType]).reshape(
                                               weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else:  # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape,
                                                    weight.scales)
        zero_initializer = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_scale_shape,
                                                   weight.zero_points)

        self.model.initializer().extend([packed_weight_initializer, scale_initializer, zero_initializer])

        self._quantized_weights.append(weight)


    def _get_quantized_weight(self, initializer, qType):
        '''
            :param initializer: TensorProto initializer
            :param qType: type to quantize to
            :return: Weight class with quantization information
        '''
        weights_data = self.find_weight_data(initializer)
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(weights_data.flatten().tolist(),
                                                                              _get_qrange_for_qType(qType), qType)
        weight = QuantizedInitializer(initializer.name,
                                      initializer, [rmin], [rmax], [zero_point], [scale],
                                      weights_data,
                                      quantized_weights_data,
                                      axis=None,
                                      qType=qType)

        # Log entry for this quantized weight
        assert (weight.name not in self.quantized_value_map)
        quantized_value = QuantizedValue(weight.name, weight.name + "_quantized", weight.name + "_scale",
                                         weight.name + "_zero_point", QuantizedValueType.Initializer, None, qType)
        self.quantized_value_map[weight.name] = quantized_value

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
            per_channel_data = np_data[i, :, :, :].flatten()
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data(
                per_channel_data.flatten().tolist(), _get_qrange_for_qType(qType), qType)
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)
        channel_index = 0  # (M x C/group x kH x kW)
        # combine per_channel_data into one
        reshape_dims = list(initializer.dims)  # deep copy
        reshape_dims[channel_index] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), axis=0)

        weight = QuantizedInitializer(initializer.name, initializer, rmin_list, rmax_list, zero_point_list, scale_list,
                                      weights,
                                      quantized_weights.flatten().tolist(), channel_index, qType)

        # Make entry for this quantized weight
        assert (weight.name not in self.quantized_value_map)
        quantized_value = QuantizedValue(weight.name, weight.name + "_quantized", weight.name + "_scale",
                                         weight.name + "_zero_point", QuantizedValueType.Initializer, None, qType)
        self.quantized_value_map[weight.name] = quantized_value

        return weight

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name, nodes_list)

        return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list)

    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to int8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name], [reduce_min_name + ":0"],
                                                reduce_min_name,
                                                keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name], [reduce_max_name + ":0"],
                                                reduce_max_name,
                                                keepdims=0)
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node("Abs", [reduce_min_node.output[0]], [reduce_min_abs_name + ":0"],
                                                    reduce_min_abs_name)
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node("Abs", [reduce_max_node.output[0]], [reduce_max_abs_name + ":0"],
                                                    reduce_max_abs_name)
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node("Max", [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
                                             [abs_max_name + ":0"], abs_max_name)
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(self.fixed_qrange_int8_name,onnx_proto.TensorProto.FLOAT,[],[_get_qrange_for_qType(qType) / 2.0])
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [abs_max_node.output[0], self.fixed_qrange_int8_name],
                                               [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType,[],[0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name], [reduce_min_name + ":0"],
                                                reduce_min_name,
                                                keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name], [reduce_max_name + ":0"],
                                                reduce_max_name,
                                                keepdims=0)
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(self.fixed_qrange_uint8_name,onnx_proto.TensorProto.FLOAT,[],[_get_qrange_for_qType(qType)])
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name,onnx_proto.TensorProto.FLOAT,[],[0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node("Sub", [reduce_max_node.output[0], reduce_min_node.output[0]],
                                               [scale_sub_name + ":0"], scale_sub_name)
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
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
        zp_div_node = onnx.helper.make_node("Div", [zp_sub_node.output[0], input_scale_name], [zp_div_name + ":0"],
                                            zp_div_name)
        nodes_list.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        nodes_list.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_quantization_params(self, param_name):
        '''
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.
            parameter param_name: Name of the quantization parameter.
            return: result, scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""

        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. "
                             "Specified values for output {}: {}".format(param_name, params))
      
        zero_point_values = [params[0].item()]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1].item()]
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name,zero_point_type,zero_point_shape,zero_point_values)   
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name,onnx_proto.TensorProto.FLOAT,scale_shape,scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(self, node, input_index, qType):
        '''
        Given an input for a node (which is not a initializer), this function
            - add nodes to compute zero point and scale for this input if they don't exist.
            - add new QuantizeLinear node to quantize the input.
            parameter node: node being quantized in NodeProto format.
            parameter input_index: index of input in node.input.
            parameter qType: type to quantize to.
            return: List of newly created nodes in NodeProto format.
        '''
        input_name = node.input[input_index]
        output_name = input_name + "_quantized"

        data_found, scale_name, zp_name, _, _ = \
            self._get_quantization_params(input_name)

        if self.static:
            if data_found == False:
                raise ValueError(
                    "Quantization parameters are not specified for param {}."
                    "In static mode quantization params for inputs and outputs of nodes to be quantized are required.".
                    format(input_name))

            qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], [output_name],
                                                 input_name + "_QuantizeLinear")
            return [qlinear_node]

        else:
            if data_found == True:
                qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], [output_name],
                                                     input_name + "_QuantizeLinear")
                return [qlinear_node]
            else:
                # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
                if self.fuse_dynamic_quant and qType == onnx_proto.TensorProto.UINT8:
                    scale_name = input_name + "_scale"
                    zeropoint_name = input_name + "_zero_point"
                    qlinear_node = onnx.helper.make_node("DynamicQuantizeLinear", [input_name],
                                                         [output_name, scale_name, zeropoint_name],
                                                         input_name + "_QuantizeLinear")
                    return [qlinear_node]

                else:
                    nodes = []
                    scale_name, zp_name, scale_shape, zp_shape = \
                        self._get_dynamic_input_quantization_params(
                            input_name, nodes, qType)
                    qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name],
                                                         [output_name], input_name + "_QuantizeLinear")

                    return nodes + [qlinear_node]

    def _get_bias_add_nodes(self, nodes, node, last_output, quantized_bias_name):
        '''
        Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node
            parameter nodes: new nodes would be appended into nodes
            parameter node: current node (Conv)
            parameter last_output: output of previous node (input to bias add)
            return: the name of output
        '''
        # Add an Add operation for bias
        # Add reshape for correct broadcase
        reshape_input = [quantized_bias_name]

        # Add tensors for the shape to be reshaped to
        init_shape = onnx.helper.make_tensor("reshape_shape",onnx_proto.TensorProto.INT64,[4],[1, -1, 1, 1])
        self.model.add_initializer(init_shape)

        reshape_input.append('reshape_shape')
        reshape_op_output = node.output[0] + "_reshape"
        reshape_node = onnx.helper.make_node("Reshape", reshape_input, [reshape_op_output],
                                             quantized_bias_name + "reshape")
        nodes.append(reshape_node)

        bias_add_input = [last_output]
        bias_add_input.append(reshape_op_output)
        add_node_output = node.output[0] + "_bias_add"
        add_node = onnx.helper.make_node("Add", bias_add_input, [add_node_output], quantized_bias_name + "bias_add")
        nodes.append(add_node)
        return add_node_output

    def _update_nodes_using_weight(self):
        '''Find all nodes using a weight that do not support quantization and
        add a DequantizeLinear node before those nodes. This includes all nodes except Conv, MatMul.
            parameter weight: Weight object
            parameter new_nodes_list: List of new nodes created before processing current node.
            return: List of new nodes created.
        '''
        nodes_list = []
        for weight in self._quantized_weights:
            nodes_using_weight = self.model.find_nodes_by_initializer(self.new_nodes, weight.initializer)

            dequantize_linear_name = weight.name + "_DequantizeLinear"
            output_name = weight.name + "_dequantized"

        # Check if DequantizeLinear node needs to be added to graph.
        if len(nodes_using_weight) != 0 and \
                self.model.find_node_by_name(dequantize_linear_name,self.new_nodes,self.model.graph()) is None:
            inputs = [weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point"]
            node = onnx.helper.make_node("DequantizeLinear", inputs, [output_name], dequantize_linear_name)
            nodes_list.append(node)

        # Update unsupported nodes to take dequantized weight as input.
        for node in nodes_using_weight:
            for i, node_input in enumerate(node.input):
                if node_input == weight.name:
                    node.input[i] = output_name

        self.new_nodes += nodes_list

    def _dynamic_quantize_bias(self, input_name, weight_scale_name, bias_name, quantized_bias_name, new_node_list):
        '''
        Adds series of nodes required to quantize the bias dynamically.
            parameter input_name: Input name
            parameter weight_scale_name: Weight scale.
            parameter bias_scale_name: Bias to quantize.
            parameter quantied_bias_name: Output name to use for quantized bias.
        '''
        qType = onnx_proto.TensorProto.INT32

        input_scale_name = input_name + "_scale"
        bias_scale_node = onnx.helper.make_node("Mul", [input_scale_name, weight_scale_name], [bias_name + "_scale"],
                                                bias_name + "_scale_node")
        new_node_list.append(bias_scale_node)

        quantize_bias_node = onnx.helper.make_node("Div", [bias_name, bias_scale_node.output[0]],
                                                   [bias_name + "_tmp_quant:0"], bias_name + "_tmp_qaunt")
        new_node_list.append(quantize_bias_node)

        bias_rounded_node = onnx.helper.make_node("Floor", quantize_bias_node.output, [bias_name + "_quant_rounded:0"],
                                                  bias_name + "_quant_rounded")
        new_node_list.append(bias_rounded_node)

        bias_cast_node = onnx.helper.make_node("Cast",
                                               bias_rounded_node.output, [quantized_bias_name],
                                               quantized_bias_name + "_node",
                                               to=qType)
        new_node_list.append(bias_cast_node)

        return

    def quantize_bias(self, node, new_node_list):
        '''
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        '''

        # get scale for weight
        weight_scale_name = self.quantized_value_map[node.input[1]].scale_name
        weight_initializer = _find_by_name(weight_scale_name, self.model.initializer())
        weight_scale = self.find_weight_data(weight_initializer)

        # get bias
        bias_name = node.input[2]
        bias_initializer = _find_by_name(bias_name, self.model.initializer())
        bias_data = self.find_weight_data(bias_initializer)
        quantized_bias_name = bias_name + "_quantized"

        # input scale is not provided and this input is dynamically quantized so it is not pre-computed at this point
        # so resort to dynamic quantization for bias
        if self.quantization_params is None or node.input[0] not in self.quantization_params and node.input[
                0] not in self.quantized_value_map:
            self._dynamic_quantize_bias(node.input[0], weight_scale_name, bias_name, quantized_bias_name, new_node_list)
        else:
            # get scale for input
            if node.input[0] in self.quantized_value_map:
                input_scale_name = self.quantized_value_map[node.input[0]].scale_name
            elif node.input[0] in self.quantization_params:
                _, input_scale_name, _, _, _ = self._get_quantization_params(node.input[0])
            else:
                raise ValueError("Expected {} to be in quantized value map for static quantization".format(
                    node.input[0]))

            inputscale_initializer = _find_by_name(input_scale_name, self.model.initializer())
            input_scale = self.find_weight_data(inputscale_initializer)

            # calcuate scale for bias

            bias_scale = input_scale * weight_scale

            # quantize bias
            quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

            # update bias initializer
            bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
            packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
            self.model.initializer().extend([packed_bias_initializer])

            # log entries for this quantized bias value
            quantized_bias_entry = QuantizedInitializer(bias_name,
                                                        bias_initializer, [0], [0], [0], [bias_scale],
                                                        bias_data,
                                                        quantized_data,                                                                     
                                                        qType=onnx_proto.TensorProto.INT32)
            self._quantized_weights.append(quantized_bias_entry)

            assert (bias_name not in self.quantized_value_map)
            quantized_value = QuantizedValue(bias_name, quantized_bias_name, "", "", QuantizedValueType.Initializer,
                                             None, onnx_proto.TensorProto.INT32)
            self.quantized_value_map[bias_name] = quantized_value

        return quantized_bias_name

    def _quantize_inputs(self, node, indices):
        '''
        Given a node, this function quantizes the inputs as follows:
            - If input is an initializer, quantize the initializer data, replace old initializer
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

        quantized_input_names = []
        zero_point_names = []
        scale_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                qType = self.weight_qType if quantized_value.value_type == QuantizedValueType.Initializer else self.input_qType
                if quantized_value.qType != qType:
                    raise ValueError(
                        "{} is being used by multiple nodes which are being quantized to different types. "
                        "This is not suported.", node_input)

                quantized_input_names.append(quantized_value.q_name)
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                continue

            # Quantize the input
            initializer = _find_by_name(node_input, self.model.initializer())
            if initializer is not None:
                if node.op_type == "Conv":
                    weight = self._get_quantized_weight_convolution(initializer, self.weight_qType)
                else:
                    weight = self._get_quantized_weight(initializer, self.weight_qType)

                # Update graph
                self._update_graph(weight)

                quantized_input_names.append(weight.name + "_quantized")
                zero_point_names.append(weight.name + "_zero_point")
                scale_names.append(weight.name + "_scale")
            else:
                # Add QuantizeLinear node.
                qlinear_node = self.model.find_node_by_name(node_input + "_QuantizeLinear",self.new_nodes,self.model.graph())
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, self.input_qType)
                    nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                if qlinear_node.op_type == "QuantizeLinear":
                    quantized_input_names.extend(qlinear_node.output)
                    scale_names.append(qlinear_node.input[1])
                    zero_point_names.append(qlinear_node.input[2])
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])

        return (quantized_input_names, zero_point_names, scale_names, nodes)

    def _dequantize_value(self, value_name):
        '''
        Given a value (input/output) which is quantized, add a DequantizeLinear node to dequantize
        it back to float32
            parameter value_name: value to dequantize
            parameter new_nodes_list: List of new nodes created before processing current node
            return: None if there is already a DequantizeLinear node that dequantizes it
                    A DequantizeLinear node otherwise
        '''
        if value_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[value_name]
            # Add DequantizeLinear Node for this input
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_node = self.model.find_node_by_name(dqlinear_name,self.new_nodes,self.model.graph())
            if dqlinear_node is None:
                dqlinear_inputs = [quantized_value.q_name, quantized_value.scale_name, quantized_value.zp_name]
                dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name],
                                                        dqlinear_name)
                return dequantize_node
            else:
                # DQ op is already present, assert it's output matches the input of current node
                assert (value_name == dqlinear_node.output[0])
        return None

    def _dequantize_outputs(self):
        '''
        Dequantize output if it is quantized
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        '''
  
        for output in self.model.graph().output:
            dequantize_node = self._dequantize_value(output.name)
            if dequantize_node is not None:
                self.new_nodes.append(dequantize_node)


def quantize(model_path,
             per_channel=False,
             nbits=8,
             quantization_mode=QuantizationMode.IntegerOps,
             static=False,
             force_fusions=False,
             symmetric_activation=False,
             symmetric_weight=False,
             quantization_params=None,
             nodes_to_quantize=None,
             nodes_to_exclude=None):
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
              specified through quantization_params.
        False: The inputs/activations are quantized using dynamic scale and zero point values
               computed while running the model.
    :param force_fusions:
        True: Fuses nodes added for dynamic quantization
        False: No fusion is applied for nodes which are added for dynamic quantization.
        Should be only used in cases where backends want to apply special fusion routines
    :param symmetric_activation:
        True: activations are quantized into signed integers.
        False: activations are quantized into unsigned integers.
    :param symmetric_weight:
        True: weights are quantized into signed integers.
        False: weights are quantized into unsigned integers.
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :return: ModelProto with quantization
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    '''
    if nbits == 8:
        input_qType = onnx_proto.TensorProto.INT8 if symmetric_activation else onnx_proto.TensorProto.UINT8
        weight_qType = onnx_proto.TensorProto.INT8 if symmetric_weight else onnx_proto.TensorProto.UINT8
        mode = quantization_mode

        #optimize the original model
        optimized_model = optimize_model(Path(model_path))
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(optimized_model)

        #check opset version of the original model
        fuse_dynamic_quant = check_opset_version(onnx.load(model_path), force_fusions)
        
        #apply shape inference to the ModelProto and get value informations
        inferred_model = shape_inference.infer_shapes(copy_model)
        value_infos = {vi.name: vi for vi in inferred_model.graph.value_info}
        
        #create ONNXModel and ONNXQuantizer
        onnx_model = ONNXModel(inferred_model)
        quantizer = ONNXQuantizer(onnx_model, value_infos, per_channel, mode, static, fuse_dynamic_quant, weight_qType, input_qType,
                                  quantization_params, nodes_to_quantize, nodes_to_exclude)
        quantizer.quantize_model()
        quantizer.model.producer_name = __producer__
        quantizer.model.producer_version = __version__
        return quantizer.model.model
    else:
        raise ValueError('Only 8 bit quantization is currently supported')