# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import numpy as np
import os
import argparse
from onnx import numpy_helper
from onnx import helper
from onnx import utils
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import shape_inference
from argparse import ArgumentParser

def graph_topological_sort(graph):
    deps_count = [0] * len(graph.node)  # dependency count of each node
    deps_to_nodes = {}  # input to node indice
    sorted_nodes = []  # initialize sorted_nodes
    for node_idx, node in enumerate(graph.node):
        # CANNOT use len(node.input) directly because input can be optional
        deps_count[node_idx] = sum(1 for _ in node.input if _)
        if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
            sorted_nodes.append(graph.node[node_idx])
            continue

        for input_name in node.input:
            if input_name not in deps_to_nodes:
                deps_to_nodes[input_name] = [node_idx]
            else:
                deps_to_nodes[input_name].append(node_idx)

    # Note: this logic only applies to top level graph since a sub graph could use intializer from parent graph
    initializer_names = [init.name for init in graph.initializer]
    graph_input_names = [input.name for input in graph.input]
    input_names = initializer_names + graph_input_names
    input_names.sort()
    prev_input_name = None
    for input_name in input_names:
        if prev_input_name == input_name:
            continue

        prev_input_name = input_name
        if input_name in deps_to_nodes:
            for node_idx in deps_to_nodes[input_name]:
                deps_count[node_idx] = deps_count[node_idx] - 1
                if deps_count[node_idx] == 0:
                    sorted_nodes.append(graph.node[node_idx])

    start = 0
    end = len(sorted_nodes)

    while start < end:
        for output in sorted_nodes[start].output:
            if output in deps_to_nodes:
                for node_idx in deps_to_nodes[output]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])
                        end = end + 1
        start = start + 1

    assert (end == len(graph.node)), "Graph is not a DAG"
    graph.ClearField('node')
    graph.node.extend(sorted_nodes)

def get_value_from_qnn_code_line(line_content):
    value = line_content.split("=")
    assert (len(value) == 2), "Error: Wrong format encountered."
    value = value[1].split(",")
    assert (len(value) == 2), "Error: Wrong format encountered."
    return value[0].replace(" ", "").replace("\"", "")

class qnn_tensor_struct:
    def __init__(self):
        self.name = ""
        self.qnn_data_type = ""
        self.onnx_data_type = TensorProto.FLOAT
        self.dim_alias = ""
        self.dim = []

def qnn_data_type_str_to_onnx_data_type(qnn_data_type_str):
    if qnn_data_type_str == "QNN_DATATYPE_UFIXED_POINT_8" or qnn_data_type_str == "QNN_DATATYPE_UINT_8":
        return TensorProto.UINT8
    elif qnn_data_type_str == "QNN_DATATYPE_UFIXED_POINT_16" or qnn_data_type_str == "QNN_DATATYPE_UINT_16":
        return TensorProto.UINT16
    elif qnn_data_type_str == "QNN_DATATYPE_UFIXED_POINT_32" or qnn_data_type_str == "QNN_DATATYPE_UINT_32":
        return TensorProto.UINT32
    elif qnn_data_type_str == "QNN_DATATYPE_UINT_64":
        return TensorProto.UINT64
    elif qnn_data_type_str == "QNN_DATATYPE_UFIXED_POINT_8":
        return TensorProto.UINT8
    elif qnn_data_type_str == "QNN_DATATYPE_FIXED_POINT_8" or qnn_data_type_str == "QNN_DATATYPE_INT_8":
        return TensorProto.INT8
    elif qnn_data_type_str == "QNN_DATATYPE_FIXED_POINT_16" or qnn_data_type_str == "QNN_DATATYPE_INT_16":
        return TensorProto.INT16
    elif qnn_data_type_str == "QNN_DATATYPE_FIXED_POINT_32" or qnn_data_type_str == "QNN_DATATYPE_INT_32":
        return TensorProto.INT32
    elif qnn_data_type_str == "QNN_DATATYPE_INT_64":
        return TensorProto.INT64
    elif qnn_data_type_str == "QNN_DATATYPE_FLOAT_16":
        return TensorProto.FLOAT16
    elif qnn_data_type_str == "QNN_DATATYPE_FLOAT_32":
        return TensorProto.FLOAT
    elif qnn_data_type_str == "QNN_DATATYPE_BOOL_8 ":
        return TensorProto.BOOL

def parse_dims(line_content):
    value = line_content.split("=")
    assert (len(value) == 2), "Error: Wrong format encountered."
    value = value[1].replace(" ", "").split(";")
    assert (len(value) == 2), "Error: Wrong format encountered."
    shape = value[0].replace("{", "").replace("}", "").split(",")
    return [eval(item) for item in shape]


def parse_qnn_cpp_file(qnn_cpp_file, qnn_input_output_tensor_dic):
    qnn_tensor_names = []
    qnn_tensor_types = []
    qnn_tensor_data_types = []
    qnn_tensor_dim_alias = []
    qnn_tensor_item_count = 0
    with open (qnn_cpp_file) as cpp_file:
        for line in cpp_file:
            if "(Qnn_Tensor_t)" in line:
                qnn_tensor_item_count = 0
            if (".name" in line) and (qnn_tensor_item_count < 4):
                name = get_value_from_qnn_code_line(line)
                qnn_tensor_names.append(name)
                qnn_tensor_item_count += 1
            if (".type" in line) and (qnn_tensor_item_count < 4):
                qnn_tensor_types.append(get_value_from_qnn_code_line(line))
                qnn_tensor_item_count += 1
            if (".dataType" in line) and (qnn_tensor_item_count < 4):
                qnn_tensor_data_types.append(get_value_from_qnn_code_line(line))
                qnn_tensor_item_count += 1
            if (".dimensions" in line) and (qnn_tensor_item_count < 4):
                qnn_tensor_dim_alias.append(get_value_from_qnn_code_line(line))
                qnn_tensor_item_count += 1
    assert (len(qnn_tensor_names) == len(qnn_tensor_types) and len(qnn_tensor_types) == len(qnn_tensor_data_types) and len(qnn_tensor_data_types) == len(qnn_tensor_dim_alias)), "Error: Something wrong!"

    for i in range(len(qnn_tensor_types)):
        # only care about the graph input and output tensors
        if qnn_tensor_types[i] == "QNN_TENSOR_TYPE_APP_WRITE" or qnn_tensor_types[i] == "QNN_TENSOR_TYPE_APP_READ":
            qnn_tensor = qnn_tensor_struct()
            qnn_tensor.name = qnn_tensor_names[i]
            qnn_tensor.qnn_data_type = qnn_tensor_data_types[i]
            qnn_tensor.onnx_data_type = qnn_data_type_str_to_onnx_data_type(qnn_tensor.qnn_data_type)
            qnn_tensor.dim_alias = qnn_tensor_dim_alias[i]
            qnn_input_output_tensor_dic[qnn_tensor.name] = qnn_tensor
    
    # Go through the cpp file again get the shape dims by the give dim alias
    with open (qnn_cpp_file) as cpp_file:
        for line in cpp_file:
            for tensor in qnn_input_output_tensor_dic.values():
                if tensor.dim_alias + "[]" in line:
                    tensor.dim = parse_dims(line)

def compare_onnx_shape_with_qnn_shape(onnx_dims, qnn_dims):
    assert (len(onnx_dims) == len(qnn_dims)), "Onnx shape and Qnn shape has different rank."
    for i in range(len(onnx_dims)):
        if onnx_dims[i].dim_value != qnn_dims[i]:
            return False

    return True

def gen_to_channel_first_perm(rank):
    assert (rank > 2), "Shape rank should >2 for the Transpose node."
    perm = []
    perm.append(0)
    perm.append(rank-1)
    for i in range(1, rank-1):
        perm.append(i)

    return perm
    
def gen_to_channel_last_perm(rank):
    assert (rank > 2), "Shape rank should >2 for the Transpose node."
    perm = []
    perm.append(0)
    for i in range(2, rank):
        perm.append(i)
    perm.append(1)

    return perm


# Onnxruntime QNN EP can support context binary file generated by QNN tool chain. However QNN generated context binary file
# uses channel last and 8 bits or 16 bits for input and output.
# This script get the QNN model input & output information from QNN converted model.cpp file,
# and insert Cast, Transpose nodes to Onnx model if required
def main():
    parser = ArgumentParser('Insert Cast, Transpose nodes into Onnx model to make it aligned with QNN generated context binary.')
    parser.add_argument('-m', '--onnx_model', help='Required. Path to Onnx model file.', required=True, type=str)
    parser.add_argument('-q', '--qnn_cpp', help='Required. Path to Qnn converted model.cpp file.', required=True, type=str)
    args = parser.parse_args()

    # Parse Qnn model.cpp file to get the graph input output information
    qnn_input_output_tensor_dic = {}
    parse_qnn_cpp_file(args.qnn_cpp, qnn_input_output_tensor_dic)

    model = onnx.load(args.onnx_model)
    
    nodes_to_add = []
    # Tranch the tensor name change to update the consumer nodes
    graph_input_output_name_dic = {}
    for graph_input in model.graph.input:
        if graph_input.name in qnn_input_output_tensor_dic.keys():
            input_name_fater_node_insert = graph_input.name
            # Insert Cast node if Onnx input and Qnn input has idfferent data type
            if graph_input.type.tensor_type.elem_type != qnn_input_output_tensor_dic[graph_input.name].onnx_data_type:
                # Insert Cast node
                cast_input_name = input_name_fater_node_insert
                cast_output_name = cast_input_name + '_qnn_cast'
                input_cast_node = helper.make_node('Cast',
                                                   name= cast_output_name,
                                                   inputs=[cast_input_name],
                                                   outputs=[cast_output_name],
                                                   to=graph_input.type.tensor_type.elem_type)
                # Change input data type to Qnn input data type
                graph_input.type.tensor_type.elem_type = qnn_input_output_tensor_dic[graph_input.name].onnx_data_type
                nodes_to_add.extend([input_cast_node])
                input_name_fater_node_insert = cast_output_name
                graph_input_output_name_dic[graph_input.name] = cast_output_name

            if not compare_onnx_shape_with_qnn_shape(graph_input.type.tensor_type.shape.dim, qnn_input_output_tensor_dic[graph_input.name].dim):
                # Add Transpose node (channel last to channel first)
                transpose_perm = gen_to_channel_first_perm(len(graph_input.type.tensor_type.shape.dim))
                transpose_input_name = input_name_fater_node_insert
                transpose_output_name = transpose_input_name + '_qnn_trans'
                input_transpose_node = helper.make_node('Transpose',
                                                        name= transpose_output_name,
                                                        inputs=[transpose_input_name],
                                                        outputs=[transpose_output_name],
                                                        perm=transpose_perm)
                nodes_to_add.extend([input_transpose_node])
                graph_input_output_name_dic[graph_input.name] = transpose_output_name
                
                # Change input shape to Qnn input shape
                for i in range(len(graph_input.type.tensor_type.shape.dim)):
                    graph_input.type.tensor_type.shape.dim[i].dim_value = qnn_input_output_tensor_dic[graph_input.name].dim[i]

    for graph_output in model.graph.output:
        if graph_output.name in qnn_input_output_tensor_dic.keys():
            output_name_fater_node_insert = graph_output.name
            # Insert Cast node if Onnx input and Qnn input has idfferent data type
            if graph_output.type.tensor_type.elem_type != qnn_input_output_tensor_dic[graph_output.name].onnx_data_type:
                # Insert Cast node
                cast_output_name = output_name_fater_node_insert
                cast_input_name = cast_output_name + "_qnn_cast"
                output_cast_node = helper.make_node('Cast',
                                                   name= cast_input_name,
                                                   inputs=[cast_input_name],
                                                   outputs=[cast_output_name],
                                                   to=qnn_input_output_tensor_dic[graph_output.name].onnx_data_type)
                # Change output data type to Onn output data type
                graph_output.type.tensor_type.elem_type = qnn_input_output_tensor_dic[graph_output.name].onnx_data_type
                nodes_to_add.extend([output_cast_node])
                output_name_fater_node_insert = cast_input_name
                graph_input_output_name_dic[graph_output.name] = cast_input_name

            if not compare_onnx_shape_with_qnn_shape(graph_output.type.tensor_type.shape.dim, qnn_input_output_tensor_dic[graph_output.name].dim):
                # Add Transpose node (channel first to channel last)
                transpose_perm = gen_to_channel_last_perm(len(graph_output.type.tensor_type.shape.dim))
                transpose_output_name = output_name_fater_node_insert
                transpose_input_name = transpose_output_name + '_qnn_trans'
                output_transpose_node = helper.make_node('Transpose',
                                                        name= transpose_input_name,
                                                        inputs=[transpose_input_name],
                                                        outputs=[transpose_output_name],
                                                        perm=transpose_perm)
                nodes_to_add.extend([output_transpose_node])
                graph_input_output_name_dic[graph_output.name] = transpose_input_name
                
                # Change output shape to Qnn output shape
                for i in range(len(graph_input.type.tensor_type.shape.dim)):
                    graph_output.type.tensor_type.shape.dim[i].dim_value = qnn_input_output_tensor_dic[graph_output.name].dim[i]

                

    for node in model.graph.node:
        node_input_index = 0
        for node_input in node.input:
            # update consumer node for graph inputs to connect to inserted node
            if node_input in graph_input_output_name_dic.keys():
                node.input[node_input_index] = graph_input_output_name_dic[node_input]
            node_input_index += 1

        node_output_index = 0
        for node_output in node.output:
            # update producer node for graph outputs to connect to inserted node
            if node_output in graph_input_output_name_dic.keys():
                node.output[node_output_index] = graph_input_output_name_dic[node_output]
            node_output_index += 1


    model.graph.node.extend(nodes_to_add)
    graph_topological_sort(model.graph)

    onnx.save(model, args.onnx_model.replace(".onnx", "_add_trans.onnx"))

if __name__ == '__main__':
    main()
