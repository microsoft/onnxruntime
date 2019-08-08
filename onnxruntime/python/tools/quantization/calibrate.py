# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnx
from onnx import helper, TensorProto

def augment_graph(model):
    '''
    Adds ReduceMin and ReduceMax nodes to all Conv and MatMul nodes_list in
    model and ensures their outputs are stored as part of the graph output
        parameter model: loaded ONNX model (not yet quantized)
        return: augmented ONNX model
    '''
    added_nodes = []
    added_outputs = []

    for node in model.graph.node:
        if (node.op_type == "Conv") or (node.op_type == "MatMul"):
            input_name = node.name
            # Adding ReduceMin nodes
            reduce_min_name = input_name + "_ReduceMin"
            reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name + ":0"],
                            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
            added_nodes.append(reduce_min_node)
            # Adding ReduceMax nodes
            reduce_max_name = input_name + "_ReduceMax"
            reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name + ":0"],
                            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
            added_nodes.append(reduce_max_node)
    model.graph.node.extend(added_nodes)

    for node in model.graph.node:
        if (node.op_type == "ReduceMin") or (node.op_type == "ReduceMax"):
            intermediate_output = helper.make_tensor_value_info(node.name + ":0", TensorProto.FLOAT, ())
            added_outputs.append(intermediate_output)
    model.graph.output.extend(added_outputs)
    return model
