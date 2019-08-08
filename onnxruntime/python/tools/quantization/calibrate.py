# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnx
from onnx import helper, TensorProto

# Adding ability to record intermediate graph outputs
selected_types = ["Conv", "MatMul"] # node types to extend

def augment_graph(model):
    '''
    Adds ReduceMin and ReduceMax nodes to all Conv and MatMul nodes in
    model and ensures their outputs are stored as part of the graph output
        parameter model: FP32 ONNX model to quantize
        return: augmented ONNX model
    '''
    added_nodes = []
    added_outputs = []
    for node in model.graph.node:
        if node.op_type in selected_types:
            input_name = node.name
            # Adding ReduceMin nodes
            reduce_min_name = input_name + "_ReduceMin"
            reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name + ":0"],
                            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.name + ":0", TensorProto.FLOAT, ()))
            # Adding ReduceMax nodes
            reduce_max_name = input_name + "_ReduceMax"
            reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name + ":0"],
                            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.name + ":0", TensorProto.FLOAT, ()))
    model.graph.node.extend(added_nodes)
    model.graph.output.extend(added_outputs)
    return model
