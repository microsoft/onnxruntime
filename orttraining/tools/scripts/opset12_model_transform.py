# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This converter is an internal util to upgrade existing bert/gpt-2 models,
# which were previously transformed/optimized from orginal model, to Opset 12
# version as well as replacing deprecated node, i.e., TrainableDropout with
# the "Dropout" node matching the Opset 12 Spec. Typically, a model to
# be run by this scripts would have "_optimized" substring in its model name,
# and the graph should have one or more "TrainableDropout" nodes in its graph.
# Example usage:
#   python opset12_model_transform.py bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx
# Output:
#   bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12.onnx

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference  # noqa: F401

if len(sys.argv) < 2:
    print("Please give model path...")
    exit(1)

input_model_name = sys.argv[1]
output_model_name = input_model_name[:-5] + "_opset12.onnx"
model = onnx.load(input_model_name)


# for a given node input, look thru the graph nodes and find the node
# whose output is matching the input
def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)  # noqa: PERF401
    return result[0] if len(result) == 1 else None


def get_node_index(model, node):
    for i, graph_node in enumerate(model.graph.node):
        if graph_node == node:
            return i
    return None


def add_const(model, name, output, t_value=None, f_value=None):
    const_node = model.graph.node.add()
    const_node.op_type = "Constant"
    const_node.name = name
    const_node.output.extend([output])
    attr = const_node.attribute.add()
    attr.name = "value"
    if t_value is not None:
        attr.type = 4
        attr.t.CopyFrom(t_value)
    else:
        attr.type = 1
        attr.f = f_value
    return const_node


def process_trainabledropout(model):
    delete_nodes = []
    index = 0
    for node in model.graph.node:
        if node.op_type == "TrainableDropout":
            new_dropout = model.graph.node.add()
            new_dropout.op_type = "Dropout"
            new_dropout.name = "Dropout_%d" % index
            # add seed attribute
            attr = new_dropout.attribute.add()
            attr.name = "seed"
            attr.type = 2
            # find old ratio node
            ratio_node = find_input_node(model, node.input[1])
            assert ratio_node.op_type == "Constant"
            delete_nodes.append(get_node_index(model, ratio_node))
            # make ratio scalar node
            ratio_attr = ratio_node.attribute
            ratio_data = numpy_helper.to_array(ratio_attr[0].t)
            ratio_scalar = ratio_data.astype(np.float32).reshape(())
            ratio_value = numpy_helper.from_array(ratio_scalar, "ratio")
            new_ratio_node = add_const(
                model, "dropout_ratio_node_%d" % index, "dropout_ratio_%d" % index, t_value=ratio_value
            )
            index += 1
            # add training_mode output
            mode_scalar = np.asarray([True]).astype(bool).reshape(())
            mode_value = numpy_helper.from_array(mode_scalar, "training_mode")
            training_mode_node = add_const(
                model, "dropout_training_mode_node_%d" % index, "dropout_training_mode_%d" % index, t_value=mode_value
            )
            index += 1

            new_dropout.input.extend([node.input[0], new_ratio_node.output[0], training_mode_node.output[0]])
            new_dropout.output.extend(node.output)
            delete_nodes.append(get_node_index(model, node))
            index += 1

    delete_nodes.sort(reverse=True)
    for d in delete_nodes:
        del model.graph.node[d]


def align_attention_mask_dim(model):
    for model_input in model.graph.input:
        if model_input.name == "attention_mask":
            model_input.type.tensor_type.shape.dim[0].dim_param = "batch"


# replace TrainableDropout with Dropout
process_trainabledropout(model)
# some gpt-2 models (large ones) still don't have this input corrected
align_attention_mask_dim(model)

# set opset version to 12
model.opset_import[0].version = 12

with open(output_model_name, "wb") as f:
    f.write(model.SerializeToString())

#
# To verify the converted model in case of bert, refer to the code at the end of model_transform.py
#
