import sys
import os.path
from onnx import *
import onnx
import numpy as np
import struct

from onnx import helper
from onnx import numpy_helper

# Transpose PostProcess

def find_input_as_initializer(model, arg):
    for initializer in model.graph.initializer:
        if initializer.name == arg:
            return initializer
    return None

def replace_input_arg(model, arg, new_arg):
    for node in model.graph.node:
        i = 0
        while i < len(node.input):
            if node.input[i] == arg:
                node.input[i] = new_arg
            i += 1

def run_postprocess(model):
    # this post pass is not required for pytorch > 1.6
    model = fuse_softmaxNLL_to_softmaxCE(model)

    model = layer_norm_transform(model)
    model = fix_expand_shape(model)
    return model

def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)
    return result[0] if len(result)== 1 else None

def find_output_node(model, arg):
    result = []
    for node in model.graph.node:
        for input in node.input:
            if input == arg:
                result.append(node)
    return result[0] if len(result) == 1 else result

def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break;
        i += 1
    return i if i < len(model.graph.node) else None;

def find_weight_index(model, name):
    index = 0
    for w in model.graph.initializer:
        if w.name == name:
            return index
        index += 1
    return None

def fix_transpose(model):
    transpose = []
    for node in model.graph.node:
        if node.op_type == 'Transpose':
            weight = find_input_as_initializer(model, node.input[0])
            if weight is not None:
                result = []
                for n in model.graph.node:
                    for input in n.input:
                        if input == weight.name:
                            result.append(n)
                if len(result) > 1:
                    continue
                perm = node.attribute[0]
                assert perm.name == 'perm'
                perm = perm.ints
                if len(perm) == 2 and perm[0] == 1 and perm[1] == 0: ##
                    weight_array = numpy_helper.to_array(weight)
                    if len(weight_array.shape) == 2: ##
                        transpose.append((get_node_index(model, node), weight))

    for t in transpose:
        node = model.graph.node[t[0]]
        weight = numpy_helper.to_array(t[1])
        weight = weight.transpose(perm)
        new_weight = numpy_helper.from_array(weight, "%s_transposed" % t[1].name)
        model.graph.initializer.extend([new_weight])
        replace_input_arg(model, node.output[0], new_weight.name)

    transpose.sort(reverse=True)
    for t in transpose:
        del model.graph.node[t[0]]

    old_ws = []
    for t in transpose:
        out_node = find_output_node(model, t[1].name) 
        if out_node is None or len(out_node) == 0: ##
            old_ws.append(find_weight_index(model, t[1].name))
    old_ws.sort(reverse=True)
    for w_i in old_ws:
        del model.graph.initializer[w_i]
    return model

# Expand Shape PostProcess

def fix_expand_shape(model):
    expand_nodes = [n for n in model.graph.node if n.op_type == 'Expand']
    model_inputs_names = [i.name for i in model.graph.input]

    for expand_node in expand_nodes:
        shape = find_input_node(model, expand_node.input[1])
        if shape.op_type == 'Shape':
            shape_input_name = shape.input[0]
            if shape_input_name in model_inputs_names:
                index = model_inputs_names.index(shape_input_name)
                expand_out = model.graph.value_info.add()
                expand_out.name = expand_node.output[0]
                expand_out.type.CopyFrom(model.graph.input[index].type)
    return model


# LayerNorm PostProcess
 
def find_nodes(graph, op_type):
    nodes = []
    for node in graph.node:
        if node.op_type == op_type:
            nodes.append(node)
    return nodes

def is_type(node, op_type):
    if node is None or isinstance(node, list):
        return False
    return node.op_type == op_type

def add_const(model, name, output, t_value = None, f_value = None):
    const_node = model.graph.node.add()
    const_node.op_type = 'Constant'
    const_node.name = name
    const_node.output.extend([output])
    attr = const_node.attribute.add()
    attr.name = 'value'
    if t_value is not None:
        attr.type = 4
        attr.t.CopyFrom(t_value)
    else:
        attr.type = 1
        attr.f = f_value
    return const_node

def layer_norm_transform(model):
    graph = model.graph

    nodes_ReduceMean = find_nodes(graph, "ReduceMean")

    id = 0
    layer_norm_nodes = []
    remove_nodes = []
    for reduce_mean in nodes_ReduceMean:
        # check that reduce_mean output is Sub
        sub = find_output_node(model, reduce_mean.output[0])
        if not is_type(sub, "Sub"):
            continue

        # check that sub output[0] is Div and output[1] is Pow
        pow, div = find_output_node(model, sub.output[0])
        if not is_type(div, "Div") or not is_type(pow, "Pow"):
            continue

        # check that pow ouput is ReduceMean
        reduce_mean2 = find_output_node(model, pow.output[0])
        if not is_type(reduce_mean2, "ReduceMean"):
            continue

        # check that reduce_mean2 output is Add
        add = find_output_node(model, reduce_mean2.output[0])
        if not is_type(add, "Add"):
            continue

        # check that add output is Sqrt
        sqrt = find_output_node(model, add.output[0])
        if not is_type(sqrt, "Sqrt"):
            continue

        # check that sqrt output is div
        if div != find_output_node(model, sqrt.output[0]):
            continue

        # check if div output is Mul
        optional_mul = find_output_node(model, div.output[0])
        if not is_type(optional_mul, "Mul"):
            optional_mul = None

        # check if mul output is Add
        if optional_mul is not None:
            optional_add = find_output_node(model, optional_mul.output[0])
        else:
            optional_add = find_output_node(model, div.output[0])
        if not is_type(optional_add, "Add"):
            optional_add = None


        # add nodes to remove_nodes
        remove_nodes.extend([reduce_mean, sub, div, pow, reduce_mean2, add, sqrt])

        # create LayerNorm node
        layer_norm_input = []
        layer_norm_output = []

        layer_norm_input.append(reduce_mean.input[0])

        if optional_mul is not None:
            remove_nodes.append(optional_mul)
            weight = optional_mul.input[1]
            layer_norm_input.append(weight)

        if optional_add is not None:
            remove_nodes.append(optional_add)
            bias = optional_add.input[1]
            layer_norm_input.append(bias)
        
        if optional_add is not None:
            layer_norm_output.append(optional_add.output[0])
        elif optional_mul is not None:
            layer_norm_output.append(optional_mul.output[0])
        else:
            layer_norm_output.append(div.output[0])

        layer_norm_output.append('saved_mean_' + str(id))
        layer_norm_output.append('saved_inv_std_var_' + str(id))

        epsilon_node = find_input_node(model, add.input[1])
        epsilon = epsilon_node.attribute[0].t.raw_data
        epsilon = struct.unpack('f', epsilon)[0]

        layer_norm = helper.make_node("LayerNormalization",
                                      layer_norm_input,
                                      layer_norm_output,
                                      "LayerNormalization_" + str(id),
                                      None,
                                      axis = reduce_mean.attribute[0].ints[0],
                                      epsilon = epsilon)
        layer_norm_nodes.append(layer_norm)
        id += 1

    # remove orphan constant nodes
    for constant in graph.node:
        if constant.op_type == "Constant" and constant not in remove_nodes:
            is_orphan = True
            for out_name in constant.output:
                out = find_output_node(model, out_name)
                if out not in remove_nodes:
                    is_orphan = False
            if is_orphan:
                remove_nodes.append(constant)
    
    all_nodes = []
    for node in graph.node:
        if node not in remove_nodes:
            all_nodes.append(node)

    for node in layer_norm_nodes:
        all_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(all_nodes)
    return model

# Fuse SoftmaxCrossEntropy

def fuse_softmaxNLL_to_softmaxCE(onnx_model):
    nll_count = 0
    while True:
        nll_count = nll_count + 1
        nll_loss_node = None
        nll_loss_node_index = 0
        for nll_loss_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "nll_loss" or node.op_type == "NegativeLogLikelihoodLoss":
                nll_loss_node = node
                break

        if nll_loss_node is None:
            break

        softmax_node = None
        softmax_node_index = 0
        label_input_name = None
        weight_input_name = None
        for softmax_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "LogSoftmax":
                # has to be connected to nll_loss
                if len(nll_loss_node.input) > 2:
                    weight_input_name = nll_loss_node.input[2]
                if node.output[0] == nll_loss_node.input[0]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[1]
                    break
                elif node.output[0] == nll_loss_node.input[1]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[0]
                    break
            else:
                if softmax_node is not None:
                    break

        if softmax_node is None:
            break

        # delete nll_loss and LogSoftmax nodes in order
        if nll_loss_node_index < softmax_node_index:
            del onnx_model.graph.node[softmax_node_index]
            del onnx_model.graph.node[nll_loss_node_index]
        else:
            del onnx_model.graph.node[nll_loss_node_index]
            del onnx_model.graph.node[softmax_node_index]

        probability_output_name = softmax_node.output[0]
        node = onnx_model.graph.node.add()
        inputs = [softmax_node.input[0], label_input_name, weight_input_name] if weight_input_name else [softmax_node.input[0], label_input_name]
        node.CopyFrom(onnx.helper.make_node("SparseSoftmaxCrossEntropy", inputs,
                                            [nll_loss_node.output[0], probability_output_name],
                                            "nll_loss_node_" + str(nll_count)))

    return onnx_model
