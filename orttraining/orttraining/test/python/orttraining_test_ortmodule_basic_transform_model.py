# coding=utf8
import copy
import sys
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import numpy as np
from onnx import numpy_helper

if len(sys.argv) < 2:
    print("Please give model path...")
    exit(1)

input_model_name = sys.argv[1]
output_forward_model_name = input_model_name[:-5] + '_forward_sliced.onnx'
output_backward_model_name = input_model_name[:-5] + '_backward_sliced.onnx'

def add_input_from_initializer(model, initializer, docstring=None):
    new_input = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims, docstring)
    model.graph.input.append(new_input)

def add_input(model, name, data_type = None, dims = None, docstring = None):
    new_input = onnx.helper.make_tensor_value_info(name, data_type, dims, docstring)
    model.graph.input.append(new_input)

def add_output(model, name, data_type = None, docstring = None):
    new_output = model.graph.value_info.add()
    new_output.name = name
    if data_type:
        new_output.type.CopyFrom(data_type)
    if docstring:
        new_output.doc_string = docstring
    model.graph.output.append(new_output)

def find_model_input(model, input_name):
    for input in model.graph.input:
        if input.name == input_name:
            return input
    return None

def find_model_output(model, output_name):
    for output in model.graph.output:
        if output.name == output_name:
            return output
    return None

def find_initializer(model, name):
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer
    return None

def find_node(model, name):
    for node in model.graph.node:
        if node.name == name:
            return node
    return None

###############################################################################
# FORWARD PASS GRAPH ##########################################################
###############################################################################
model = onnx.load(input_model_name)

# Remove model inputs
nodes = ['label']
for node in nodes:
    node = find_model_input(model, node)
    model.graph.input.remove(node)

# Remove model outputs
nodes = ['loss', 'fc1.bias_grad', 'fc1.weight_grad', 'fc2.bias_grad', 'fc2.weight_grad']
for node in nodes:
    node = find_model_output(model, node)
    model.graph.output.remove(node)

# Add input with same name, type and shape as the initializers
#   They are: [fc1.bias, fc1.weight, fc2.bias, fc2.weight]
forward_initializer_names = ['fc1.bias', 'fc1.weight', 'fc2.bias', 'fc2.weight']
forward_initializer = {}
for node in forward_initializer_names:
    node = find_initializer(model, node)
    add_input_from_initializer(model, node, f'thiagofc: add {node.name} as model input')
    forward_initializer.update({node.name : copy.deepcopy(node)})

# Remove initializers from model
# TODO: Do this when we are able to distinguish inputs from initializers
# for node in forward_initializer_names:
#    node = find_initializer(model, init)
#    model.graph.initializer.remove(node)

# Remove backward-related initializers
nodes = ['loss_grad', 'ZeroConstant']
for node in nodes:
    node = find_initializer(model, node)
    model.graph.initializer.remove(node)

# Remove OPs
nodes = ['SoftmaxCrossEntropyLoss_3', 'SoftmaxCrossEntropyLoss_3_Grad/SoftmaxCrossEntropyLossGrad_0',
         'Gemm_2_Grad/ReduceSum_3', 'Gemm_2_Grad/Identity_4', 'Gemm_2_Grad/Gemm_2', 'Gemm_2_Grad/Gemm_1',
         'Relu_1_Grad/ReluGrad_0', 'Gemm_0_Grad/Gemm_1', 'Gemm_0_Grad/ReduceSum_2', 'Gemm_0_Grad/Identity_3']
for node in nodes:
    node = find_node(model, node)
    model.graph.node.remove(node)

# Add new outputs:
#   They are: 7
add_output(model, '7', None, 'thiagofc: add 7 as model output')

with open(output_forward_model_name, "wb") as f:
    f.write(model.SerializeToString())


###############################################################################
# BACKWARD PASS GRAPH ##########################################################
###############################################################################
model = onnx.load(input_model_name)

# Add new inputs:
# TODO: Should we specify types here? ORT graph doesn't have that info available, but ONNX API needs it
add_input_from_initializer(model, forward_initializer['fc2.weight'], 'thiagofc: add fc2.weight as model input')
add_input(model, '7', 1, None, 'thiagofc: add 7 as model input')
add_input(model, 'probability_grad', 1, None, 'thiagofc: add probability_grad as model input')

# Remove model inputs
#   They are: label
node = find_model_input(model, 'label')
model.graph.input.remove(node)

# Remove model outputs
nodes = ['loss', 'probability']
for node in nodes:
    node = find_model_output(model, node)
    model.graph.output.remove(node)

# Remove OP nodes from forward pass
nodes = ['Gemm_0', 'Relu_1', 'Gemm_2', 'SoftmaxCrossEntropyLoss_3', 'SoftmaxCrossEntropyLoss_3_Grad/SoftmaxCrossEntropyLossGrad_0']
for node in nodes:
    node = find_node(model, node)
    model.graph.node.remove(node)

# Remove initializers
forward_initializer_names.extend(['loss_grad'])
for node in forward_initializer_names:
   node = find_initializer(model, node)
   model.graph.initializer.remove(node)

with open(output_backward_model_name, "wb") as f:
    f.write(model.SerializeToString())
