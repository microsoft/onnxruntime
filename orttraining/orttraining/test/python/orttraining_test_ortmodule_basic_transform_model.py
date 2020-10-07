# coding=utf8
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

def add_model_input_from_initializer(model, initializer, docstring=None):
    new_input = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims, docstring)
    model.graph.input.append(new_input)

def add_model_input(model, name, data_type, dims, docstring=None):
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
#   They are: label
node = find_model_input(model, 'label')
model.graph.input.remove(node)

# Remove model outputs
#   They are: loss
node = find_model_output(model, 'loss')
model.graph.output.remove(node)
node = find_model_output(model, 'fc1.bias_grad')
model.graph.output.remove(node)
node = find_model_output(model, 'fc1.weight_grad')
model.graph.output.remove(node)
node = find_model_output(model, 'fc2.bias_grad')
model.graph.output.remove(node)
node = find_model_output(model, 'fc2.weight_grad')
model.graph.output.remove(node)

# Add input with same name, type and shape as the initializers
#   They are: [fc1.bias, fc1.weight, fc2.bias, fc2.weight]
node = find_initializer(model, 'fc1.bias')
add_model_input_from_initializer(model, node, 'thiagofc: add fc1.bias as model input')
node = find_initializer(model, 'fc1.weight')
add_model_input_from_initializer(model, node, 'thiagofc: add fc1.weight as model input')
node = find_initializer(model, 'fc2.bias')
add_model_input_from_initializer(model, node, 'thiagofc: add fc2.bias as model input')
node = find_initializer(model, 'fc2.weight')
add_model_input_from_initializer(model, node, 'thiagofc: add fc2.weight as model input')

# Remove initializers from model
#   They are: [fc1.bias, fc1.weight, fc2.bias, fc2.weight]
# TODO: Do this when we are able to distinguish inputs from initializers
# model.graph.initializer.remove(node)
# model.graph.initializer.remove(node)
# model.graph.initializer.remove(node)
# model.graph.initializer.remove(node)

# Remove backward-related initializers
#   They are: [loss_grad, ZeroConstant]
node = find_initializer(model, 'loss_grad')
model.graph.initializer.remove(node)
node = find_initializer(model, 'ZeroConstant')
model.graph.initializer.remove(node)

# Remove OPs
#   They are: [SoftmaxCrossEntropyLoss_3, SoftmaxCrossEntropyLoss_3_Grad/SoftmaxCrossEntropyLossGrad_0,
#              Gemm_2_Grad/ReduceSum_3, Gemm_2_Grad/Identity_4, Gemm_2_Grad/Gemm_2, Gemm_2_Grad/Gemm_1,
#              Relu_1_Grad/ReluGrad_0, Gemm_0_Grad/Gemm_1, Gemm_0_Grad/ReduceSum_2, Gemm_0_Grad/Identity_3]
node = find_node(model, 'SoftmaxCrossEntropyLoss_3')
model.graph.node.remove(node)
node = find_node(model, 'SoftmaxCrossEntropyLoss_3_Grad/SoftmaxCrossEntropyLossGrad_0')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_2_Grad/ReduceSum_3')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_2_Grad/Identity_4')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_2_Grad/Gemm_2')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_2_Grad/Gemm_1')
model.graph.node.remove(node)
node = find_node(model, 'Relu_1_Grad/ReluGrad_0')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_0_Grad/Gemm_1')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_0_Grad/ReduceSum_2')
model.graph.node.remove(node)
node = find_node(model, 'Gemm_0_Grad/Identity_3')
model.graph.node.remove(node)

# Add new outputs:
#   They are: 7
add_output(model, '7', None, 'thiagofc: add 7 as model output')

with open(output_forward_model_name, "wb") as f:
    f.write(model.SerializeToString())


###############################################################################
# FORWARD PASS GRAPH ##########################################################
###############################################################################
model = onnx.load(input_model_name)



with open(output_backward_model_name, "wb") as f:
    f.write(model.SerializeToString())
