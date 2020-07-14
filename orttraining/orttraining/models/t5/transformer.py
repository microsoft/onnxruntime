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
output_model_name = input_model_name[:-5] + '_optimized.onnx'

model = onnx.load(input_model_name)
batch_size = 2
sequence_len = 16
sequence_len_output = 4
for model_input in model.graph.input:
    if model_input.name in ["inputs", "inputs_mask"]:
        del model_input.type.tensor_type.shape.dim[1]
        del model_input.type.tensor_type.shape.dim[0]
        d = model_input.type.tensor_type.shape.dim.add()
        d.dim_value = batch_size
        d2 = model_input.type.tensor_type.shape.dim.add()
        d2.dim_value = sequence_len
    if model_input.name in ["targets", "targets_mask"]:
        del model_input.type.tensor_type.shape.dim[1]
        del model_input.type.tensor_type.shape.dim[0]
        d = model_input.type.tensor_type.shape.dim.add()
        d.dim_value = batch_size
        d2 = model_input.type.tensor_type.shape.dim.add()
        d2.dim_value = sequence_len_output

for model_output in model.graph.output:
    if model_output.name in ["lm_logits"]:
        del model_output.type.tensor_type.shape.dim[2]
        del model_output.type.tensor_type.shape.dim[1]
        del model_output.type.tensor_type.shape.dim[0]
        d = model_output.type.tensor_type.shape.dim.add()
        d.dim_value = batch_size
        d2 = model_output.type.tensor_type.shape.dim.add()
        d2.dim_value = sequence_len_output
        d3 = model_output.type.tensor_type.shape.dim.add()
        d3.dim_value = 32128

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

def get_node_index(model, node):
    i = 0
    while i < len(model.graph.node):
        if model.graph.node[i] == node:
            break
        i += 1
    return i if i < len(model.graph.node) else None

def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)
    return result[0] if len(result)== 1 else None

for node in model.graph.node:
    if node.op_type == 'Reshape' and node.name in ["Reshape_455"]:
        #insert new shape to reshape
        data_tensor = numpy_helper.from_array(np.asarray([0] * batch_size).astype(np.int64).reshape((batch_size)))
        const_node = add_const(model, 'reshape_data_node_for_' + node.name, 'reshape_data_for' + node.name, data_tensor)
        previous_data_out = node.input[0]
        node.input[0] = const_node.output[0]
        #delete the const input
        del model.graph.node[get_node_index(model, find_input_node(model, previous_data_out))]

onnx.save(model, "./tiny_optimized/tiny_optimized.onnx")