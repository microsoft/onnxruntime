import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np
"""
Generate test model for Gelu subgraph pattern 2:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)
"""

has_bias = True  # change it to True to generate gelu_format2_*_with_bias.onnx
gelu_use_graph_input = False  # change it to False to let Gelu don't have graph inputs as inputs.
node_has_graph_output = True # change it to False to let Gelu don't have graph output
switch_order = True  # switch order of inputs for Mul and Add

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 64])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", 64])
Z = helper.make_tensor_value_info('div', TensorProto.FLOAT, ["batch", "seqlen", 64])

value = (0.01 * np.arange(64)).astype(np.float32).reshape((64))
bias_initializer = numpy_helper.from_array(value, "input_bias")

value = np.asarray([1.4142099618911743]).astype(np.float32).reshape(())
initializer_sqrt_2 = numpy_helper.from_array(value, "mul1_init")

value = np.asarray([0.5]).astype(np.float32).reshape(())
initializer_0_5 = numpy_helper.from_array(value, "mul3_init")

value = np.asarray([1.0]).astype(np.float32).reshape(())
initializer_1 = numpy_helper.from_array(value, "add1_init")

nodes = []
gelu_input = "input"
if not gelu_use_graph_input:
    leading_identity = helper.make_node('Identity', [gelu_input], ['identity_leading'], name="identity_leading")
    gelu_input = "identity_leading"
    nodes.append(leading_identity)

gelu_root = gelu_input
if has_bias:
    add0 = helper.make_node(
        'Add', [gelu_input, bias_initializer.name] if switch_order else [bias_initializer.name, gelu_input], ['add0'],
        name="add0_node")
    gelu_root = "add0"
    nodes.append(add0)

div = helper.make_node('Div', [gelu_root, initializer_sqrt_2.name], ['div'], name="div_node")
nodes.append(div)

erf = helper.make_node('Erf', ['div'], ['erf'], name="erf_node")
nodes.append(erf)

add1 = helper.make_node('Add', ['erf', initializer_1.name] if switch_order else [initializer_1.name, 'erf'], ['add1'],
                        name="add1")
nodes.append(add1)

mul = helper.make_node('Mul', [gelu_root, 'add1'] if switch_order else ['add1', gelu_root], ['mul'], name="mul_node")
nodes.append(mul)

mul2 = helper.make_node('Mul', ['mul', initializer_0_5.name] if switch_order else [initializer_0_5.name, 'mul'],
                        ['mul2'],
                        name="mul2_node")
nodes.append(mul2)

ending_identity = helper.make_node('Identity', ['mul2'], ['output'], name="identity_ending")
nodes.append(ending_identity)

initializers = []
if has_bias:
    initializers = [bias_initializer]

initializers.extend([initializer_sqrt_2, initializer_1, initializer_0_5])

# Create the graph (GraphProto)
graph_def = helper.make_graph(nodes, 'gelu_pattern_2', [X], [Y, Z] if node_has_graph_output else [Y], initializers)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 10
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

file_name = "gelu_format2_0" if switch_order else "gelu_format2_1"
if has_bias:
    file_name += "_with_bias"

if gelu_use_graph_input:
    file_name += "_use_graph_input"

if node_has_graph_output:
    file_name += "_use_graph_output"

file_name += ".onnx"
onnx.save(model_def, file_name)
print(file_name)
