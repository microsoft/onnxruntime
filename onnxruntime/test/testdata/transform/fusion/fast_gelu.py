import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

has_bias = True # change it to True to generate fast_gelu_with_bias.onnx

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 64])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", 64])

bias_np_vals = (0.01 * np.arange(64)).astype(np.float32).reshape((64))
bias_initializer = numpy_helper.from_array(bias_np_vals, "input_bias")

a_weight_np_vals = np.asarray([0.044714998453855515]).astype(np.float32).reshape(())
a_weight_initializer = numpy_helper.from_array(a_weight_np_vals, "mul1_init")

b_weight_np_vals = np.asarray([0.7978845834732056]).astype(np.float32).reshape(())
b_weight_initializer = numpy_helper.from_array(b_weight_np_vals, "mul2_init")

c_weight_np_vals = np.asarray([0.5]).astype(np.float32).reshape(())
c_weight_initializer = numpy_helper.from_array(c_weight_np_vals, "mul3_init")

a_bias_np_vals = np.asarray([1.0]).astype(np.float32).reshape(())
a_bias_initializer = numpy_helper.from_array(a_bias_np_vals, "add1_init")

b_bias_np_vals = np.asarray([1.0]).astype(np.float32).reshape(())
b_bias_initializer = numpy_helper.from_array(b_bias_np_vals, "add2_init")

mul_input_name = "input"
if has_bias:
    add0 = helper.make_node(
        'Add', 
        ['input', bias_initializer.name], 
        ['add0'], 
        name="add0"
    )
    mul_input_name = "add0"


mul1 = helper.make_node(
    'Mul', 
    [mul_input_name, a_weight_initializer.name], 
    ['mul1'], 
    name="mul1"
)

mul2 = helper.make_node(
    'Mul', 
    [mul_input_name, 'mul1'], 
    ['mul2'], 
    name="mul2"
)

add1 = helper.make_node(
    'Add', 
    ['mul2', a_bias_initializer.name], 
    ['add1'], 
    name="add1"
)

mul3 = helper.make_node(
    'Mul', 
    [mul_input_name, b_weight_initializer.name], 
    ['mul3'], 
    name="mul3"
)

mul4 = helper.make_node(
    'Mul', 
    ['mul3', 'add1'], 
    ['mul4'], 
    name="mul4"
)

tanh = helper.make_node(
    'Tanh', 
    ['mul4'], 
    ['tanh'], 
    name="tanh"
)

add2 = helper.make_node(
    'Add', 
    ['tanh', b_bias_initializer.name], 
    ['add2'], 
    name="add2"
)

mul5 = helper.make_node(
    'Mul', 
    [mul_input_name, c_weight_initializer.name], 
    ['mul5'], 
    name="mul5"
)

mul6 = helper.make_node(
    'Mul', 
    ['mul5', 'add2'], 
    ['mul6'], 
    name="mul6"
)

identity = helper.make_node(
    'Identity', 
    ['mul6'], 
    ['output'], 
    name="identity"
)

nodes = []
initializers = []
if has_bias:
    nodes = [add0]
    initializers = [bias_initializer]
nodes.extend([mul1, mul2, add1, mul3, mul4, tanh, add2, mul5, mul6, identity])
initializers.extend([a_weight_initializer, a_bias_initializer, b_weight_initializer, b_bias_initializer, c_weight_initializer])
# Create the graph (GraphProto)
graph_def = helper.make_graph(
    nodes,
    'test-model',
    [X],
    [Y],
    initializers
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 10
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs={}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

file_name = "fast_gelu.onnx"
if has_bias:
    file_name = "fast_gelu_with_bias.onnx"
onnx.save(model_def, file_name)

