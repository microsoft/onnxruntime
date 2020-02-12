import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np


X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 64])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", 64])

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


mul1 = helper.make_node(
    'Mul', # node name
    ['input', a_weight_initializer.name], # inputs
    ['mul1'], # outputs
    name="mul1"
)

mul2 = helper.make_node(
    'Mul', # node name
    ['input', 'mul1'], # inputs
    ['mul2'], # outputs
    name="mul2"
)

add1 = helper.make_node(
    'Add', # node name
    ['mul2', a_bias_initializer.name], # inputs
    ['add1'], # outputs
    name="add1"
)

mul3 = helper.make_node(
    'Mul', # node name
    ['input', b_weight_initializer.name], # inputs
    ['mul3'], # outputs
    name="mul3"
)

mul4 = helper.make_node(
    'Mul', # node name
    ['mul3', 'add1'], # inputs
    ['mul4'], # outputs
    name="mul4"
)

tanh = helper.make_node(
    'Tanh', # node name
    ['mul4'], # inputs
    ['tanh'], # outputs
    name="tanh"
)

add2 = helper.make_node(
    'Add', # node name
    ['tanh', b_bias_initializer.name], # inputs
    ['add2'], # outputs
    name="add2"
)

mul5 = helper.make_node(
    'Mul', # node name
    ['input', c_weight_initializer.name], # inputs
    ['mul5'], # outputs
    name="mul5"
)

mul6 = helper.make_node(
    'Mul', # node name
    ['mul5', 'add2'], # inputs
    ['mul6'], # outputs
    name="mul6"
)

identity = helper.make_node(
    'Identity', # node name
    ['mul6'], # inputs
    ['output'], # outputs
    name="identity"
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [mul1, mul2, add1, mul3, mul4, tanh, add2, mul5, mul6, identity],
    'test-model',
    [X],
    [Y],
    [a_weight_initializer, a_bias_initializer, b_weight_initializer, b_bias_initializer, c_weight_initializer]
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

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, 'fast_gelu.onnx')

