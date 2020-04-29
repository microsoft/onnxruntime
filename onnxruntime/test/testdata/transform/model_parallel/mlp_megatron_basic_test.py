import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

hidden_size = 4
weight_dim_to_split = 16

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", hidden_size])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", hidden_size])

a_weight_np_vals = (0.01 * np.arange(hidden_size * weight_dim_to_split, dtype=np.float32)).reshape((hidden_size, weight_dim_to_split))
a_weight_initializer = numpy_helper.from_array(a_weight_np_vals, "transformer.layers.0.mlp.dense_h_to_4h.weight_transposed")

a_bias_np_vals = (0.01 * np.arange(weight_dim_to_split, dtype=np.float32))  # weight_dim_to_split numbers in total
a_bias_initializer = numpy_helper.from_array(a_bias_np_vals, "transformer.layers.0.mlp.dense_h_to_4h.bias")

b_weight_np_vals = (0.01 * np.arange(weight_dim_to_split * hidden_size, dtype=np.float32)).reshape((weight_dim_to_split, hidden_size))
b_weight_initializer = numpy_helper.from_array(b_weight_np_vals, "transformer.layers.0.mlp.dense_4h_to_h.weight_transposed")

b_bias_np_vals = (0.01 * np.arange(hidden_size, dtype=np.float32))  # hidden_size numbers in total
b_bias_initializer = numpy_helper.from_array(b_bias_np_vals, "transformer.layers.0.mlp.dense_4h_to_h.bias")

matmul = helper.make_node(
    'MatMul', # node name
    ['input', a_weight_initializer.name], # inputs
    ['matmul'], # outputs
    name="matmul"
)

add = helper.make_node(
    'Add', # node name
    ['matmul', a_bias_initializer.name], # inputs
    ['add'], # outputs
    name="add"
)

gelu = helper.make_node(
    'Gelu', # node name
    ['add'], # inputs
    ['gelu'], # outputs
    name="gelu",
    doc_string="",
    domain="com.microsoft"
)

matmul2 = helper.make_node(
    'MatMul', # node name
    ['gelu', b_weight_initializer.name], # inputs
    ['matmul2'], # outputs
    name="matmul2"
)

add2 = helper.make_node(
    'Add', # node name
    ['matmul2', b_bias_initializer.name], # inputs
    ['add2'], # outputs
    name="add2"
)

identity = helper.make_node(
    'Identity', # node name
    ['add2'], # inputs
    ['output'], # outputs
    name="identity"
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [matmul, add, gelu, matmul2, add2, identity],
    'test-model',
    [X],
    [Y],
    [a_weight_initializer, a_bias_initializer, b_weight_initializer, b_bias_initializer]
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

onnx.save(model_def, 'mlp_megatron_basic_test.onnx')