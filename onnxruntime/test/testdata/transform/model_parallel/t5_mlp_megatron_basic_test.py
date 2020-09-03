import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

hidden_size = 4
weight_dim_to_split = 16

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", hidden_size])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "seqlen", hidden_size])

a_weight_np_vals = (0.01 * np.arange(hidden_size * weight_dim_to_split, dtype=np.float32)).reshape((weight_dim_to_split, hidden_size))
a_weight_initializer = numpy_helper.from_array(a_weight_np_vals, "encoder.t5_stack.block.1.layer.1.DenseReluDense.wi.weight")

#dropout_np_vals = (0.1 * np.arange(1, dtype=np.float32)).reshape((1,))
dropout_np_vals = np.asarray([0.1], dtype=np.float32).reshape(())
dropout_initializer = numpy_helper.from_array(dropout_np_vals, "ratio")
 
#dropout_mode_np_vals = np.array([True], dtype=np.bool).reshape((1,))
dropout_mode_np_vals = np.array([True], dtype=np.bool).reshape(())
dropout_mode_initializer = numpy_helper.from_array(dropout_mode_np_vals, "mode")

b_weight_np_vals = (0.01 * np.arange(hidden_size * weight_dim_to_split, dtype=np.float32)).reshape((hidden_size, weight_dim_to_split))
b_weight_initializer = numpy_helper.from_array(b_weight_np_vals, "encoder.t5_stack.block.1.layer.1.DenseReluDense.wo.weight")


transpose1 = helper.make_node('Transpose', [a_weight_initializer.name], ['transpose1'], name='transpose1', perm=[1,0])
transpose2 = helper.make_node('Transpose', [b_weight_initializer.name], ['transpose2'], name='transpose2', perm=[1,0])
matmul = helper.make_node(
    'MatMul', # node name
    ['input', 'transpose1'], # inputs
    ['matmul'], # outputs
    name="matmul"
)

relu = helper.make_node(
    'Relu', # node name
    ['matmul'], # inputs
    ['relu'], # outputs
    name="relu"
)

dropout1 = helper.make_node('Dropout', 
    ["relu", dropout_initializer.name, dropout_mode_initializer.name], 
    ['dropout1', "dropout1_mask"], 
    name='dropout1')

matmul2 = helper.make_node(
    'MatMul', # node name
    ['dropout1', "transpose2"], # inputs
    ['matmul2'], # outputs
    name="matmul2"
)

dropout2 = helper.make_node('Dropout', 
    ["matmul2", dropout_initializer.name, dropout_mode_initializer.name], 
    ['dropout2', "dropout2_mask"], 
    name='dropout2')

identity = helper.make_node(
    'Identity', # node name
    ['dropout2'], # inputs
    ['output'], # outputs
    name="identity"
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [transpose1, transpose2, matmul, relu, dropout1, matmul2, dropout2, identity],
    'test-model',
    [X],
    [Y],
    [a_weight_initializer, b_weight_initializer, dropout_initializer, dropout_mode_initializer]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
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

onnx.save(model_def, 't5_mlp_megatron_basic_test.onnx')