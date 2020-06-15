import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

hidden_size = 4
attention_head = 2
hidden_per_attention = 2

# Self-attention.
# Handle self-attension.
# MatMul->Add->Split->Reshape->Transpose->MatMul->Div->Mul->Sub->Softmax->Dropout->MatMul->Transpose->Reshape->MatMul->Add
#                  |->Reshape->Transpose->|                                        |
#                  |->Reshape->Transpose------------------------------------------>|

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 'seqlen', hidden_size])
X_mask = helper.make_tensor_value_info('mask', TensorProto.FLOAT, ['batch', 1, 'seqlen', 'seqlen'])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 'seqlen', hidden_size])

qkv_weight_np_vals = (0.01 * np.arange(hidden_size * (hidden_size * 3), dtype=np.float32)).reshape((hidden_size, hidden_size * 3))
qkv_weight_initializer = numpy_helper.from_array(qkv_weight_np_vals, 'transformer.attention.query_key_value.weight_transposed')

qkv_bias_np_vals = (0.01 * np.arange(hidden_size * 3, dtype=np.float32))
qkv_bias_initializer = numpy_helper.from_array(qkv_bias_np_vals, 'transformer.attention.query_key_value.bias')

dense_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
dense_weight_initializer = numpy_helper.from_array(dense_weight_np_vals, 'transformer.attention.dense.weight_transposed')

dense_bias_np_vals = (0.01 * np.arange(hidden_size, dtype=np.float32))
dense_bias_initializer = numpy_helper.from_array(dense_bias_np_vals, 'transformer.attention.dense.bias')

shape_val = np.array([0, 0, attention_head, hidden_per_attention], dtype=np.int64)
shape_initializer = numpy_helper.from_array(shape_val, 'concat_shape_0')
shape_val1 = np.array([0, 0, attention_head, hidden_per_attention], dtype=np.int64)
shape_initializer1 = numpy_helper.from_array(shape_val1, 'concat_shape_1')
shape_val2 = np.array([0, 0, attention_head, hidden_per_attention], dtype=np.int64)
shape_initializer2 = numpy_helper.from_array(shape_val2, 'concat_shape_2')
shape_val3 = np.array([0, 0, hidden_size], dtype=np.int64)
shape_initializer3 = numpy_helper.from_array(shape_val3, 'concat_shape_3')

matmul1 = helper.make_node('MatMul', ['input', qkv_weight_initializer.name], ['matmul1'], name='matmul1')
add1 = helper.make_node('Add', ['matmul1', qkv_bias_initializer.name], ['add1'], name='add1')
split = helper.make_node('Split', ['add1'], ['mixed_query_layer', 'mixed_key_layer', 'mixed_value_layer'], name='split', axis=2)
reshape = helper.make_node('Reshape', ['mixed_query_layer', shape_initializer.name], ['reshape'], name='reshape')
reshape1 = helper.make_node('Reshape', ['mixed_key_layer', shape_initializer1.name], ['reshape1'], name='reshape1')
reshape2 = helper.make_node('Reshape', ['mixed_value_layer', shape_initializer2.name], ['reshape2'], name='reshape2')
transpose = helper.make_node('Transpose', ['reshape'], ['transpose'], name='transpose', perm=[0,2,1,3])
transpose1 = helper.make_node('Transpose', ['reshape1'], ['transpose1'], name='transpose1', perm=[0,2,3,1])
transpose2 = helper.make_node('Transpose', ['reshape2'], ['transpose2'], name='transpose2', perm=[0,2,1,3])
matmul2 = helper.make_node('MatMul', ['transpose', 'transpose1'], ['matmul2'], name='matmul2')
# Use the mask input for below 3 nodes. This is different from the original GPT-2 model, but it's OK for te sub-graph test.
div = helper.make_node('Div', ['matmul2', 'mask'], ['div'], name='div')
mul = helper.make_node('Mul', ['div', 'mask'], ['mul'], name='mul')
sub = helper.make_node('Sub', ['mul', 'mask'], ['sub'], name='sub')
softmax = helper.make_node('Softmax', ['sub'], ['softmax'], name='softmax', axis=3)
dropout1 = helper.make_node('Dropout', ['softmax'], ['dropout1'], name='dropout1')
matmul3 = helper.make_node('MatMul', ['dropout1', 'transpose2'], ['matmul3'], name='matmul3')
transpose3 = helper.make_node('Transpose', ['matmul3'], ['transpose3'], name='transpose3', perm=[0,2,1,3])
reshape3 = helper.make_node('Reshape', ['transpose3', shape_initializer3.name], ['reshape3'], name='reshape3')
matmul4 = helper.make_node('MatMul', ['reshape3', dense_weight_initializer.name], ['matmul4'], name='matmul4')
add2 = helper.make_node('Add', ['matmul4', dense_bias_initializer.name], ['add2'], name='add2')
dropout2 = helper.make_node('Dropout', ['add2'], ['dropout2'], name='dropout2')
# Add dummy Identity so during inference dropout2 can be removed for testing.
identity = helper.make_node('Identity', ['dropout2'], ['output'], name='identity')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [matmul1, add1, split, reshape, reshape1, reshape2, transpose, transpose1, transpose2, matmul2, div, mul, sub, softmax, dropout1, matmul3, transpose3, reshape3, matmul4, add2, dropout2, identity],
    'self-attention-megatron-test-model',
    [X, X_mask],
    [Y],
    [qkv_weight_initializer, qkv_bias_initializer, dense_weight_initializer, dense_bias_initializer, shape_initializer, shape_initializer1, shape_initializer2, shape_initializer3]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'self_attention_megatron_basic_test.onnx')
