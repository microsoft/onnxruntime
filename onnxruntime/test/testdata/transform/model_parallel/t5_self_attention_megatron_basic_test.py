import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np
import random

hidden_size = 4
attention_head = 2
hidden_per_attention = 2

relative_attention_num_buckets=32
input_len=8
output_len=8

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', input_len, hidden_size])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', output_len, hidden_size])

gather_data_np_vals = (0.01 * np.arange(relative_attention_num_buckets * attention_head, dtype=np.float32)).reshape((relative_attention_num_buckets, attention_head))
gather_data_initializer = numpy_helper.from_array(gather_data_np_vals,
    "encoder.t5_stack.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
gather_indices_np_vals = np.array([random.randint(0, 31) for i in range(input_len * output_len)], dtype=np.int64).reshape((input_len, output_len))
gather_indices_initializer = numpy_helper.from_array(gather_indices_np_vals,
    "encoder.t5_stack.block.0.layer.0.SelfAttention.relative_attention_bias.indices")

relative_bias_add_np_vals = np.array([0], dtype=np.float32).reshape(())
relative_bias_add_initializer = numpy_helper.from_array(relative_bias_add_np_vals,
    "relative_bias_add")

q_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
q_weight_initializer = numpy_helper.from_array(q_weight_np_vals, 'encoder.t5_stack.block.1.layer.0.SelfAttention.q.weight')

k_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
k_weight_initializer = numpy_helper.from_array(k_weight_np_vals, 'encoder.t5_stack.block.1.layer.0.SelfAttention.k.weight')

v_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
v_weight_initializer = numpy_helper.from_array(v_weight_np_vals, 'encoder.t5_stack.block.1.layer.0.SelfAttention.v.weight')

reshape_shape_np_vals = np.asarray([0, 0, attention_head, hidden_per_attention], dtype=np.int64)
q_shape_initializer = numpy_helper.from_array(reshape_shape_np_vals, 'q_shape')
k_shape_initializer = numpy_helper.from_array(reshape_shape_np_vals, 'k_shape')
v_shape_initializer = numpy_helper.from_array(reshape_shape_np_vals, 'v_shape')

dropout_np_vals = np.asarray([0.1], dtype=np.float32).reshape(())
dropout_initializer = numpy_helper.from_array(dropout_np_vals, "ratio")
 
dropout_mode_np_vals = np.array([True], dtype=np.bool).reshape(())
dropout_mode_initializer = numpy_helper.from_array(dropout_mode_np_vals, "mode")

dense_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
dense_weight_initializer = numpy_helper.from_array(dense_weight_np_vals, 'encoder.t5_stack.block.1.layer.0.SelfAttention.o.weight')

shape_val3 = np.array([0, 0, hidden_size], dtype=np.int64)
shape_initializer3 = numpy_helper.from_array(shape_val3, 'concat_shape_3')

gather1 = helper.make_node('Gather', [gather_data_initializer.name, gather_indices_initializer.name], ['gather_output'], name="gather_output", axis=0)
transpose_relative_bias = helper.make_node('Transpose', ["gather_output"], ['transpose_relative_bias'], name='transpose_relative_bias', perm=[2,0,1])
unsqueeze_relative_bias = helper.make_node('Unsqueeze', ["transpose_relative_bias"], 
    ['unsqueeze_relative_bias'], name='unsqueeze_relative_bias', axes=[0])
add = helper.make_node('Add', ['unsqueeze_relative_bias', relative_bias_add_initializer.name], ['add'], name='add')

transpose_q = helper.make_node('Transpose', [q_weight_initializer.name], ['transpose_q'], name='transpose_q', perm=[1,0])
transpose_k = helper.make_node('Transpose', [k_weight_initializer.name], ['transpose_k'], name='transpose_k', perm=[1,0])
transpose_v = helper.make_node('Transpose', [v_weight_initializer.name], ['transpose_v'], name='transpose_v', perm=[1,0])

matmul_q = helper.make_node('MatMul', ['input', 'transpose_q'], ['matmul_q'], name='matmul_q')
matmul_k = helper.make_node('MatMul', ['input', 'transpose_k'], ['matmul_k'], name='matmul_k')
matmul_v = helper.make_node('MatMul', ['input', 'transpose_v'], ['matmul_v'], name='matmul_v')

reshape_q = helper.make_node('Reshape', ['matmul_q', q_shape_initializer.name], ['reshape_q'], name='reshape_q')
reshape_k = helper.make_node('Reshape', ['matmul_k', k_shape_initializer.name], ['reshape_k'], name='reshape_k')
reshape_v = helper.make_node('Reshape', ['matmul_v', v_shape_initializer.name], ['reshape_v'], name='reshape_v')

transpose_q2 = helper.make_node('Transpose', ['reshape_q'], ['transpose_q2'], name='transpose_q2', perm=[0,2,1,3])
transpose_k2 = helper.make_node('Transpose', ['reshape_k'], ['transpose_k2'], name='transpose_k2', perm=[0,2,3,1])
transpose_v2 = helper.make_node('Transpose', ['reshape_v'], ['transpose_v2'], name='transpose_v2', perm=[0,2,1,3])

matmul = helper.make_node('MatMul', ['transpose_q2', 'transpose_k2'], ['matmul'], name='matmul')
add2 = helper.make_node('Add', ['matmul', 'add'], ['add2'], name='add2')
softmax = helper.make_node('Softmax', ['add2'], ['softmax'], name='softmax', axis=3)
dropout1 = helper.make_node('Dropout', 
    ["softmax", dropout_initializer.name, dropout_mode_initializer.name], 
    ['dropout1', "dropout1_mask"], 
    name='dropout1')

matmul2 = helper.make_node('MatMul', ['dropout1', 'transpose_v2'], ['matmul2'], name='matmul2')
transpose = helper.make_node('Transpose', ['matmul2'], ['transpose'], name='transpose', perm=[0,2,1,3])
reshape = helper.make_node('Reshape', ['transpose', shape_initializer3.name], ['reshape'], name='reshape')

transpose_o_weight = helper.make_node('Transpose', [dense_weight_initializer.name], ['transpose_o_weight'], name='transpose_o_weight', perm=[1,0])
matmul3 = helper.make_node('MatMul', ['reshape', 'transpose_o_weight'], ['matmul3'], name='matmul3')
identity = helper.make_node('Identity', ['matmul3'], ['output'], name='identity')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [gather1, transpose_relative_bias, unsqueeze_relative_bias, add,
     transpose_q, transpose_k, transpose_v, matmul_q, matmul_k, matmul_v, reshape_q, reshape_k, reshape_v,
     transpose_q2,transpose_k2,transpose_v2,matmul,add2, softmax,dropout1,matmul2, transpose, reshape, transpose_o_weight,
     matmul3, identity],
    'self-attention-megatron-test-model',
    [X],
    [Y],
    [gather_data_initializer, gather_indices_initializer, relative_bias_add_initializer, q_weight_initializer,
    k_weight_initializer, v_weight_initializer, q_shape_initializer, k_shape_initializer, v_shape_initializer,
    dropout_initializer, dropout_mode_initializer, dense_weight_initializer, shape_initializer3]
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
onnx.save(model_def, 't5_self_attention_megatron_basic_test.onnx')
