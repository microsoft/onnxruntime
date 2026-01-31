from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Configuration
batch_size = 2
sequence_length = 4
past_sequence_length = 0
max_sequence_length = 128
head_size = 16
num_heads = 4
kv_num_heads = 2
hidden_size = num_heads * head_size
kv_hidden_size = kv_num_heads * head_size

# 1. Create the ONNX model
# Define input shapes
query_shape = [batch_size, sequence_length, hidden_size]
key_shape = [batch_size, sequence_length, kv_hidden_size]
value_shape = [batch_size, sequence_length, kv_hidden_size]
past_shape = [batch_size, kv_num_heads, max_sequence_length, head_size]
seqlens_k_shape = [batch_size]
total_seq_len_shape = [] # Scalar
cache_shape = [max_sequence_length, head_size // 2]

# Define inputs
input_infos = [
    helper.make_tensor_value_info('query', TensorProto.FLOAT, query_shape),
    helper.make_tensor_value_info('key', TensorProto.FLOAT, key_shape),
    helper.make_tensor_value_info('value', TensorProto.FLOAT, value_shape),
    helper.make_tensor_value_info('past_key', TensorProto.FLOAT, past_shape),
    helper.make_tensor_value_info('past_value', TensorProto.FLOAT, past_shape),
    helper.make_tensor_value_info('seqlens_k', TensorProto.INT32, seqlens_k_shape),
    helper.make_tensor_value_info('total_sequence_length', TensorProto.INT32, total_seq_len_shape),
    helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT, cache_shape),
    helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT, cache_shape),
]

# Define outputs
output_infos = [
    helper.make_tensor_value_info('output', TensorProto.FLOAT, query_shape),
    helper.make_tensor_value_info('present_key', TensorProto.FLOAT, past_shape),
    helper.make_tensor_value_info('present_value', TensorProto.FLOAT, past_shape),
]

# Create the GroupQueryAttention node
gqa_node = helper.make_node(
    'GroupQueryAttention',
    inputs=['query', 'key', 'value', 'past_key', 'past_value', 'seqlens_k', 'total_sequence_length', 'cos_cache', 'sin_cache'],
    outputs=['output', 'present_key', 'present_value'],
    domain='com.microsoft',
    name='GQA_Node',
    do_rotary=1,
    kv_num_heads=kv_num_heads,
    local_window_size=-1,
    num_heads=num_heads,
    rotary_interleaved=0,
    scale=0.25,
    softcap=0.0
)

# Create the graph
graph_def = helper.make_graph(
    [gqa_node],
    'gqa-test-model',
    input_infos,
    output_infos
)

# Create the model with domain imports
opset_imports = [
    helper.make_opsetid("", 14),
    helper.make_opsetid("com.microsoft", 1)
]
model_def = helper.make_model(graph_def, producer_name='onnx-gqa-example', opset_imports=opset_imports)

# 2. Convert model to string (bytes)
model_str = model_def.SerializeToString()

# 3. Prepare input data
np.random.seed(0)
input_feed = {
    'query': np.random.rand(*query_shape).astype(np.float32),
    'key': np.random.rand(*key_shape).astype(np.float32),
    'value': np.random.rand(*value_shape).astype(np.float32),
    'past_key': np.random.rand(*past_shape).astype(np.float32),
    'past_value': np.random.rand(*past_shape).astype(np.float32),
    'seqlens_k': np.array([past_sequence_length] * batch_size, dtype=np.int32),
    'total_sequence_length': np.array(past_sequence_length + sequence_length, dtype=np.int32),
    'cos_cache': np.random.rand(*cache_shape).astype(np.float32),
    'sin_cache': np.random.rand(*cache_shape).astype(np.float32),
}

# 4. Run on CPUExecutionProvider
sess_cpu = ort.InferenceSession(model_str, providers=['CPUExecutionProvider'])
res_cpu = sess_cpu.run(['output', 'present_key', 'present_value'], input_feed)
print("CPU Result Output Shape:", res_cpu[0].shape)

# 5. Run on WebGpuExecutionProvider
try:
    sess2 = ort.InferenceSession(model_str, providers=['CPUExecutionProvider'])
    res2 = sess2.run(['output', 'present_key', 'present_value'], input_feed)

    # Compare results (Output tensor)
    print(f'{res_cpu[0].mean().item()=}')
    print(f'{res2[0].mean().item()=}')
    diff = np.abs(res_cpu[0] - res2[0])
    max_diff = diff.max().item()
    print(f"Max diff output: {max_diff}")

    if max_diff < 1e-3:
        print("Results match!")
    else:
        print("Results do not match within tolerance.")

except Exception as e:
    print(f"WebGPU execution failed or not available: {e}")
