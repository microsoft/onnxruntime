import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto

def create_model(num_heads, kv_num_heads, head_size, seq_len, buffer_len):
    query_shape = [1, seq_len, num_heads * head_size]
    key_shape = [1, seq_len, kv_num_heads * head_size]
    value_shape = [1, seq_len, kv_num_heads * head_size]
    past_shape = [1, kv_num_heads, buffer_len, head_size]
    cache_shape = [buffer_len, head_size // 2]

    node = helper.make_node(
        'GroupQueryAttention',
        inputs=['query', 'key', 'value', 'past_key', 'past_value', 'seqlens_k', 'total_sequence_length', 'cos_cache', 'sin_cache'],
        outputs=['output', 'present_key', 'present_value'],
        domain='com.microsoft',
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        do_rotary=1,
    )

    graph = helper.make_graph(
        [node], 'test',
        [
            helper.make_tensor_value_info('query', TensorProto.FLOAT, query_shape),
            helper.make_tensor_value_info('key', TensorProto.FLOAT, key_shape),
            helper.make_tensor_value_info('value', TensorProto.FLOAT, value_shape),
            helper.make_tensor_value_info('past_key', TensorProto.FLOAT, past_shape),
            helper.make_tensor_value_info('past_value', TensorProto.FLOAT, past_shape),
            helper.make_tensor_value_info('seqlens_k', TensorProto.INT32, [1]),
            helper.make_tensor_value_info('total_sequence_length', TensorProto.INT32, [1]),
            helper.make_tensor_value_info('cos_cache', TensorProto.FLOAT, cache_shape),
            helper.make_tensor_value_info('sin_cache', TensorProto.FLOAT, cache_shape),
        ],
        [
            helper.make_tensor_value_info('output', TensorProto.FLOAT, query_shape),
            helper.make_tensor_value_info('present_key', TensorProto.FLOAT, past_shape),
            helper.make_tensor_value_info('present_value', TensorProto.FLOAT, past_shape),
        ]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)])
    return model.SerializeToString()

num_heads = 64
kv_num_heads = 1
head_size = 32
seq_len = 128
buffer_len = 256

model_str = create_model(num_heads, kv_num_heads, head_size, seq_len, buffer_len)
sess = ort.InferenceSession(model_str, providers=['CPUExecutionProvider'])

np.random.seed(0)
feeds = {
    'query': np.random.randn(1, seq_len, num_heads * head_size).astype(np.float32),
    'key': np.random.randn(1, seq_len, kv_num_heads * head_size).astype(np.float32),
    'value': np.random.randn(1, seq_len, kv_num_heads * head_size).astype(np.float32),
    'past_key': np.random.randn(1, kv_num_heads, buffer_len, head_size).astype(np.float32),
    'past_value': np.random.randn(1, kv_num_heads, buffer_len, head_size).astype(np.float32),
    'seqlens_k': np.array([seq_len - 1], dtype=np.int32),
    'total_sequence_length': np.array([seq_len], dtype=np.int32),
    'cos_cache': np.random.randn(buffer_len, head_size // 2).astype(np.float32),
    'sin_cache': np.random.randn(buffer_len, head_size // 2).astype(np.float32),
}

results = []
for i in range(10):
    out = sess.run(None, feeds)[0]
    results.append(out)
    print(f"Run {i} mean: {out.mean()}")

diffs = [np.abs(results[i] - results[0]).max() for i in range(1, 10)]
max_diff = max(diffs)
print(f"Max diff: {max_diff}")
if max_diff > 1e-6:
    print("FAILED: Non-deterministic results!")
else:
    print("PASSED: Deterministic results.")
