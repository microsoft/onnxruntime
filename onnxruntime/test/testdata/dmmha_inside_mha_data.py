import numpy as np

import onnxruntime as ort
from onnxruntime import OrtValue

np.random.seed(0)


# Whisper decoder self attention with past_kv, present_kv, buffer sharing enabled, mask, and bias
# Used in decoder-with-past's self-attention layers
# For CUDA, K caches are transposed and reshaped from 4D to 5D for DecoderMaskedMultiHeadAttention
# See onnxruntime/core/graph/contrib_ops/bert_defs.cc for more details
def dmmha_inside_mha_self_attn():
    batch_size, num_heads, head_size = 2, 2, 32
    hidden_size = num_heads * head_size
    past_sequence_length, sequence_length, max_sequence_length = 4, 1, 6
    num_beams = 1
    device = "cuda"

    inputs = {
        "q": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "k": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "v": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "b": np.random.randn(hidden_size * 3).astype(np.float32),
        "past_k": np.zeros((batch_size, num_heads, max_sequence_length, head_size)).astype(np.float32),
        "past_v": np.zeros((batch_size, num_heads, max_sequence_length, head_size)).astype(np.float32),
        "past_seq_len": np.array([past_sequence_length]).astype(np.int32),
        "cache_indir": np.zeros((batch_size, num_beams, max_sequence_length)).astype(np.int32),
    }
    inputs["past_k"][:batch_size, :num_heads, :past_sequence_length, :head_size] = np.random.randn(
        batch_size, num_heads, past_sequence_length, head_size
    ).astype(np.float32)
    inputs["past_v"][:batch_size, :num_heads, :past_sequence_length, :head_size] = np.random.randn(
        batch_size, num_heads, past_sequence_length, head_size
    ).astype(np.float32)
    print_vals(inputs)

    sess = ort.InferenceSession("dmmha_inside_mha_self_attn.onnx", providers=[f"{device.upper()}ExecutionProvider"])
    io_binding = sess.io_binding()
    past_k_ortvalue, past_v_ortvalue = None, None
    for k, v in inputs.items():
        v_device = OrtValue.ortvalue_from_numpy(v, device_type=device.lower(), device_id=0)
        io_binding.bind_ortvalue_input(k, v_device)
        if k == "past_k":
            past_k_ortvalue = v_device
        elif k == "past_v":
            past_v_ortvalue = v_device
    for output in sess.get_outputs():
        name = output.name
        if name == "present_k":
            io_binding.bind_ortvalue_output(name, past_k_ortvalue)
        elif name == "present_v":
            io_binding.bind_ortvalue_output(name, past_v_ortvalue)
        else:
            io_binding.bind_output(name, device_type=device.lower(), device_id=0)

    sess.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()

    print_vals(outputs)


# Whisper decoder self attention with past_kv, present_kv, buffer sharing enabled, mask, and bias
# Used in decoder-with-past's self-attention layers
# For CUDA, K caches are transposed and reshaped from 4D to 5D for DecoderMaskedMultiHeadAttention
# See onnxruntime/core/graph/contrib_ops/bert_defs.cc for more details
def dmmha_self_attn():
    batch_size, num_heads, head_size = 2, 2, 32
    hidden_size = num_heads * head_size
    past_sequence_length, sequence_length, max_sequence_length = 4, 1, 6
    num_beams = 1
    device = "cuda"

    inputs = {
        "q": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "k": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "v": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "b": np.random.randn(hidden_size * 3).astype(np.float32),
        "past_k": np.zeros((batch_size, num_heads, max_sequence_length, head_size)).astype(np.float32),
        "past_v": np.zeros((batch_size, num_heads, max_sequence_length, head_size)).astype(np.float32),
        "past_seq_len": np.array([past_sequence_length]).astype(np.int32),
        "beam_width": np.array([num_beams]).astype(np.int32),
        "cache_indir": np.zeros((batch_size, num_beams, max_sequence_length)).astype(np.int32),
    }
    inputs["past_k"][:batch_size, :num_heads, :past_sequence_length, :head_size] = np.random.randn(
        batch_size, num_heads, past_sequence_length, head_size
    ).astype(np.float32)
    inputs["past_v"][:batch_size, :num_heads, :past_sequence_length, :head_size] = np.random.randn(
        batch_size, num_heads, past_sequence_length, head_size
    ).astype(np.float32)
    print_vals(inputs)

    sess = ort.InferenceSession("dmmha_self_attn.onnx", providers=[f"{device.upper()}ExecutionProvider"])
    io_binding = sess.io_binding()
    past_k_ortvalue, past_v_ortvalue = None, None
    for k, v in inputs.items():
        v_device = OrtValue.ortvalue_from_numpy(v, device_type=device.lower(), device_id=0)
        io_binding.bind_ortvalue_input(k, v_device)
        if k == "past_k":
            past_k_ortvalue = v_device
        elif k == "past_v":
            past_v_ortvalue = v_device
    for output in sess.get_outputs():
        name = output.name
        if name == "present_k":
            io_binding.bind_ortvalue_output(name, past_k_ortvalue)
        elif name == "present_v":
            io_binding.bind_ortvalue_output(name, past_v_ortvalue)
        else:
            io_binding.bind_output(name, device_type=device.lower(), device_id=0)

    sess.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()

    print_vals(outputs)


# Whisper decoder cross attention with past_kv used directly as K and V, no mask, and bias
# Used in decoder-with-past's cross-attention layers
def dmmha_inside_mha_cross_attn():
    batch_size, num_heads, head_size = 2, 2, 32
    hidden_size = num_heads * head_size
    past_sequence_length, sequence_length, kv_sequence_length, max_sequence_length = 4, 1, 10, 6
    num_beams = 1
    device = "cuda"

    inputs = {
        "q": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "k": np.random.randn(batch_size, num_heads, kv_sequence_length, head_size).astype(np.float32),
        "v": np.random.randn(batch_size, num_heads, kv_sequence_length, head_size).astype(np.float32),
        "b": np.zeros(hidden_size * 3).astype(np.float32),
        "past_seq_len": np.array([past_sequence_length]).astype(np.int32),
        "cache_indir": np.zeros((batch_size, num_beams, max_sequence_length)).astype(np.int32),
    }
    inputs["b"][:hidden_size] = np.random.randn(hidden_size).astype(np.float32)
    print_vals(inputs)

    sess = ort.InferenceSession("dmmha_inside_mha_cross_attn.onnx", providers=[f"{device.upper()}ExecutionProvider"])
    outputs = sess.run(None, inputs)

    print_vals(outputs)


# Whisper decoder cross attention with past_kv used directly as K and V, no mask, and bias
# Used in decoder-with-past's cross-attention layers
def dmmha_cross_attn():
    batch_size, num_heads, head_size = 2, 2, 32
    hidden_size = num_heads * head_size
    past_sequence_length, sequence_length, kv_sequence_length, max_sequence_length = 4, 1, 10, 6
    num_beams = 1
    device = "cuda"

    inputs = {
        "q": np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32),
        "k": np.random.randn(batch_size, num_heads, kv_sequence_length, head_size).astype(np.float32),
        "v": np.random.randn(batch_size, num_heads, kv_sequence_length, head_size).astype(np.float32),
        "b": np.zeros(hidden_size * 3).astype(np.float32),
        "past_seq_len": np.array([past_sequence_length]).astype(np.int32),
        "beam_width": np.array([num_beams]).astype(np.int32),
        "cache_indir": np.zeros((batch_size, num_beams, max_sequence_length)).astype(np.int32),
    }
    inputs["b"][:hidden_size] = np.random.randn(hidden_size).astype(np.float32)
    print_vals(inputs)

    sess = ort.InferenceSession("dmmha_cross_attn.onnx", providers=[f"{device.upper()}ExecutionProvider"])
    outputs = sess.run(None, inputs)

    print_vals(outputs)


# Print values in format for onnxruntime/test/testdata/attention/attention_test_data.txt
def print_vals(L):
    if isinstance(L, list):
        for idx, elm in enumerate(L):
            print(f"\nOutput {idx}:", flush=True)
            for i, entry in enumerate(elm.flatten()):
                print(entry, end=",", flush=True)
                if i % 8 == 0 and i != 0:
                    print("\n", end="", flush=True)
    elif isinstance(L, dict):
        for key, val in L.items():
            print(f"\n{key}:", flush=True)
            for i, entry in enumerate(val.flatten()):
                print(entry, end=",", flush=True)
                if i % 8 == 0 and i != 0:
                    print("\n", end="", flush=True)

    print("\n=====================================================", flush=True)


dmmha_inside_mha_self_attn()
dmmha_inside_mha_cross_attn()

dmmha_self_attn()
dmmha_cross_attn()
