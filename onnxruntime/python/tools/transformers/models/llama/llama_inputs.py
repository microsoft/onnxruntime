from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import PretrainedConfig

from onnxruntime import InferenceSession, OrtValue


# Get position_ids from attention_mask
def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        # Shape: (batch_size, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)

    # Shape: (batch_size, sequence_length)
    return position_ids


# Inputs for first pass to get initial past_key_values
#   input_ids: (batch_size, sequence_length)
#   attention_mask: (batch_size, sequence_length)
#   position_ids: (batch_size, sequence_length)
def get_sample_inputs(
    config: PretrainedConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    engine: str = "pt",
    return_dict: bool = False,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, use_past_kv=False)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)

    if not return_dict:
        # For export
        return (input_ids, attention_mask, position_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    return inputs


# Inputs for subsequent passes with past_key_values
#   input_ids: (batch_size, 1)
#   attention_mask: (batch_size, past_sequence_length + 1)
#   position_ids: (batch_size, 1)
#   past_key: (batch_size, num_heads, past_sequence_length, head_size)
#   past_value: (batch_size, num_heads, past_sequence_length, head_size)
def get_sample_with_past_kv_inputs(
    config: PretrainedConfig,
    device: torch.device,
    batch_size: int,
    past_seq_len: int,
    use_fp16: bool = False,
    engine: str = "pt",
    return_dict: bool = False,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 1), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, dtype=torch.int64)
    # position_ids is of shape (batch_size, 1)
    position_ids = get_position_ids(attention_mask, use_past_kv=True)
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)
    past_kv = (
        flatten_past_kv_inputs(past_kv)
        if engine == "ort"
        else list(map(lambda kv: (kv[0].to(device), kv[1].to(device)), past_kv))
    )

    if not return_dict:
        # For export
        assert isinstance(past_kv, list)
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if engine == "ort":
        assert isinstance(past_kv, dict)
        inputs.update(past_kv)
    else:
        assert isinstance(past_kv, list)
        inputs["past_key_values"] = past_kv

    return inputs


# Inputs for all passes with past_key_values
#   input_ids: (batch_size, sequence_length)
#   attention_mask: (batch_size, past_sequence_length + sequence_length)
#   position_ids: (batch_size, sequence_length)
#   past_key: (batch_size, num_heads, kv_sequence_length, head_size)
#      For models with GQA, kv_sequence_length = max_sequence_length
#      For models without GQA, kv_sequence_length = past_sequence_length
#   past_value: (batch_size, num_heads, kv_sequence_length, head_size)
#      For models with GQA, kv_sequence_length = max_sequence_length
#      For models without GQA, kv_sequence_length = past_sequence_length
def get_merged_sample_with_past_kv_inputs(
    config: PretrainedConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    max_seq_len: int,
    use_fp16: bool = False,
    engine: str = "pt",
    return_dict: bool = False,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)
    past_kv = (
        flatten_past_kv_inputs(past_kv)
        if engine == "ort"
        else list(map(lambda kv: (kv[0].to(device), kv[1].to(device)), past_kv))
    )

    if not return_dict:
        # For export
        assert isinstance(past_kv, list)
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if engine == "ort":
        assert isinstance(past_kv, dict)
        inputs.update(past_kv)

        if use_fp16:  # If model has GQA
            del inputs["attention_mask"]
            inputs["past_sequence_length"] = np.array([past_seq_len], dtype=np.int64)
            inputs = enable_past_present_share_buffer(inputs, past_seq_len, max_seq_len)

    else:
        assert isinstance(past_kv, list)
        inputs["past_key_values"] = past_kv

    return inputs


# Inputs for Microsoft export from https://github.com/microsoft/Llama-2-Onnx
def get_msft_sample_inputs(
    config: PretrainedConfig,
    batch_size: int,
    past_seq_len: int,
    seq_len: int,
    max_seq_len: int,
    use_fp16: bool,
    split_kv: bool,
):
    np_dtype = np.float16 if use_fp16 else np.float32
    head_size = config.hidden_size // config.num_attention_heads

    if not split_kv:
        ort_inputs = {
            "x": np.random.rand(batch_size, seq_len, config.hidden_size).astype(np_dtype),
            "attn_mask": (-10000.0 * np.triu(np.ones((batch_size, max_seq_len, max_seq_len)), k=1)).astype(np_dtype),
            "k_cache": np.random.rand(
                batch_size, config.num_hidden_layers, past_seq_len, config.num_attention_heads, head_size
            ).astype(np_dtype),
            "v_cache": np.random.rand(
                batch_size, config.num_hidden_layers, past_seq_len, config.num_attention_heads, head_size
            ).astype(np_dtype),
            "pos": np.array(past_seq_len, dtype=np.int64),
        }
    else:
        ort_inputs = {
            "x": np.random.rand(batch_size, seq_len, config.hidden_size).astype(np_dtype),
            "attn_mask": (np.triu(np.ones((batch_size, max_seq_len, max_seq_len), dtype=np.int32), k=1) - 1).astype(
                np.int32
            ),
            "pos": np.array(past_seq_len, dtype=np.int64),
        }
        for i in range(config.num_hidden_layers):
            ort_inputs.update(
                {
                    f"k_{i}_cache": np.random.rand(
                        batch_size, config.num_attention_heads, past_seq_len, head_size
                    ).astype(np_dtype),
                    f"v_{i}_cache": np.random.rand(
                        batch_size, config.num_attention_heads, past_seq_len, head_size
                    ).astype(np_dtype),
                }
            )

        if use_fp16:  # If model has GQA
            del ort_inputs["attn_mask"]
            ort_inputs = enable_past_present_share_buffer(ort_inputs, past_seq_len, max_seq_len)

    return ort_inputs


# Create past_key_values
# Each is of shape (batch_size, num_heads, past_sequence_length, head_size)
def get_past_kv_inputs(config: PretrainedConfig, batch_size: int, past_seq_len: int, use_fp16: bool):
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    past_kv = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return past_kv


# Convert list of past_key_values to dict of past_key and past_value
def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
    past_kv = {}
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k.detach().cpu().numpy()
        past_kv[f"past_key_values.{i}.value"] = past_v.detach().cpu().numpy()
    return past_kv


# Format PyTorch inputs to ONNX Runtime inputs
def convert_inputs_for_ort(
    pt_inputs: dict,
    use_fp16: bool,
    use_buffer_share: bool = False,
    past_seq_len: int = 0,
    max_seq_len: int = 2048,
    device: str = "",
    device_id: int = -1,
):
    ort_inputs = {}
    for k, v in pt_inputs.items():
        if isinstance(v, np.ndarray):
            ort_inputs[k] = v
        elif k == "past_key_values":
            ort_inputs.update(flatten_past_kv_inputs(v))
        elif k == "attention_mask" and use_fp16 and use_buffer_share:
            # Skip because FP16 model has GroupQueryAttention, uses buffer sharing,
            # and GQA supports a causal mask by default

            # Instead, add the past sequence length input for GQA
            ort_inputs["past_sequence_length"] = np.array([past_seq_len], dtype=np.int64)
        else:
            ort_inputs[k] = v.detach().cpu().numpy()

    # Reshape kv caches if using past-present-share-buffer
    if use_buffer_share and device != "" and device != "cpu" and device_id > -1:
        ort_inputs = enable_past_present_share_buffer(ort_inputs, past_seq_len, max_seq_len)

    return ort_inputs


def enable_past_present_share_buffer(ort_inputs: dict, past_seq_len: int, max_seq_len: int):
    for k, v in ort_inputs.items():
        # Allocate new buffers with max_sequence_length for GQA
        if "cache" in k or "past_key_values" in k:
            # Copy v (BxSxPxH) into new_v (BxSxMxH)
            batch_size, num_heads, _, head_size = v.shape
            new_v = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
            new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
            ort_inputs[k] = new_v
    return ort_inputs


# Add IO bindings for execution providers
def add_io_bindings(model: InferenceSession, ort_inputs: dict, device: str, device_id: int, kv_cache_ortvalues: dict):
    use_fp16 = False
    io_binding = model.io_binding()

    for k, v in ort_inputs.items():
        # Detect if model is in FP16
        if v.dtype == np.float16:
            use_fp16 = True

        # Bind OrtValue inputs to device
        if use_fp16 and ("cache" in k or "past_key_values" in k):
            if k not in kv_cache_ortvalues:
                v_device = OrtValue.ortvalue_from_numpy(v, device_type=device, device_id=device_id)
                io_binding.bind_ortvalue_input(k, v_device)
                kv_cache_ortvalues[k] = v_device
            else:
                kv_cache_ortvalues[k].update_inplace(v)
                io_binding.bind_ortvalue_input(k, kv_cache_ortvalues[k])
        else:
            v_device = OrtValue.ortvalue_from_numpy(v, device_type=device, device_id=device_id)
            io_binding.bind_ortvalue_input(k, v_device)

    for output in model.get_outputs():
        name = output.name
        if use_fp16 and ("out" in name or "present" in name):
            # Bind present KV cache outputs to past KV cache inputs in order to buffer share
            input_name = name.replace("out", "cache").replace("present", "past_key_values")
            io_binding.bind_ortvalue_output(name, kv_cache_ortvalues[input_name])
        else:
            io_binding.bind_output(name, device_type=device, device_id=device_id)

    return io_binding, kv_cache_ortvalues
