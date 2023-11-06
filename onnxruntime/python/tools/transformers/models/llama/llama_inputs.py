from typing import List, Tuple

import numpy as np
import torch
from transformers import LlamaConfig

from onnxruntime import OrtValue


# Get position_ids from attention_mask
def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    return position_ids


# Inputs for first pass to get initial past_key_values
def get_sample_inputs(
    config: LlamaConfig, device: torch.device, batch_size: int, seq_len: int, return_dict: bool = False
):
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.int64
    )
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len)
    position_ids = get_position_ids(attention_mask, use_past_kv=False)

    if not return_dict:
        return (input_ids, attention_mask, position_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    return inputs


# Inputs for subsequent passes with past_key_values
def get_sample_with_past_kv_inputs(
    config: LlamaConfig,
    device: torch.device,
    batch_size: int,
    past_seq_len: int,
    use_fp16: bool = False,
    return_dict: bool = False,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 1), device=device, dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + 1, device=device, dtype=torch.int64)
    # position_ids is of shape (batch_size, 1)
    position_ids = get_position_ids(attention_mask, use_past_kv=True)
    past_kv = get_sample_past_kv_inputs(config, device, batch_size, past_seq_len, use_fp16)

    if not return_dict:
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_kv,
    }
    return inputs


# Inputs for all passes with past_key_values
def get_merged_sample_with_past_kv_inputs(
    config: LlamaConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    use_fp16: bool = False,
    return_dict: bool = False,
):
    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.int64
    )
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, device=device, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))
    past_kv = get_sample_past_kv_inputs(config, device, batch_size, past_seq_len, use_fp16)

    if not return_dict:
        return (input_ids, attention_mask, position_ids, past_kv)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_kv,
    }
    return inputs


# Create past_key_values
def get_sample_past_kv_inputs(
    config: LlamaConfig, device: torch.device, batch_size: int, past_seq_len: int, use_fp16: bool
):
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_key_value_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    past_kv = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return past_kv


# Convert list of past_kv to dict of past_key and past_value
def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], use_fp16: bool):
    past_kv = {}
    np_dtype = np.float16 if use_fp16 else np.float32
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k.detach().cpu().numpy().astype(np_dtype)
        past_kv[f"past_key_values.{i}.value"] = past_v.detach().cpu().numpy().astype(np_dtype)
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
            ort_inputs.update(flatten_past_kv_inputs(v, use_fp16))
        elif k == "attention_mask" and use_fp16 and use_buffer_share:
            # Skip because FP16 model has GroupQueryAttention, uses buffer sharing,
            # and GQA supports a causal mask by default

            # Instead, add the past sequence length input for GQA
            ort_inputs["past_sequence_length"] = np.array([past_seq_len], dtype=np.int64)
        else:
            ort_inputs[k] = v.detach().cpu().numpy()

    # Enable past-present-share-buffer by using device memory directly
    if use_buffer_share and device != "" and device != "cpu" and device_id > -1:
        for k, v in ort_inputs.items():
            new_v = v
            # Allocate new buffers with max_sequence_length for GQA
            if "cache" in k or "past_key_values" in k:
                # Copy v (BxSxPxH) into new_v (BxSxMxH)
                batch_size, num_heads, _, head_size = v.shape
                new_v = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
                new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
            ort_inputs[k] = OrtValue.ortvalue_from_numpy(new_v, device_type=device, device_id=device_id)

    return ort_inputs


# Inputs for Microsoft export from https://github.com/microsoft/Llama-2-Onnx
def get_msft_sample_inputs(
    config: LlamaConfig, batch_size: int, past_seq_len: int, seq_len: int, use_fp16: bool, split_kv: bool
):
    np_dtype = np.float16 if use_fp16 else np.float32
    head_size = config.hidden_size // config.num_attention_heads
    max_seq_len = 2048

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

    return ort_inputs


# Inputs for Olive-optimized model from https://github.com/microsoft/Olive/tree/user/pavignol/directml-llama-sample/examples/directml/llama_v2
# specialized for DirectML
def get_dml_sample_inputs(
    config: LlamaConfig, batch_size: int, seq_len: int, use_fp16: bool, use_cache_branch: bool
):
    np_dtype = np.float16 if use_fp16 else np.float32
    head_size = config.hidden_size // config.num_attention_heads

    ort_inputs = {
        "tokens": np.random.rand(batch_size, seq_len).astype(np.int64),
        "tokens_increment": np.random.rand(batch_size, 1).astype(np.int64),
        "position_ids": np.ones((batch_size, seq_len), dtype=np.int64),
        "position_ids_increment": np.ones((batch_size, 1), dtype=np.int64),
        "attn_mask": np.ones((batch_size, seq_len), dtype=np.int32),
        "use_cache_branch": np.ones([1], dtype=np.bool_) if use_cache_branch else np.zeros([1], dtype=np.bool_)
    }

    for layer_idx in range(config.num_hidden_layers):
        ort_inputs[f"cache.{layer_idx}.key"] = np.random.rand(
            batch_size, config.num_attention_heads, seq_len, head_size
        ).astype(np_dtype)
        ort_inputs[f"cache.{layer_idx}.value"] = np.random.rand(
            batch_size, config.num_attention_heads, seq_len, head_size
        ).astype(np_dtype)

    return ort_inputs
