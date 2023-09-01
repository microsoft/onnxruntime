from typing import List, Tuple

import numpy as np
import torch
from transformers import LlamaConfig


# Get position_ids from attention_mask
def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
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


# Create past_key_values
def get_sample_past_kv_inputs(
    config: LlamaConfig, device: torch.device, batch_size: int, past_seq_len: int, use_fp16: bool
):
    num_heads, head_size = config.num_attention_heads, config.hidden_size // config.num_attention_heads
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
def convert_inputs_for_ort(pt_inputs: dict, use_fp16: bool):
    ort_inputs = {}
    for k, v in pt_inputs.items():
        if k == "past_key_values":
            ort_inputs.update(flatten_past_kv_inputs(v, use_fp16))
        else:
            ort_inputs[k] = v.detach().cpu().numpy()
    return ort_inputs


# Inputs for Microsoft export from https://github.com/microsoft/Llama-2-Onnx
def get_msft_sample_inputs(config: LlamaConfig, batch_size: int, past_seq_len: int, seq_len: int, use_fp16: bool):
    np_dtype = np.float16 if use_fp16 else np.float32
    head_size = config.hidden_size // config.num_attention_heads
    max_seq_len = 2048

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
    return ort_inputs
