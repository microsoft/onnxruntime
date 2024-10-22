# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import torch
from transformers import WhisperConfig
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Create audio_features for encoder
# Shape is (batch_size, feature_size, sequence_length) = (batch_size, num_mel_filters, num_frames)
# where num_mel_filters is a model attribute and num_frames = (chunk_length * sample_rate) // hop_length.
#
# Hard-coded audio hyperparameters:
# SAMPLE_RATE = 16000
# N_FFT = 400
# HOP_LENGTH = 160
# CHUNK_LENGTH = 30  (i.e. 30-second chunk of audio)
# N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE = 30 * 16000 = 480000  (i.e. 480,000 samples in a 30-second chunk of audio)
# N_FRAMES = N_SAMPLES // HOP_LENGTH = 480000 // 160 = 3000  (i.e. 3000 frames in a mel spectrogram input)
#
# N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 = 160 * 2 = 320
# FRAMES_PER_TOKEN = SAMPLE_RATE // HOP_LENGTH = 16000 // 160 = 100  (i.e. 10 ms per audio frame)
# TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN = 16000 // 320 = 50  (i.e. 20 ms per audio token)
def get_sample_audio_features(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int = 3000,
    use_fp16: bool = False,
):
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    audio_features = torch.randn(batch_size, config.num_mel_bins, sequence_length, device=device, dtype=torch_dtype)
    return audio_features

# Create input_ids for decoder
# Shape is (batch_size, sequence_length) where sequence_length is the initial decoder sequence length
def get_sample_decoder_input_ids(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    use_int32: bool = True,
):
    torch_dtype = torch.int32 if use_int32 else torch.int64
    decoder_input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, sequence_length), device=device, dtype=torch_dtype)
    return decoder_input_ids

# Create encoder_hidden_states for decoder-init
# Shape is (batch_size, num_frames // 2, hidden_size)
def get_sample_encoder_hidden_states(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    use_fp16: bool = False,
):
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    encoder_hidden_states = torch.randn(batch_size, config.max_source_positions, config.d_model, device=device, dtype=torch_dtype)
    return encoder_hidden_states

# Create past_key_values
# Self-attention KV caches are of shape (batch_size, num_heads, past_sequence_length, head_size)
# Cross-attention KV caches are of shape (batch_size, num_heads, num_frames // 2, head_size)
def get_sample_past_key_values(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    past_seq_len: int,
    use_fp16: bool = False,
):
    num_heads = config.decoder_attention_heads
    head_size = config.d_model // num_heads
    max_source_positions = config.max_source_positions  # equal to num_frames // 2 = encoder's sequence_length // 2 = 3000 // 2 = 1500
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    self_attention_kv_caches = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    cross_attention_kv_caches = [
        (
            torch.rand(batch_size, num_heads, max_source_positions, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, max_source_positions, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return group_past_key_values(self_attention_kv_caches, cross_attention_kv_caches)
    # return flatten_past_key_values(self_attention_kv_caches, cross_attention_kv_caches)

# Group KV caches into pairs-of-4 where each pair is defined as:
# (self_attn_key_cache, self_attn_value_cache, cross_attn_key_cache, cross_attn_value_cache)
def group_past_key_values(
    self_attn_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    cross_attn_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
):
    past_key_values = []
    for (self_k_cache, self_v_cache), (cross_k_cache, cross_v_cache) in zip(self_attn_kv_caches, cross_attn_kv_caches):
        layer_kv_caches = (self_k_cache, self_v_cache, cross_k_cache, cross_v_cache)
        past_key_values.append(layer_kv_caches)
    return past_key_values

# Flatten KV caches into a 1D list where the list is defined as:
# [past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...] + 
# [past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...]
def flatten_past_key_values(
    self_attn_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    cross_attn_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
):
    past_key_values = []
    for (self_k_cache, self_v_cache) in self_attn_kv_caches:
        past_key_values.append(self_k_cache)
        past_key_values.append(self_v_cache)
    for (cross_k_cache, cross_v_cache) in cross_attn_kv_caches:
        past_key_values.append(cross_k_cache)
        past_key_values.append(cross_v_cache)
    return past_key_values

# Create inputs for encoder component of Whisper
def get_sample_encoder_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int = 3000,
    use_fp16: bool = False,
):
    audio_features = get_sample_audio_features(config, device, batch_size, sequence_length, use_fp16)
    return {"audio_features": audio_features}

# # Create inputs for first pass through decoder component of Whisper
# def get_sample_decoder_init_inputs(
#     config: WhisperConfig,
#     device: torch.device,
#     batch_size: int,
#     sequence_length: int,
#     use_int32: bool = True,
#     use_fp16: bool = False,
# ):
#     decoder_input_ids = get_sample_decoder_input_ids(config, device, batch_size, sequence_length, use_int32)
#     encoder_hidden_states = get_sample_encoder_hidden_states(config, device, batch_size, use_fp16)
#     return {"decoder_input_ids": decoder_input_ids, "encoder_hidden_states": encoder_hidden_states}

# Create inputs for encoder component + first pass through decoder component of Whisper
def get_sample_encoder_decoder_init_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    decoder_sequence_length: int,
    encoder_sequence_length: int = 3000,
    use_fp16: bool = False,
    use_int32: bool = True,
):
    audio_features = get_sample_audio_features(config, device, batch_size, encoder_sequence_length, use_fp16)
    decoder_input_ids = get_sample_decoder_input_ids(config, device, batch_size, decoder_sequence_length, use_int32)
    return {"audio_features": audio_features, "decoder_input_ids": decoder_input_ids}

# Create inputs for decoder component of Whisper
# Inputs for first pass through the decoder (i.e. decoder-init): decoder_input_ids, encoder_hidden_states
# Inputs for subsequent passes through the decoder (i.e. decoder-with-past): decoder_input_ids, past_key_values
def get_sample_decoder_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    past_sequence_length: int,
    sequence_length: int,
    use_fp16: bool = False,
    use_int32: bool = True,
):
    decoder_input_ids = get_sample_decoder_input_ids(config, device, batch_size, sequence_length, use_int32)
    encoder_hidden_states = get_sample_encoder_hidden_states(config, device, batch_size, use_fp16)
    past_key_values = get_sample_past_key_values(config, device, batch_size, past_sequence_length, use_fp16)
    return {"decoder_input_ids": decoder_input_ids, "encoder_hidden_states": encoder_hidden_states, "past_key_values": past_key_values}

# Get dynamic axes for all inputs and outputs to the model
def get_model_dynamic_axes(
    config: WhisperConfig,
    input_names: List[str],
    output_names: List[str],
):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"audio_features", "encoder_input_ids"}:
            # shape is (batch_size, num_mels, num_frames)
            dynamic_axes[name] = {0: "batch_size"}
        elif name in {"input_ids", "decoder_input_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "encoder_hidden_states":
            # shape is (batch_size, num_frames // 2, hidden_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "past_key_self" in name or "past_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif "present_key_self" in name or "present_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size),
            # which is equal to (batch_size, num_heads, total_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
        elif "past_key_cross" in name or "past_value_cross" in name or "present_key_cross" in name or "present_value_cross" in name:
            # shape is (batch_size, num_heads, num_frames // 2, head_size)
            dynamic_axes[name] = {0: "batch_size"}
        else:
            raise Exception(f"Unknown input or output name found: {name}")
    return dynamic_axes
