// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Split kv_row [batch_seq, kv_width] into key [batch_seq, head_size] and value [batch_seq, head_size].
// The key occupies kv_row[:, :head_size] and the value kv_row[:, head_size:2*head_size].
template <typename T>
Status LaunchSplitKVRowKernel(
    cudaStream_t stream,
    T* key,
    T* value,
    const T* kv_row,
    int batch_seq,
    int head_size,
    int kv_width,
    int max_threads_per_block);

// Build HCA/CSA compressor state from projected current tokens and past pending state.
// `entries` must already contain the old entries in its first old_entry_count rows.
template <typename T>
Status LaunchDeepSeekV4CompressorKernel(
    cudaStream_t stream,
    T* entries,
    T* pending_kv_out,
    T* pending_gate_out,
    T* overlap_kv_out,
    T* overlap_gate_out,
    const T* current_kv,
    const T* current_gate,
    const T* past_pending_kv,
    const T* past_pending_gate,
    const T* past_overlap_kv,
    const T* past_overlap_gate,
    const T* position_bias,
    const T* norm_weight,
    const T* cos_cache,
    const T* sin_cache,
    int batch_size,
    int sequence_length,
    int pending_token_count,
    int old_entry_count,
    int new_entry_count,
    int width,
    int head_size,
    int compress_rate,
    int rotary_dim,
    int cos_cache_width,
    float epsilon,
    bool is_csa,
    int max_threads_per_block);

// Update KV cache and compute sliding-window attention with a sink token for each batch element.
//
// For each batch element b and sequence step s (processed serially within the GPU block):
//   1. If seqlens_k[b] + s >= cache_capacity, shift the cache left by 1 slot.
//   2. Write new_key / new_value into the cache at the resulting write position.
//   3. Compute sliding-window attention over the last min(local_window_size, …) cache slots,
//      with a per-head sink logit, and accumulate the weighted value sum into `context`.
//
// Parameters
//   context         : output [B, S, num_heads, head_size] — context vectors before output projection
//   present_key     : in-out KV cache [B, 1, cache_capacity, head_size]
//   present_value   : in-out KV cache [B, 1, cache_capacity, head_size]
//   q_full          : [B, S, num_heads, head_size] — query vectors (after RoPE)
//   new_key         : [B, S, head_size]            — key vectors (after RoPE)
//   new_value       : [B, S, head_size]            — value vectors
//   attention_bias  : optional [bias_b, bias_h, bias_q, bias_k] additive bias; nullptr to skip
//   head_sink       : [1 or num_heads] — per-head sink logit (float device buffer)
//   seqlens_k       : [B] — number of valid past tokens in the KV cache for each batch element
template <typename T>
Status LaunchDeepSeekV4CacheAndAttentionKernel(
    cudaStream_t stream,
    T* context,
    float* attention_workspace,
    T* present_key,
    T* present_value,
    const T* q_full,
    const T* new_key,
    const T* new_value,
    const T* compressed_entries,
    const T* attention_bias,
    int64_t bias_b_dim,
    int64_t bias_h_dim,
    int64_t bias_q_dim,
    int64_t bias_k_dim,
    const T* head_sink,
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    int batch_size,
    int sequence_length,
    int num_heads,
    int head_size,
    int cache_capacity,
    int local_window_size,
    int compressed_entry_count,
    int compress_rate,
    int index_topk,
    int attention_mode,
    float scale,
    int head_sink_count,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
