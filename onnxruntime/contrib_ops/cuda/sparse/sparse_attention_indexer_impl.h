// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Everything the device code needs to know about a SparseAttentionIndexer call. All of it is
// derived from attributes and input shapes, so no device data is ever read on the host.
struct SparseAttentionIndexerParams {
  int batch_size = 0;
  int sequence_length = 0;
  int num_heads = 0;
  int head_size = 0;
  int rotary_width = 0;       // cos_cache.shape[2]
  int max_rotary_length = 0;  // cos_cache.shape[1]
  int compress_ratio = 0;
  int capacity = 0;  // selected_indices.shape[2]
  float epsilon = 1e-6f;
  float scale = 0.0f;

  // policy_mode = "qsa"
  int past_sequence_length = 0;
  int total_sequence_length = 0;
  int max_block_count = 0;  // total_sequence_length / compress_ratio
  int block_topk = 0;       // token_budget / compress_ratio

  // policy_mode = "csa"
  int past_compressed_length = 0;
  int present_compressed_length = 0;
  int past_buffer_length = 0;
  int overlap_length = 0;
  int new_window_count = 0;
  int present_buffer_length = 0;
  int present_buffer_start = 0;
  int index_topk = 0;
  float head_weight_scale = 0.0f;
};

// Scratch requirements, in elements.
size_t GetQsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params);
size_t GetQsaWorkspaceIntCount(const SparseAttentionIndexerParams& params);
size_t GetCsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params);

template <typename T>
Status LaunchQsaSparseAttentionIndexer(
    cudaStream_t stream,
    const SparseAttentionIndexerParams& params,
    const T* query,
    const T* key,
    const T* key_norm_weight,
    const T* cos_cache,
    const T* sin_cache,
    const bool* mask,
    const T* past_key,
    int32_t* selected_indices,
    T* present_key,
    float* float_workspace,
    int32_t* int_workspace);

template <typename T>
Status LaunchCsaSparseAttentionIndexer(
    cudaStream_t stream,
    const SparseAttentionIndexerParams& params,
    const T* query,
    const T* key,
    const T* key_norm_weight,
    const T* cos_cache,
    const T* sin_cache,
    const T* gate,
    const T* position_bias,
    const T* head_weights,
    const int64_t* position_ids,
    const T* past_compressed_key,
    const T* past_kv_buffer,
    const T* past_gate_buffer,
    int32_t* selected_indices,
    T* present_compressed_key,
    T* present_kv_buffer,
    T* present_gate_buffer,
    float* float_workspace);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
