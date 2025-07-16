// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if USE_MEMORY_EFFICIENT_ATTENTION

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kEfficientAttentionMaxHeadSize = 1024;

struct MemoryEfficientAttentionParams {
  int32_t sm = 50;
  bool is_half = false;
  bool is_kv_bsnh = true;
  int32_t batch_size = 0;
  int32_t num_heads = 0;
  int32_t sequence_length = 0;
  int32_t kv_sequence_length = 0;
  int32_t max_sequence_length = 0;
  int32_t qk_head_size = 0;
  int32_t v_head_size = 0;
  int32_t local_window_size = -1;
  bool causal = false;
  bool use_smooth_softmax = false;
  bool broadcast_attn_bias_dim_0 = false;
  bool broadcast_attn_bias_dim_1 = false;
  bool has_custom_right_padding = false;
  float scale = 1.0f;
  float softcap = 0.0;

  cudaStream_t stream = nullptr;
  const int32_t* seqstart_q_ptr = nullptr;  // [B + 1], cumulated sequence lengths of queries
  const int32_t* seqstart_k_ptr = nullptr;  // [B + 1], cumulated sequence lengths of keys
  const int32_t* seqlen_k_ptr = nullptr;    // [B], sequence lengths of keys
  const void* query = nullptr;              // [B, S, N, H]
  const void* key = nullptr;                // [B, L, N, H], where L is kv_sequence_length
  const void* value = nullptr;              // [B, L, N, H_v]
  const void* attn_bias = nullptr;          // [B or 1, N or 1, S, L] or null
  void* workspace = nullptr;                // [B, S, N, H_v] when kNeedsOutputAccumulatorBuffer, nullptr otherwise
  void* output = nullptr;                   // [B, S, N, H_v]

  static bool need_workspace(size_t v_head_size, bool is_float) {
    return (v_head_size > 128 && !is_float);
  }
};

void run_memory_efficient_attention(const MemoryEfficientAttentionParams& params);

inline bool has_memory_efficient_attention(int32_t sm, bool is_half, int qk_head_size, int v_head_size) {
  return sm >= (is_half ? 53 : 50) &&
         (qk_head_size & 7) == 0 &&
         (v_head_size & 7) == 0 &&
         qk_head_size <= kEfficientAttentionMaxHeadSize && v_head_size <= kEfficientAttentionMaxHeadSize;
}

void run_memory_efficient_attention_sm80(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm75(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm70(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm50(const MemoryEfficientAttentionParams& params);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
