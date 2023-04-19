// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if USE_FLASH_ATTENTION

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MemoryEfficientAttentionParams {
  int32_t sm;
  bool is_half;
  int32_t batch_size;
  int32_t num_heads;
  int32_t sequence_length;
  int32_t kv_sequence_length;
  int32_t qk_head_size;
  int32_t v_head_size;
  bool causal;

  int32_t* cu_seqlens_q;
  int32_t* cu_seqlens_k;

  const void* query;  // [B, S, N, H]
  const void* key;    // [B, L, N, H], where L is kv_sequence_length
  const void* value;  // [B, L, N, H_v]
  void* output;       // [B, S, N, H_v]
  void* workspace;    // [B, S, N, H_v] when kNeedsOutputAccumulatorBuffer, nullptr otherwise
  cudaStream_t stream;

  static bool need_workspace(size_t v_head_size, bool is_float) {
    return (v_head_size > 128 && !is_float);
  }
};

void run_memory_efficient_attention(const MemoryEfficientAttentionParams& params);

inline bool has_memory_efficient_attention(int32_t sm, bool is_half) {
  return sm >= (is_half ? 53 : 50);
}

void run_memory_efficient_attention_sm80(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm75(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm70(const MemoryEfficientAttentionParams& params);
void run_memory_efficient_attention_sm50(const MemoryEfficientAttentionParams& params);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_FLASH_ATTENTION
