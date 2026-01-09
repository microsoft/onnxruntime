// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <iostream>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct AttentionData {
  T* gemm_buffer = nullptr;
  const T* bias = nullptr;
  int* seqlens_k_total = nullptr;

  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const int* mask_index = nullptr;
  gsl::span<const int64_t> mask_index_dims;
  const T* past = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const int32_t* cache_indirection = nullptr;
  const T* attention_bias = nullptr;

  bool has_qkv_workspace = false;
  T* workspace = nullptr;

  T* output = nullptr;
  T* present = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
  void* output_qk = nullptr;

  void* fused_runner = nullptr;
  const void* fused_cross_attention_kernel = nullptr;

  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
  bool use_decoder_masked_multihead_attention = false;

  const int32_t* cumulated_sequence_length_q_cache = nullptr;
  const int32_t* cumulated_sequence_length_kv_cache = nullptr;

  // Intermediate data
  T* q = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  T* scratch = nullptr;
  AttentionQkvFormat qkv_format = AttentionQkvFormat::UNKNOWN;

  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;

  // Flash Atttention and Lean Attention
  int num_splits;

  // Lean Attention
  bool use_lean_attention = false;
#if USE_LEAN_ATTENTION
  int grid_dim_z = 0;
  int max_tiles_per_tb = 0;
  int high_load_tbs = 0;
  int tiles_per_head = 0;
  int* lean_sync_flag = nullptr;
#endif

  // For Debugging
  size_t workspace_bytes = 0;
  bool allow_debug_info = false;

  // For MultiHeadAttention only.
  AttentionKernelType kernel_type = AttentionKernelType::AttentionKernel_Default;
  AllocatorPtr allocator = nullptr;
  bool IsUnfused() const {
    return kernel_type == AttentionKernelType::AttentionKernel_Unfused;
  }

  // For DecoderMaskedMultiHeadAttention
  T* q_bias = nullptr;
  T* k_bias = nullptr;
  T* v_bias = nullptr;

  void PrintDebugInfo() const {
    std::cout << "flash=" << use_flash_attention
              << ", lean=" << use_lean_attention
              << ", efficient=" << use_memory_efficient_attention
              << ", fused_runner=" << (fused_runner != nullptr)
              << ", fused_cross=" << (fused_cross_attention_kernel != nullptr)
              << ", bias=" << (bias != nullptr)
              << ", attn_bias=" << (attention_bias != nullptr)
              << ", mask_dims=" << mask_index_dims.size()
              << ", has_qkv_workspace=" << has_qkv_workspace
              << ", workspace=" << workspace_bytes
              << ", past=" << (past != nullptr ? 1 : (past_key != nullptr ? 2 : 0))
              << ", present=" << (present != nullptr ? 1 : (present_key != nullptr ? 2 : 0))
              << std::endl;
  }
};

template <typename T>
struct PackedAttentionData {
  T* gemm_buffer;
  const T* bias;
  const T* attention_bias;
  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  T* workspace;
  T* output;

  void* fused_runner;

  bool use_memory_efficient_attention;
};

template <typename T>
struct PackedMultiHeadAttentionData {
  const T* query;
  const T* key;
  const T* value;
  const T* bias;
  const T* attention_bias;

  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  AttentionQkvFormat source_qkv_format;

  bool no_qkv_workspace;
  T* workspace;
  T* output;

  void* fused_runner;

  bool use_flash_attention;
  bool use_memory_efficient_attention;
};

template <typename T>
struct GroupQueryAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  int* seqlens_k = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  const T* head_sink = nullptr;

  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;
  int* seqlens_k_buff = nullptr;

  // Memory Efficient buffers
  T* fmha_buffer = nullptr;
  T* unpacked_qkv_buffer = nullptr;
  T* rotary_buffer = nullptr;
  T* k = nullptr;
  T* v = nullptr;

  // Output Tensors
  T* output = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;

  // Kernel Flags
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
};

template <typename T>
struct PagedAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  T* key_cache = nullptr;
  T* value_cache = nullptr;
  const int* cumulative_seqlens_q = nullptr;
  const int* past_seqlens = nullptr;
  const int* block_table = nullptr;
  const int* slot_mappings = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;

  // Flash buffers
  T* softmax_lse = nullptr;
  int* cumulative_seqlens_kv = nullptr;  // Flash api takes cumulative sequence length for kv-cache

  // Fused op buffers
  T* workspace_buffer = nullptr;

  // Output Tensors
  T* output = nullptr;

  // Kernel Flags
  bool use_flash_attention = false;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
