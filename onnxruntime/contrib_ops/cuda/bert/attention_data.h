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

template <typename T, typename U>
struct GroupQueryAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const U* past_key = nullptr;
  const U* past_value = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  const T* head_sink = nullptr;

  // Optional additive attention bias, shape (batch_size or 1, num_heads or 1, sequence_length,
  // total_sequence_length). Broadcast on dims 0/1 is carried by
  // parameters.broadcast_attn_bias_dim_0/1. Only consumed by the unfused fallback path.
  const T* attention_bias = nullptr;

  // Optional per-head Q/K RMSNorm (QK-Norm) weights, shape (head_size,), shared across heads.
  // Both are non-null together (validated in the op) and trigger the fused normalization before RoPE.
  const T* q_norm_weight = nullptr;
  const T* k_norm_weight = nullptr;
  float qk_norm_epsilon = 1e-6f;

  const float* k_scale = nullptr;
  const float* v_scale = nullptr;

  // Total sequence length for each batch. It has shape [batch_size].
  int* total_seq_lens = nullptr;

  // Past sequence length for each batch (i.e., the offset to append new tokens). Shape [batch_size].
  // For first prompt: past_seq_lens[b] = 0
  // For token generation or subsequent prompt: past_seq_lens[b] = total_seq_lens[b] - sequence_length
  int* past_seq_lens = nullptr;

  // Padded sequence length for each batch. Shape [batch_size].
  // Only used for first prompt: padded_seq_lens[b] = sequence_length
  int* padded_seq_lens = nullptr;

  // Cache-relative sequence lengths, used when parameters.is_windowed_kv_cache is set. Shape [batch_size].
  // For a full-length (non-windowed) cache these simply alias past_seq_lens / total_seq_lens.
  //   cache_past_seq_lens[b]  : append offset inside the capacity-C buffer, after eviction. May be
  //                             negative on a first prompt longer than the capacity, in which case
  //                             the leading (out-of-window) tokens are skipped by the append kernel.
  //   cache_total_seq_lens[b] : number of valid cache entries after the append, i.e. min(T, C).
  //   evict_counts[b]         : number of entries D dropped from the front of the cache this step.
  int* cache_past_seq_lens = nullptr;
  int* cache_total_seq_lens = nullptr;
  int* evict_counts = nullptr;

  // Scratch used by the windowed-cache compaction shift. Sized for one KV cache:
  // batch_size * kv_num_heads * capacity * head_size elements of the storage type U.
  void* compaction_scratch = nullptr;

  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;

  // Position IDs from Input
  const int64_t* position_ids = nullptr;

  // Memory Efficient buffers
  T* fmha_buffer = nullptr;
  T* qkv_buffer = nullptr;

  T* k = nullptr;
  T* v = nullptr;

  // Output Tensors
  T* output = nullptr;
  U* present_key = nullptr;
  U* present_value = nullptr;

  // Kernel Flags
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
  bool use_flash_attention_fast_decode = false;
  bool use_xqa = false;
  // cuDNN SDPA (cudnn_frontend) path: preferred on SM>=90 for non-quantized FP16/BF16 GQA.
  bool use_cudnn_sdpa = false;
  // GQA-capable unfused fallback (issue #28195): used when Flash/MEA/XQA are all ineligible,
  // e.g. fp16 head_size > 256 with past_key, or GQA on old GPUs without MEA/Flash support.
  bool use_unfused = false;

  // XQA buffer
  void* xqa_buffer = nullptr;
  size_t xqa_buffer_bytes = 0;
  // FP32 per-head attention sink consumed by the XQA kernel (nullptr when no head_sink input).
  // Either points to a PrePack-cached buffer or to scratch that is filled at launch time.
  float* xqa_head_sink = nullptr;
  // When true, head_sink was not prepacked (e.g. dynamic/non-initializer input) and the FP16/BF16
  // head_sink must be converted to xqa_head_sink (FP32 scratch) before launching XQA.
  bool xqa_head_sink_needs_conversion = false;

  // Unfused fallback buffers (see LaunchUnfusedAttention in unfused_attention.h):
  //   unfused_q_bnsh : [B, N_q, S_q, H]   (Q transposed from BSNH to BNSH)
  //   unfused_y_bnsh : [B, N_q, S_q, H_v] (output BNSH, transposed to BSNH before leaving op)
  //   unfused_workspace: FP32 QK scratch + T softmax scratch (sized by
  //                      GetUnfusedAttentionWorkspaceSize)
  T* unfused_q_bnsh = nullptr;
  T* unfused_y_bnsh = nullptr;
  void* unfused_workspace = nullptr;

  // cuDNN SDPA path: temp-space allocator and cuDNN handle (stored as void* to avoid pulling the
  // cuDNN headers into this file; cast to cudnnHandle_t in the .cu runner).
  AllocatorPtr allocator = nullptr;
  void* cudnn_handle = nullptr;
};

// TCACHE is the element type of the paged key/value cache. It equals T for an unquantized cache and
// is int8_t / Float8E4M3FN when the cache is quantized (see PagedAttentionParameters::k_quant_type).
template <typename T, typename TCACHE = T>
struct PagedAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  TCACHE* key_cache = nullptr;
  TCACHE* value_cache = nullptr;
  // FP32 quantization scales for the paged cache: (1,) for PER_TENSOR and
  // (kv_num_heads, 1, head_size) for PER_CHANNEL. nullptr when the cache is not quantized.
  const float* k_scale = nullptr;
  const float* v_scale = nullptr;
  const int* cumulative_seqlens_q = nullptr;
  const int* past_seqlens = nullptr;
  const int* block_table = nullptr;
  // Optional explicit write slots, one per query token, into the cache viewed as
  // [num_blocks * block_size, kv_num_heads, head_size]. A value of -1 suppresses the K/V store
  // for that token (prefix cache hit / rejected speculative token). nullptr keeps the legacy
  // derived mapping (past_seqlens + position within the sequence).
  const int* slot_mapping = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  // Per-head attention sink (num_heads,). nullptr with use_smooth_softmax means a sink value of 0.
  const T* head_sink = nullptr;
  // QK-Norm weights (head_size,), shared across heads. Both are set or neither is.
  const T* q_norm_weight = nullptr;
  const T* k_norm_weight = nullptr;

  // Flash buffers. FlashAttention always emits FP32 log-sum-exp regardless of T; with
  // params.num_splits <= 1 (which mha_varlen_fwd never overrides) the varlen layout is
  // [num_heads, token_count].
  float* softmax_lse = nullptr;
  int* cumulative_seqlens_kv = nullptr;  // Flash api takes cumulative sequence length for kv-cache

  // Fused op buffers
  T* workspace_buffer = nullptr;

  // Dense KV staging buffers. Always used by the memory-efficient (CUTLASS fMHA) fallback, which
  // needs a packed-varlen [total_kv_tokens, num_heads, head_size] GQA-expanded view of the cache.
  // The FlashAttention path also uses them when the cache is quantized: Flash cannot read a
  // quantized page directly, so the cache is dequantized into [total_kv_tokens, kv_num_heads,
  // head_size] (no GQA expansion) and fed to the non-paged varlen entry point.
  T* gathered_key = nullptr;
  T* gathered_value = nullptr;
  T* fmha_buffer = nullptr;  // CUTLASS fMHA output-accumulator workspace
  // Populated by the caller after a D->H sync on cumulative_seqlens_kv[batch_size].
  int total_kv_tokens = 0;
  // Max per-batch total KV length. Only needed when the gathered (non-paged) Flash path is used,
  // where it becomes mha_varlen_fwd's max_seqlen_k.
  int max_kv_len = 0;

  // Actual max of per-batch new-query lengths (cumulative_seqlens_q[i+1] - cumulative_seqlens_q[i]).
  // Populated by the caller via the same D->H sync so the MEA path's rotary grid and MEA's
  // grid_x (ceil_div(sequence_length, kQueriesPerBlock)) cover every query token. The previous
  // heuristic `token_count - batch_size + 1` underestimates when any batch has 0 new tokens,
  // producing silent per-token dropout in MEA and rotary.
  int max_query_len = 0;

  // Paged decode (flash-decoding style) split-KV workspaces. Only allocated when the paged decode
  // backend is selected. Layouts are [num_splits, token_count, num_heads, head_size] for the
  // accumulator and [num_splits, token_count, num_heads] for the running max / denominator.
  float* decode_partial_out = nullptr;
  float* decode_partial_max = nullptr;
  float* decode_partial_sum = nullptr;
  int num_splits = 1;

  // Paged XQA decode workspaces. Only allocated when the XQA decode backend is selected
  // (quantized cache, one new token per sequence -- see use_xqa_decode).
  //   xqa_workspace   : XQA semaphores + multi-block scratch (GetXQAScratchSize bytes).
  //   xqa_page_table  : block_table expanded from PagedAttention blocks to XQA's fixed 128-token
  //                     pages, shape [batch_size, max_num_blocks_per_seq * pages_per_block].
  //   xqa_query       : scratch for Q pre-scaled by a PER_CHANNEL k_scale; unused otherwise.
  //   xqa_head_sink   : head_sink converted to fp32, which is what XQA consumes.
  void* xqa_workspace = nullptr;
  size_t xqa_workspace_size = 0;
  int* xqa_page_table = nullptr;
  T* xqa_query = nullptr;
  float* xqa_head_sink = nullptr;

  // Output Tensors
  T* output = nullptr;

  // Kernel Flags
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
  // Paged decode kernel: reads the paged cache in place and dequantizes inside the kernel, so it
  // needs neither the dense staging buffers nor FlashAttention's page-alignment constraint.
  bool use_paged_decode = false;
  // XQA paged decode kernel: same in-place paged read, but tensor-core based and an order of
  // magnitude faster than the generic decode kernel on a quantized cache. Takes precedence over
  // use_paged_decode when set.
  bool use_xqa_decode = false;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
