// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  int batch_size = 0;
  int sequence_length = 0;
  int kv_sequence_length = 0;     // input sequence length of K or V
  int past_sequence_length = 0;   // sequence length in past state of K or V
  int total_sequence_length = 0;  // total sequence length of K or V
  int max_sequence_length = 0;    // max sequence length from 4D mask
  int input_hidden_size = 0;      // first dimension of weights for input projection
  int hidden_size = 0;            // hidden size of Q or K
  int head_size = 0;              // hidden size per head of Q or K
  int v_hidden_size = 0;          // hidden size of V
  int v_head_size = 0;            // hidden size per head of V
  int num_heads = 0;
  int num_splits = 0;  // number of splits for splitkv
  int rotary_dim = 0;  // rotary embedding dimension
  int beam_width = 0;
  bool is_unidirectional = false;
  bool past_present_share_buffer = false;
  bool is_packed_qkv = false;  // whether qkv is packed
  bool do_rotary = false;
  bool broadcast_attn_bias_dim_0 = false;
  bool broadcast_attn_bias_dim_1 = false;
  float mask_filter_value = 0.0f;
  float scale = 0.0f;
  float softcap = 0.0f;
  bool use_tf32 = false;
  bool is_output_bnsh = false;  // whether the output format is BNSH
  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  AttentionQkvFormat qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
};

// Parameters deduced from node attributes and inputs/outputs.
struct PackedAttentionParameters : AttentionParameters {
  int token_count;
};

struct DecoderMaskedMultiHeadAttentionParameters : AttentionParameters {
  int beam_width = 1;

  // Only NeoX style rotary embedding is supported
  int t_step = 0;

  // Weather to use multihead attention(excludes matmul and bias)
  bool is_mha = false;
  bool is_cross_attention = false;

  // Useful to better use global memory bandwidth on certain CUDA architectures.
  // Turned off by default for now until we fully understand performance implications
  // for all types of workloads.
  // Can be turned on by appropriate environment variable (see attention_common.h).
  bool kv_data_in_flight = false;

  void* q = nullptr;
  void* q_bias = nullptr;

  void* k = nullptr;
  void* k_bias = nullptr;

  void* v = nullptr;
  void* v_bias = nullptr;

  void* attention_bias = nullptr;

  void* k_cache = nullptr;
  void* v_cache = nullptr;

  void* out = nullptr;
  void* out_qk = nullptr;

  const int32_t* cache_indir = nullptr;
  const int32_t* mask = nullptr;  // [B, total_sequence_length]
};

// Parameters deduced from node attributes and inputs/outputs.
struct GroupQueryAttentionParameters : AttentionParameters {
  int kv_num_heads;             // number of heads of key or value
  int kv_hidden_size;           // hidden size of key or value
  int seqlen_past_kv_cache;     // sequence length of past kv tensor
  int seqlen_present_kv_cache;  // sequence length of present kv tensor
  int local_window_size;        // Mask out tokens prior to total_sequence_length - local_window_size
  bool is_subsequent_prompt;    // indicates whether we have past context and seqlen > 1
  bool is_first_prompt;         // indicates whether this is first decoding step
  bool rotary_interleaved;
  bool use_smooth_softmax;
  bool use_qk_norm = false;       // per-head Q/K RMSNorm (QK-Norm) prologue before RoPE (inputs 14/15)
  float qk_norm_epsilon = 1e-6f;  // epsilon for the QK-Norm RMSNorm
  float softcap;
  AttentionQkvFormat past_kv_format;
  int zeros_count;
  int* zero_ptr;

  // Quantization parameters for KV cache
  KVQuantizationType k_quant_type = KVQuantizationType::NONE;
  KVQuantizationType v_quant_type = KVQuantizationType::NONE;
  int kv_cache_bit_width = 0;

  // Windowed (sliding-window) KV cache. Set from the sliding_window_cache attribute.
  // When true, past/present KV buffers are allocated with capacity C = kv_cache_capacity, which may
  // be much smaller than total_sequence_length. The cache then holds only the min(T, C) most recent
  // tokens at cache indices [0, L) and the kernel uses cache-relative indexing plus a shift
  // compaction step. Requires local_window_size > 0. See docs: windowed KV cache design.
  bool is_windowed_kv_cache = false;
  int kv_cache_capacity = 0;       // Capacity of the KV buffer used by this step.
  int kv_cache_real_capacity = 0;  // C: allocated sequence dim of the past/present KV buffers.
  // A multi-token step runs against a longer staging buffer (see GroupQueryAttention::ComputeInternal),
  // in which case kv_cache_capacity is the staging capacity and differs from kv_cache_real_capacity.

  // Upper bound (exclusive) for absolute RoPE positions: dim 0 of the cos/sin caches.
  // 0 when rotary is not configured.
  int rotary_max_position = 0;
};

// Parameters deduced from node attributes and inputs/outputs.
struct PagedAttentionParameters : AttentionParameters {
  int kv_num_heads;            // number of heads of key or value
  int kv_hidden_size;          // hidden size of key or value
  int token_count;             // number of tokens in packed query
  int block_size;              // block size for kv cache
  int max_num_blocks_per_seq;  // max number of blocks per sequence for kv cache
  int num_blocks;              // number of blocks in kv cache
  int local_window_size;       // The window size includes new token. It only includes tokens on the left side.
  bool rotary_interleaved;
  float softcap;
  // Internal attention-sink path, enabled when head_sink (input 11) is provided.
  bool use_smooth_softmax = false;
  // Per-head Q/K RMSNorm (QK-Norm) prologue applied before RoPE (inputs 12/13).
  bool use_qk_norm = false;
  float qk_norm_epsilon = 1e-6f;
  // Quantized paged KV cache. Scales are inputs 14/15 and are always FP32, as in
  // GroupQueryAttention. The storage element type is carried by the kernel's TCACHE specialization;
  // the k_cache_dtype / v_cache_dtype attributes only override it for sub-byte formats packed into
  // uint8, which no backend supports yet.
  KVQuantizationType k_quant_type = KVQuantizationType::NONE;
  KVQuantizationType v_quant_type = KVQuantizationType::NONE;
  // Multi-head Latent Attention (kv_cache_layout == "LATENT"). There is a single physical cache:
  // V of every head is the leading v_head_size channels of the same key_cache row, so 'value' and
  // 'value_cache' are absent. The inherited v_head_size / v_hidden_size hold the effective V width
  // and the output width; in SEPARATE mode they equal head_size / hidden_size.
  bool is_latent_kv = false;
  // First channel within head_size covered by rotary embedding. RoPE covers
  // [rotary_offset, rotary_offset + rotary_dim); channels outside are copied through. Default 0
  // reproduces the original prefix-RoPE behavior. MLA uses rotary_offset == kv_lora_rank.
  int rotary_offset = 0;
};

// Parameters for sparse attention.
struct SparseAttentionParameters : AttentionParameters {
  int kv_hidden_size;              // hidden size of key or value
  int kv_num_heads;                // number of heads of key or value
  bool do_rotary;                  // whether to use rotary embedding
  bool rotary_interleaved;         // whether to use interleaved rotary embedding
  int sparse_block_size;           // block size for sparse attention
  int num_sparse_layout;           // number of sparse layout
  int stride_col_indices;          // shape of block_col_indices is [num_sparse_layout, stride_col_indices]
  int stride_row_indices;          // shape of block_row_indices is [num_sparse_layout, stride_row_indices]
  int max_rotary_sequence_length;  // max sequence length for rotary cos/sin cache
  int max_cache_sequence_length;   // max sequence length for kv cache buffer
};

}  // namespace contrib
}  // namespace onnxruntime
