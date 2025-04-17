// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  int batch_size;
  int sequence_length;
  int kv_sequence_length;     // input sequence length of K or V
  int past_sequence_length;   // sequence length in past state of K or V
  int total_sequence_length;  // total sequence length of K or V
  int max_sequence_length;    // max sequence length from 4D mask
  int input_hidden_size;      // first dimension of weights for input projection
  int hidden_size;            // hidden size of Q or K
  int head_size;              // hidden size per head of Q or K
  int v_hidden_size;          // hidden size of V
  int v_head_size;            // hidden size per head of V
  int num_heads;
  int num_splits;
  int rotary_embedding;
  int beam_width;
  bool is_unidirectional;
  bool past_present_share_buffer;
  bool do_rotary;
  bool broadcast_attn_bias_dim_0;
  bool broadcast_attn_bias_dim_1;
  float mask_filter_value;
  float scale;
  bool use_tf32;
  AttentionMaskType mask_type;
  AttentionQkvFormat qkv_format;
};

// Parameters deduced from node attributes and inputs/outputs.
struct PackedAttentionParameters : AttentionParameters {
  int token_count;
};

struct DecoderMaskedMultiHeadAttentionParameters : AttentionParameters {
  int beam_width = 1;

  // Only NeoX style rotary embedding is supported
  int rotary_embedding_dim = 0;
  int t_step = 0;

  // Weather to use multihead attention(excludes matmul and bias)
  bool is_mha = false;
  bool is_cross_attention = false;
  bool is_packed_qkv = false;

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
  int seqlen_past_kv_cache;     // sequence length of past kv tensor
  int seqlen_present_kv_cache;  // sequence length of present kv tensor
  int kv_hidden_size;
  int kv_num_heads;
  int num_splits;         // number of splits for splitkv
  int rotary_dim;         // rotary embedding dimension
  int local_window_size;  // The window size excludes current token. It only includes tokens on the left side.
  bool kv_share_buffer;
  bool is_packed_qkv;
  bool is_subsequent_prompt;  // indicates whether we have past context and seqlen > 1
  bool is_first_prompt;       // indicates whether this is first decoding step
  bool rotary_interleaved;
  bool use_smooth_softmax;
  float softcap;
  AttentionQkvFormat past_kv_format;
  int zeros_count;
  int* zero_ptr;
};

// Parameters for sparse attention.
struct SparseAttentionParameters : AttentionParameters {
  int kv_hidden_size;              // hidden size of key or value
  int kv_num_heads;                // number of heads of key or value
  bool do_rotary;                  // whether to use rotary embedding
  bool rotary_interleaved;         // whether to use interleaved rotary embedding
  int rotary_dim;                  // rotary embedding dimension
  int sparse_block_size;           // block size for sparse attention
  int num_sparse_layout;           // number of sparse layout
  int stride_col_indices;          // shape of block_col_indices is [num_sparse_layout, stride_col_indices]
  int stride_row_indices;          // shape of block_row_indices is [num_sparse_layout, stride_row_indices]
  bool is_packed_qkv;              // whether qkv is packed
  int max_rotary_sequence_length;  // max sequence length for rotary cos/sin cache
  int max_cache_sequence_length;   // max sequence length for kv cache buffer
};

}  // namespace contrib
}  // namespace onnxruntime
