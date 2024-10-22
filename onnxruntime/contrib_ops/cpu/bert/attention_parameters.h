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
struct PackedAttentionParameters {
  int batch_size;
  int sequence_length;
  int input_hidden_size;  // hidden size of input
  int hidden_size;        // hidden size of Q or K
  int head_size;          // hidden size per head of Q or K
  int v_hidden_size;      // hidden size of V
  int v_head_size;        // hidden size per head of V
  int num_heads;
  float scale;
  int token_count;
  bool broadcast_attn_bias_dim_0;
  bool broadcast_attn_bias_dim_1;
  bool use_tf32;
};

// Parameters deduced from node attributes and inputs/outputs.
struct GroupQueryAttentionParameters {
  int batch_size;
  int sequence_length;          // sequence length of input query, key, value
  int seqlen_past_kv_cache;     // sequence length of past kv tensor
  int seqlen_present_kv_cache;  // sequence length of present kv tensor
  int hidden_size;
  int num_heads;
  int head_size;
  int kv_hidden_size;
  int kv_num_heads;
  int num_splits;          // number of splits for splitkv
  int rotary_dim;          // rotary embedding dimension
  bool is_unidirectional;  // causal
  int local_window_size;
  bool kv_share_buffer;
  bool is_packed_qkv;
  bool is_prompt;  // determines if seqlens_k is past or kv sequence length tensor
  bool do_rotary;
  bool rotary_interleaved;
  float scale;
  AttentionQkvFormat qkv_format;
  AttentionQkvFormat past_kv_format;
  int zeros_count;
  int* zero_ptr;
};

// Parameters for sparse attention.
struct SparseAttentionParameters {
  int batch_size;                  // batch size
  int sequence_length;             // sequence length of input query, key, value
  int hidden_size;                 // hidden size of query
  int num_heads;                   // number of heads of query
  int head_size;                   // hidden size per head of query, key or value
  int kv_hidden_size;              // hidden size of key or value
  int kv_num_heads;                // number of heads of key or value
  bool do_rotary;                  // whether to use rotary embedding
  bool rotary_interleaved;         // whether to use interleaved rotary embedding
  int rotary_dim;                  // rotary embedding dimension
  int sparse_block_size;           // block size for sparse attention
  int num_sparse_layout;           // number of sparse layout
  int stride_col_indices;          // shape of block_col_indices is [num_sparse_layout, stride_col_indices]
  int stride_row_indices;          // shape of block_row_indices is [num_sparse_layout, stride_row_indices]
  float scale;                     // scaling factor applied prior to softmax
  bool is_packed_qkv;              // whether qkv is packed
  int total_sequence_length;       // maximum total sequence length (past_sequence_length + sequence_length) among keys
  int max_sequence_length;         // max sequence length for sparse layout
  int max_rotary_sequence_length;  // max sequence length for rotary cos/sin cache
  int max_cache_sequence_length;   // max sequence length for kv cache buffer
  bool past_present_share_buffer;  // whether past_key and present_key share buffer, so is past_value and present_value
};

}  // namespace contrib
}  // namespace onnxruntime
