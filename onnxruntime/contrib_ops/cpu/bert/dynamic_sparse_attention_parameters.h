// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

enum class DynamicSparseAttentionMode : int {
  kSelectedOnly = 0,
  kLocalPlusSelected = 1,
};

enum class DynamicSparseAttentionKvSource : int {
  kMain = 0,
  kAuxiliary = 1,
};

struct DynamicSparseAttentionParameters {
  int batch_size = 0;
  int sequence_length = 0;
  int num_heads = 0;
  int kv_num_heads = 0;
  int head_size = 0;
  int query_hidden_size = 0;
  int kv_hidden_size = 0;
  int cache_capacity = 0;
  int past_cache_capacity = 0;
  int total_sequence_length = 0;
  int auxiliary_sequence_length = 0;
  int max_selected = 0;
  int rotary_dim = 0;
  int rotary_max_position = 0;
  int rotary_offset = 0;
  int local_window_size = -1;
  float scale = 0.0f;
  float qk_norm_epsilon = 1e-6f;
  bool is_packed_qkv = false;
  bool do_rotary = false;
  bool rotary_interleaved = false;
  bool use_qk_norm = false;
  bool use_smooth_softmax = false;
  bool auxiliary_kv_shared = false;
  DynamicSparseAttentionMode attention_mode = DynamicSparseAttentionMode::kSelectedOnly;
  DynamicSparseAttentionKvSource selected_kv_source = DynamicSparseAttentionKvSource::kMain;
};

}  // namespace contrib
}  // namespace onnxruntime
