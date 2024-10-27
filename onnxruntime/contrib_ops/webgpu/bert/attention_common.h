// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

struct WebgpuAttentionParameters {
    AttentionParameters(onnxruntime::contrib::AttentionParameters parameters):
    is_gqa_parameters_(false),
    batch_size_(parameters.parameters.batch_size),
    sequence_length_(parameters.parameters.sequence_length),
    kv_sequence_length_(parameters.parameters.kv_sequence_length),
    past_sequence_length_(parameters.parameters.past_sequence_length),
    total_sequence_length_(parameters.parameters.total_sequence_length),
    max_sequence_length_(parameters.parameters.max_sequence_length),
    input_hidden_size_(parameters.parameters.input_hidden_size),
    hidden_size_(parameters.parameters.hidden_size),
    head_size_(parameters.parameters.head_size),
    v_hidden_size_(parameters.parameters.v_hidden_size),
    v_head_size_(parameters.parameters.v_head_size),
    num_heads_(parameters.parameters.num_heads),
    is_unidirectional_(parameters.parameters.is_unidirectional),
    past_present_share_buffer_(parameters.parameters.past_present_share_buffer),
    do_rotary_(parameters.parameters.do_rotary),
    broadcast_attn_bias_dim_0_(parameters.parameters.broadcast_attn_bias_dim_0),
    broadcast_attn_bias_dim_1_(parameters.parameters.broadcast_attn_bias_dim_1),
    mask_filter_value_(parameters.parameters.mask_filter_value),
    scale_(parameters.parameters.scale),
    mask_type_(parameters.parameters.mask_type),
    qkv_format_(parameters.parameters.qkv_format) {
    }

    AttentionParameters(onnxruntime::contrib::GroupQueryAttentionParameters parameters):
    is_gqa_parameters_(true),
    batch_size_(parameters.batch_size),
    sequence_length_(parameters.sequence_length),
    seqlen_past_kv_cache_(parameters.seqlen_past_kv_cache),
    seqlen_present_kv_cache_(parameters.seqlen_present_kv_cache),
    total_sequence_length_(parameters.total_sequence_length),
    hidden_size_(parameters.hidden_size),
    num_heads_(parameters.num_heads),
    head_size_(parameters.head_size),
    kv_hidden_size_(parameters.kv_hidden_size),
    kv_num_heads_(parameters.kv_num_heads),
    num_splits_(parameters.num_splits),
    rotary_dim_(parameters.rotary_dim),
    is_unidirectional_(parameters.is_unidirectional),
    do_rotary_(parameters.do_rotary_),
    rotary_interleaved_(parameters.rotary_interleaved_),
    use_smooth_softmax_(parameters.use_smooth_softmax_),
    mask_filter_value_(parameters.scale_),
    softcap_(parameters.softcap_),
    qkv_format_(parameters.qkv_format),
    zeros_count_(parameters.zeros_count_),
    zero_ptr_(parameters.zero_ptr_),
    n_reps(parameters.num_heads / parameters.kv_num_heads) {
   }

  boolean is_gqa_parameters_;
  int batch_size_(0)
  int sequence_length_(0)
  int kv_sequence_length_(0)     // input sequence length of K or V
  int past_sequence_length_(0)   // sequence length in past state of K or V
  int total_sequence_length_(0)  // total sequence length of K or V
  int max_sequence_length_(0)    // max sequence length from 4D mask
  int input_hidden_size_(0)      // first dimension of weights for input projection
  int hidden_size_(0)            // hidden size of Q or K
  int head_size_(0)              // hidden size per head of Q or K
  int v_hidden_size_(0)          // hidden size of V
  int v_head_size_(0)            // hidden size per head of V
  int num_heads_(0)
  int rotary_embedding_(0)
  bool is_unidirectional_(false)
  bool past_present_share_buffer_(false)
  bool do_rotary_(false)
  bool broadcast_attn_bias_dim_0_(false)
  bool broadcast_attn_bias_dim_1_(false)
  float mask_filter_value_;
  float scale_;
  bool use_tf32_(false);
  // The following members are in onnxruntime::contrib::GroupQueryAttentionParameters
  // and not in onnxruntime::contrib::AttentionParameters
  int seqlen_past_kv_cache_(0)     // sequence length of past kv tensor
  int seqlen_present_kv_cache_(0)  // sequence length of present kv tensor
  int kv_hidden_size_(0)
  int kv_num_heads_(0)
  int num_splits_(0)          // number of splits for splitkv
  int rotary_dim_(0)          // rotary embedding dimension
  int local_window_size_(0)
  bool kv_share_buffer_(false)
  bool is_packed_qkv_(false)
  bool is_subsequent_prompt_(false)  // indicates whether we have past context and seqlen > 1
  bool is_first_prompt_(false)       // indicates whether this is first decoding step
  bool do_rotary_(false)
  bool rotary_interleaved_(false)
  bool use_smooth_softmax_(false)
  float scale_;
  float softcap_;
  int zeros_count_(0);
  int* zero_ptr_(nullptr);
  // Computed values
  int n_reps(1);
  AttentionMaskType mask_type_;
  AttentionQkvFormat qkv_format_;
  };

}
}
}
