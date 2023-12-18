// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 public:
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* relative_position_bias,
                     void* parameters,
                     const int max_threads_per_block,  // for CUDA
                     const Tensor* past_seq_len = nullptr) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int kv_sequence_length,
                     int& past_sequence_length) const;

 protected:
  AttentionBase(const OpKernelInfo& info, bool require_same_hidden_size) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_embedding_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding", 0));
    mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

    if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK()) {
      qkv_hidden_sizes_.clear();
    }

    past_present_share_buffer_ = info.GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL);

    require_same_hidden_size_ = require_same_hidden_size;
  }

  Status CheckMask(const Tensor* mask_index,
                   AttentionMaskType& mask_type,
                   int64_t& max_sequence_length,  // output: max_sequence_length when mask_index is 4D tensor
                   int64_t batch_size,
                   int64_t sequence_length,
                   int64_t total_sequence_length) const;

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* relative_position_bias,
                     void* parameters,
                     const Tensor* past_seq_len = nullptr) const;

  int num_heads_;                          // number of attention heads
  bool is_unidirectional_;                 // whether every token can only attend to previous tokens.
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V hidden sizes parsed from the qkv_hidden_sizes attribute.
  bool require_same_hidden_size_;          // whether the implementation supports different hidden sizes of Q/K/V.
  bool past_present_share_buffer_;         // whether or not the past (if used) and present tensor share the same buffer
  bool do_rotary_;                         // whether or not to use rotary embeddings
  int rotary_embedding_;                   // rotary embedding dimension
  float mask_filter_value_;                // the value to be used for filtered out positions
  float scale_;                            // the scale to be used for softmax
};

}  // namespace contrib
}  // namespace onnxruntime
