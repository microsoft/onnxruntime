// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 public:
  // This check function is specifically used in cuda
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* extra_add_qk,
                     const Tensor* key,
                     const Tensor* value,
                     const Tensor* weight_key,
                     const Tensor* weight_value,
                     const int max_threads_per_block) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int sequence_length,
                     int& past_sequence_length) const;

 protected:
  AttentionBase(const OpKernelInfo& info, bool require_same_hidden_size, bool require_merged_weights) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;

    if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK() || qkv_hidden_sizes_.empty()) {
      qkv_hidden_sizes_.resize(0);
    }

    use_merged_weights_ = info.GetAttrOrDefault<int64_t>("merged_weights", 1) == 1;

    require_same_hidden_size_ = require_same_hidden_size;
    require_merged_weights_ = require_merged_weights;
  }

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor*& mask_index,  // Dummy mask of shape (1 or batch_size, 1) will be updated to nullptr.
                     const Tensor* past,
                     const Tensor* extra_add_qk,
                     const Tensor* key,
                     const Tensor* value,
                     const Tensor* weight_key,
                     const Tensor* weight_value) const;

  int num_heads_;                          // number of attention heads
  bool is_unidirectional_;                 // whether every token can only attend to previous tokens.
  bool use_merged_weights_;                // whether the weights for Q/K/V are merged.
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V path hidden layer sizes
  bool require_same_hidden_size_;          // whether the kernel supports different hidden sizes of Q/K/V
  bool require_merged_weights_;            // whether the kernel supports separated weights for Q/K/V
};

}  // namespace contrib
}  // namespace onnxruntime
