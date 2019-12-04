// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
namespace contrib {

class WordConvEmbedding final : public OpKernel {
 public:
  explicit WordConvEmbedding(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  void CharEmbeddingLookup(
      const int* seq_ptr,
      const float* char_embedding_weight_p,
      size_t seq_len,
      size_t word_len,
      size_t char_embedding_size,
      size_t filter_width,
      const int* words_len_ptr,
      float* dst) const;
  void ComputeConvMaxPoolWithActivation(
      AllocatorPtr allocator,
      const float* input,
      const float* weights,
      const float* bias,
      const int* words_len_ptr,
      int64_t seq_len,
      int64_t word_len,
      int64_t char_embedding_size,
      int64_t filter_width,
      int64_t num_filters,
      float* output, onnxruntime::concurrency::ThreadPool* tp) const;
  void CalculateLengthOfEachWordInSequence(
      const int* seq_ptr,
      int* words_len_ptr,
      size_t seq_len,
      size_t word_len) const;

  Status ValidateInputShape(
      const TensorShape& w_conv_shape,
      const TensorShape& w_char_embedding_shape) const;

 private:
  int64_t embedding_size_{Info().GetAttrOrDefault<int64_t>("embedding_size", -1)};
  int64_t conv_window_size_{Info().GetAttrOrDefault<int64_t>("conv_window_size", -1)};
  int64_t char_embedding_size_{Info().GetAttrOrDefault<int64_t>("char_embedding_size", -1)};
};

}  // namespace contrib
}  // namespace onnxruntime
