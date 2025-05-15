// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "rotary_embedding_onnx_helper.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status RunRotaryEmbeddingONNX(onnxruntime::concurrency::ThreadPool* tp, rotary_embedding_onnx_helper::RotaryParameters parameters, const T* input,
                          const T* cos_cache, const T* sin_cache, const int64_t* position_ids, T* output,
                          bool interleaved);

template <typename T>
class RotaryEmbeddingONNX final : public OpKernel {
 public:
  RotaryEmbeddingONNX(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  bool interleaved;
  int rotary_embedding_dim;
  int num_heads;
};

}  // namespace contrib
}  // namespace onnxruntime
