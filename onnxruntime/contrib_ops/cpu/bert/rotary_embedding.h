// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "rotary_embedding_helper.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status RunRotaryEmbedding(onnxruntime::concurrency::ThreadPool* tp, rotary_embedding_helper::RotaryParameters parameters, const T* input,
                          const int64_t* position_ids, const T* cos_cache, const T* sin_cache, T* output,
                          bool interleaved);

template <typename T>
class RotaryEmbedding final : public OpKernel {
 public:
  RotaryEmbedding(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  float scale;
  int num_heads;
  int rotary_embedding_dim;
  bool interleaved;
  bool is_packed_batching;
};

}  // namespace contrib
}  // namespace onnxruntime
