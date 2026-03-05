// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class LinearAttentionChunkParallel final : public CudaKernel {
 public:
  LinearAttentionChunkParallel(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  int chunk_size_;
  float scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
