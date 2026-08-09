// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class LinearAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  LinearAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int q_num_heads_;
  int kv_num_heads_;
  std::string update_rule_;
  float scale_;
  int chunk_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
