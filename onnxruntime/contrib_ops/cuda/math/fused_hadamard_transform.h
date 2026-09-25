// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class FusedHadamardTransform final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit FusedHadamardTransform(const OpKernelInfo& info) : CudaKernel(info) {
    block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 1024);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t block_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime