// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <bool inputk>
class TopK final : public CudaKernel {
 public:
  TopK(const OpKernelInfo&);
  Status ComputeInternal(OpKernelContext*) const override;

 private:
  int64_t axis_;
  int64_t largest_;
  int64_t sorted_;
  int64_t attr_k_;
};
}  // namespace cuda
}  // namespace onnxruntime
