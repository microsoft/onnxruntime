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
class ShortConvWithState final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit ShortConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string activation_;
  int64_t dilation_;
  float epsilon_;
  int64_t kernel_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
