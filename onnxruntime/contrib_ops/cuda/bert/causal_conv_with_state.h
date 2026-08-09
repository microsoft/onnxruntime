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
class CausalConvWithState final : public onnxruntime::cuda::CudaKernel {
 public:
  CausalConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int ndim_;
  std::string activation_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
