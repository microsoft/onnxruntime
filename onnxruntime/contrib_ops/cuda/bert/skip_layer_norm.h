// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, bool Simplified>
class SkipLayerNorm final : public CudaKernel {
 public:
  SkipLayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  bool strict_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
