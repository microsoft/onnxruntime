// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {
namespace deep_speed {

using namespace onnxruntime::cuda;

template <typename T>
class BiasGelu final : public CudaKernel {
 public:
  explicit BiasGelu(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace deep_speed
}  // namespace cuda
}  // namespace onnxruntime
