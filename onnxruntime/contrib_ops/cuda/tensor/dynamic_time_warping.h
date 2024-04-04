// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include <core/common/safeint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::OpKernelContext;
using onnxruntime::OpKernelInfo;
using onnxruntime::cuda::CudaKernel;
class DynamicTimeWarping final : public CudaKernel {
 public:
  DynamicTimeWarping(const OpKernelInfo& info) : CudaKernel(info) {}

  ~DynamicTimeWarping() = default;

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
