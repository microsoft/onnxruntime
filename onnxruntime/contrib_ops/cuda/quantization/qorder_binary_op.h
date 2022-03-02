// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

using namespace onnxruntime::cuda;

class QOrderedAdd final : public CudaKernel {
 public:
  QOrderedAdd(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_A_;
  int order_B_;
  int order_Y_;
};

class QOrderedBiasGelu final : public CudaKernel {
 public:
  QOrderedBiasGelu(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_A_;
  int order_B_;
  int order_Y_;
};

// #endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
