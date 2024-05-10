// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// S2SModelSplitQuickGelu

class S2SModelSplitQuickGelu final : public onnxruntime::cuda::CudaKernel {
  public:
    S2SModelSplitQuickGelu(const OpKernelInfo& info) : CudaKernel{info} {}

    Status ComputeInternal(OpKernelContext* context) const override;

  private:
    template <typename T>
    struct KernelLaunchDispatcher {
      void operator()(cudaStream_t stream, int64_t input_size, int64_t axis, int64_t alpha, const Tensor& X,
                      const Tensor& S, Tensor& Y) const;
    };
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
