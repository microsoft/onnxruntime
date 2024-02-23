// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// AddGelu fuse Add + Gelu
class BiasGelu final : public CudaKernel {
 public:
  BiasGelu(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct KernelLaunchDispatcher {
    void operator()(cudaStream_t stream, int64_t input_size, int64_t bias_size, const Tensor& X, const Tensor& B,
                    Tensor& Y) const;
  };
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
