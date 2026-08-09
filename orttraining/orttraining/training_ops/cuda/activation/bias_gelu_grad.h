// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename GeluComputationMode>
class BiasGeluGrad_dX : public CudaKernel {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct KernelLaunchDispatcher {
    void operator()(
        cudaStream_t stream,
        int64_t input_size, int64_t bias_size,
        const Tensor& dY, const Tensor& X, const Tensor& B,
        Tensor& dX) const;
  };
};

}  // namespace cuda
}  // namespace onnxruntime
