// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ZeroGradient final : public CudaKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename T_GRAD>
class InPlaceAccumulator final : public CudaKernel {
 public:
  InPlaceAccumulator(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    cudaStream_t stream,
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime