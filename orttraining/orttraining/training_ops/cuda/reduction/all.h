// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class All final : public CudaKernel {
 public:
  All(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template<typename T>
void LaunchAllKernel(cudaStream_t stream, const T* data, const int size, bool* output);

}  // namespace cuda
}  // namespace onnxruntime
