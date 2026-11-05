// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulGGMLQ8_0 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulGGMLQ8_0(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

enum class Q8ActivationType { Float,
                              Float16,
                              BFloat16 };

Status LaunchMatMulGGMLQ8_0(const void* a, const uint8_t* b, const void* bias, void* y,
                            int64_t m, int64_t n, int64_t k_blocks,
                            Q8ActivationType type, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
