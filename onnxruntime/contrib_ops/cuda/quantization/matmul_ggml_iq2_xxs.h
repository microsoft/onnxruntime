// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulGGMLIQ2XXS final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulGGMLIQ2XXS(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

enum class IQ2XXSActivationType { Float,
                                  Float16,
                                  BFloat16 };

// B retains serialized GGML [N, K/256, 66] blocks. No expanded weight buffer is allocated.
Status LaunchMatMulGGMLIQ2XXS(const void* a, const uint8_t* b, const void* bias, void* y,
                              int64_t m, int64_t n, int64_t k_blocks,
                              IQ2XXSActivationType type, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
