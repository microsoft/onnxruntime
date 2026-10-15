// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulNVFP4A16 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulNVFP4A16(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;
};

// Only the local E2M1*E4M3 factor is reconstructed; the global FP32 scale is excluded.
Status LaunchDequantizeNVFP4Local(void* weight, const uint8_t* packed, const uint8_t* scales,
                                  int64_t n, int64_t k, bool is_bf16, cudaStream_t stream);

// A null accumulator denotes the empty K=0 sum.
Status LaunchNVFP4A16Epilogue(void* y, const float* accumulator, const float* global_scale,
                              int64_t count, bool is_bf16, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
