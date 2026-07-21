// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

// Weight-only NVFP4 (E2M1) matrix multiplication.
//
// The weight tensor B is stored as packed NVFP4: two E2M1 values per byte (low nibble first),
// with a per-16-block E4M3 scale (weight_scale) and a single global fp32 scale (weight_scale_2).
// The weight is dequantized to the activation type (FP16/BF16) and multiplied with the FP16/BF16
// activation via cuBLAS. This path works on any CUDA architecture (including Hopper/SM90) because
// it does not rely on native NVFP4 block-scaled tensor cores (SM100/SM120 only).
class MatMulBlockScaledFp4 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulBlockScaledFp4(const OpKernelInfo& info);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int sm_{0};
  IAllocatorUniquePtr<uint8_t> b_scale_prepacked_;
};

// Dequantizes NVFP4 (E2M1) weights with per-block E4M3 scales and a global fp32 scale into
// FP16/BF16. b_packed is [N, K/2] uint8 (two E2M1 values per byte, low nibble first),
// weight_scale is [N, ceil(K/block_size)] uint8 (raw E4M3 bytes), weight_scale_2 is a device
// fp32 scalar. Output b_dequant is [N, K] in the activation type (is_bf16 selects BF16 vs FP16).
Status LaunchDequantizeNvFp4(void* b_dequant,
                             const void* b_packed,
                             const void* weight_scale,
                             const float* weight_scale_2,
                             int n,
                             int k,
                             int block_size,
                             bool is_bf16,
                             cudaStream_t stream);

// Adds a per-column bias of shape [N] to a [M, N] row-major output in place.
Status LaunchAddBiasNvFp4(void* y,
                          const void* bias,
                          int m,
                          int n,
                          bool is_bf16,
                          cudaStream_t stream);

// Fused NVFP4 weight-only GEMV fast path for the decode phase (small M). Reads the packed
// NVFP4 weight directly (no [N, K] dequant buffer). a is [M, K] activation (FP16/BF16),
// b_packed is [N, K/2] uint8 (two E2M1 values per byte), weight_scale is [N, ceil(K/block_size)]
// uint8 (raw E4M3 bytes), weight_scale_2 is a device fp32 scalar, bias is an optional [N] vector
// (may be null). Output y is [M, N] in the activation type. Requires block_size == 16 and
// k % 32 == 0. Runs on any architecture with NVFP4 conversion intrinsics (CUDA >= 12.8).
Status LaunchMatMulBlockScaledFp4Gemv(void* y,
                                      const void* a,
                                      const void* b_packed,
                                      const void* weight_scale,
                                      const float* weight_scale_2,
                                      const void* bias,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream);

// Native Blackwell SM120 NVFP4 x NVFP4 GEMM path. The caller provides scratch buffers for
// packed activation FP4, swizzled A/B scale tensors, alpha, and CUTLASS workspace. A is [M, K]
// FP16/BF16, B is [N, K/2] packed NVFP4, weight_scale is [N, K/16] E4M3, and Y is [M, N]
// FP16/BF16. Requires block_size == 16, K % 32 == 0, and N % 32 == 0.
Status LaunchRepackWeightScaleNvFp4ForNativeSm120(void* b_scale,
                                                  const void* weight_scale,
                                                  int n,
                                                  int k,
                                                  int block_size,
                                                  cudaStream_t stream);

Status LaunchMatMulBlockScaledFp4NativeSm120(void* y,
                                             const void* a,
                                             const void* b_packed,
                                             const void* weight_scale,
                                             const float* weight_scale_2,
                                             const float* input_scale,
                                             void* a_packed,
                                             void* a_scale,
                                             const void* b_scale,
                                             float* alpha,
                                             int m,
                                             int n,
                                             int k,
                                             int block_size,
                                             bool is_bf16,
                                             void* workspace,
                                             size_t workspace_size,
                                             cudaStream_t stream);
size_t GetMatMulBlockScaledFp4NativeSm120WorkspaceSize(int m, int n, int k, bool is_bf16);

}  // namespace onnxruntime::contrib::cuda
