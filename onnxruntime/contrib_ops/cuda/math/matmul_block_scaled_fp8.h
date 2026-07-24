// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

// Weight-only block-scaled FP8 (E4M3) matrix multiplication.
//
// The weight tensor B is FP8 E4M3 of shape [N, K] with one FP32 scale per block_size consecutive
// K values (b_scale of shape [N, ceil(K/block_size)]). The weight is dequantized to the
// activation type (FP16/BF16) and multiplied with the FP16/BF16 activation A via cuBLAS. This path
// works on any CUDA architecture (SM80+): it does not rely on native FP8 block-scaled tensor cores.
//
// When the optional a_scale (a single fp32 scalar) is provided, the activation A is statically
// quantized to FP8 E4M3 and dequantized back (a_deq = fp8_e4m3(A / a_scale) * a_scale) before the
// matmul. This realizes W8A8 activation numerics while keeping the GEMM in the activation type, so
// the result matches native W8A8 execution modulo the FP8 activation rounding error.
class MatMulBlockQuantizedFp8Weight final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  int64_t block_size_;
};

// Dequantizes FP8 (E4M3) weights with per-block FP32 scales into FP16/BF16. b_fp8 is [N, K]
// row-major FP8 E4M3, weight_scale is [N, ceil(K/block_size)] fp32. Output b_dequant is [N, K]
// in the activation type (is_bf16 selects BF16 vs FP16).
Status LaunchDequantizeBlockScaledFp8(void* b_dequant,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream);

// Adds a per-column bias of shape [N] to a [M, N] row-major output in place.
Status LaunchAddBiasBlockScaledFp8(void* y,
                                   const void* bias,
                                   int m,
                                   int n,
                                   bool is_bf16,
                                   cudaStream_t stream);

// Statically quantizes a [M, K] FP16/BF16 activation to FP8 E4M3 using a single per-tensor fp32
// scale and dequantizes it back to the activation type in place-compatible scratch a_out
// (a_out = fp8_e4m3(a_in / a_scale) * a_scale). Realizes W8A8 activation numerics. a_scale is a
// device fp32 scalar. Runs on any architecture with FP8 conversion intrinsics (CUDA >= 11.8).
Status LaunchQuantizeDequantizeActivationFp8(void* a_out,
                                             const void* a_in,
                                             const float* a_scale,
                                             int m,
                                             int k,
                                             bool is_bf16,
                                             cudaStream_t stream);

// Fused FP8 weight-only GEMV fast path for the decode phase (small M). Reads the FP8 weight
// directly (no [N, K] dequant buffer). a is [M, K] activation (FP16/BF16), b_fp8 is [N, K]
// FP8 E4M3, weight_scale is [N, ceil(K/block_size)] fp32, bias is an optional [N] vector (may be
// null). Output y is [M, N] in the activation type. Requires k % 16 == 0 and block_size % 16 == 0.
// Runs on any architecture with FP8 conversion intrinsics (CUDA >= 11.8).
Status LaunchMatMulBlockScaledFp8Gemv(void* y,
                                      const void* a,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      const void* bias,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
