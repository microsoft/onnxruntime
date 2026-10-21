// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/matmul_sigmoid_gated_residual.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void MatMulSigmoidGatedResidualKernel(
    const half* x, const half* weight, const half* shared, const half* routed,
    const half* residual, half* output, int k, int n) {
  constexpr int kThreads = 256;
  __shared__ float warp_sums[kThreads / 32];
  __shared__ half gate;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int64_t row = blockIdx.x;
  float dot = 0.0f;
  for (int64_t c = tid; c < k; c += kThreads) {
    dot = fmaf(__half2float(x[row * k + c]), __half2float(weight[c]), dot);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = dot;
  }
  __syncthreads();
  if (warp == 0) {
    dot = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
    }
    if (lane == 0) {
      const float rounded_dot = __half2float(__float2half_rn(dot));
      gate = __float2half_rn(1.0f / (1.0f + expf(-rounded_dot)));
    }
  }
  __syncthreads();
  for (int64_t c = tid; c < n; c += kThreads) {
    const int64_t index = row * n + c;
    // Keep the original Mul -> Add -> Add rounding boundaries; do not contract into FMA.
    const half gated = __hmul_rn(shared[index], gate);
    const half combined = __hadd_rn(routed[index], gated);
    output[index] = __hadd_rn(residual[index], combined);
  }
}

}  // namespace

Status LaunchMatMulSigmoidGatedResidual(cudaStream_t stream, const void* x, const void* weight,
                                       const void* shared, const void* routed, const void* residual,
                                       void* output, int rows, int k, int n) {
  if (rows == 0) {
    return Status::OK();
  }
  MatMulSigmoidGatedResidualKernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(x), static_cast<const half*>(weight), static_cast<const half*>(shared),
      static_cast<const half*>(routed), static_cast<const half*>(residual), static_cast<half*>(output), k, n);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
