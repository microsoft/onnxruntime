// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/sinkhorn_normalize_impl.h"

#include "contrib_ops/cuda/math/sinkhorn_normalize_impl.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

__global__ void SinkhornNormalizeKernel(const float* input, float* output, int order,
                                        int iterations, float epsilon, int num_matrices) {
  extern __shared__ float smem[];

  const int lane = threadIdx.x;
  const int matrix = blockIdx.x * blockDim.y + threadIdx.y;
  // Uniform across the warp, so the early exit never splits a __syncwarp.
  if (matrix >= num_matrices) return;

  const int count = order * order;
  float* tile = smem + threadIdx.y * (count + order);
  float* sums = tile + count;

  const float* src = input + static_cast<size_t>(matrix) * count;
  for (int e = lane; e < count; e += kWarpSize) tile[e] = src[e];
  __syncwarp();

  SinkhornNormalizeWarp(tile, sums, order, lane, iterations, epsilon);

  float* dst = output + static_cast<size_t>(matrix) * count;
  for (int e = lane; e < count; e += kWarpSize) dst[e] = tile[e];
}

}  // namespace

Status LaunchSinkhornNormalize(cudaStream_t stream, const float* input, float* output,
                               int num_matrices, int order, int iterations, float epsilon) {
  if (num_matrices == 0) {
    return Status::OK();
  }

  const int blocks = (num_matrices + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t shared_bytes =
      static_cast<size_t>(kWarpsPerBlock) * (order * order + order) * sizeof(float);

  SinkhornNormalizeKernel<<<blocks, dim3(kWarpSize, kWarpsPerBlock), shared_bytes, stream>>>(
      input, output, order, iterations, epsilon, num_matrices);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
