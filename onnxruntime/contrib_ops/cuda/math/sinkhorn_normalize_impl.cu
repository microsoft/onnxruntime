// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/sinkhorn_normalize_impl.h"

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

// Divide every element by its row (or column) sum, epsilon-padded. The calling warp owns
// `tile` exclusively, so __syncwarp is enough to order the sum against the division.
__device__ __forceinline__ void NormalizeAxis(float* tile, float* sums, int order, int lane,
                                              float epsilon, bool by_row) {
  if (lane < order) {
    float sum = 0.f;
    if (by_row) {
      for (int j = 0; j < order; ++j) sum += tile[lane * order + j];
    } else {
      for (int i = 0; i < order; ++i) sum += tile[i * order + lane];
    }
    sums[lane] = sum + epsilon;
  }
  __syncwarp();

  const int count = order * order;
  for (int e = lane; e < count; e += kWarpSize) {
    tile[e] /= by_row ? sums[e / order] : sums[e % order];
  }
  __syncwarp();
}

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

  NormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/false);
  for (int it = 1; it < iterations; ++it) {
    NormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/true);
    NormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/false);
  }

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
