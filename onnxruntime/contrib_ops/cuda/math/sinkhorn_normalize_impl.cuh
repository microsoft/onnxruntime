// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Divide every element by its row (or column) sum, epsilon-padded. The calling warp owns
// `tile` and `sums` exclusively, so __syncwarp is enough to order the sum against the division.
__device__ __forceinline__ void SinkhornNormalizeAxis(float* tile, float* sums, int order, int lane,
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
  for (int e = lane; e < count; e += 32) {
    tile[e] /= by_row ? sums[e / order] : sums[e % order];
  }
  __syncwarp();
}

// `iterations` alternations in place, starting with a column normalization. Every caller must
// use this so the fused and standalone operators stay bit-identical.
__device__ __forceinline__ void SinkhornNormalizeWarp(float* tile, float* sums, int order, int lane,
                                                      int iterations, float epsilon) {
  SinkhornNormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/false);
  for (int it = 1; it < iterations; ++it) {
    SinkhornNormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/true);
    SinkhornNormalizeAxis(tile, sums, order, lane, epsilon, /*by_row=*/false);
  }
}

// Register-resident form of the two routines above. Lane `l` of a fully converged warp owns
// element (l / ORDER, l % ORDER); lanes past ORDER * ORDER must still call in so the shuffles
// see the whole warp, and their value is ignored. The axis sum starts at 0.f and walks the axis
// in index order, and the result is a single `v / (sum + epsilon)`, so every rounding step
// matches SinkhornNormalizeAxis element for element -- the two forms are bit-identical.
template <int ORDER>
__device__ __forceinline__ float SinkhornNormalizeAxisReg(float v, int lane, float epsilon,
                                                          bool by_row) {
  const int i = lane / ORDER;
  const int j = lane - i * ORDER;
  const int src0 = by_row ? i * ORDER : j;
  const int stride = by_row ? 1 : ORDER;
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < ORDER; ++k) sum += __shfl_sync(0xffffffffu, v, src0 + k * stride);
  sum += epsilon;
  return v / sum;
}

template <int ORDER>
__device__ __forceinline__ float SinkhornNormalizeWarpReg(float v, int lane, int iterations,
                                                          float epsilon) {
  v = SinkhornNormalizeAxisReg<ORDER>(v, lane, epsilon, /*by_row=*/false);
  for (int it = 1; it < iterations; ++it) {
    v = SinkhornNormalizeAxisReg<ORDER>(v, lane, epsilon, /*by_row=*/true);
    v = SinkhornNormalizeAxisReg<ORDER>(v, lane, epsilon, /*by_row=*/false);
  }
  return v;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
