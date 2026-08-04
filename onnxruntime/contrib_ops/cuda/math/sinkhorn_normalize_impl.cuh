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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
