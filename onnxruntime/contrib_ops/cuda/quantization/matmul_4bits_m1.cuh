// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Returns 0 for the staged kernel, or the direct-scale unroll factor.
constexpr int SelectMatMul4BitsM1DirectUnroll(int mode, int n, int k, int block_size, bool has_zero_point) {
  if ((mode != 1 && mode != 2) || block_size != 32 || has_zero_point ||
      n <= 0 || n % 8 != 0 || k <= 0 || k % 8 != 0) {
    return 0;
  }
  // Keep the staged kernel for the measured long-K, narrow-N regression.
  if (mode == 2 && k >= 4096 && n < 1024) {
    return 0;
  }
  // Four-way unrolling regressed long reductions.
  return mode == 2 && k < 4096 ? 4 : 16;
}

template <class T>
bool TryMatMul4BitsM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
