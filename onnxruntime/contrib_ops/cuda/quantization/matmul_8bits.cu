// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/matmul_8bits_batched.cuh"
#include "contrib_ops/cuda/quantization/matmul_8bits_m1.cuh"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMul8Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
  constexpr int kColsPerThreadBlock = 8;
  constexpr int kElementsPerThreadPerIteration = 8;
  constexpr int kWarpSize = onnxruntime::cuda::GPU_WARP_SIZE;

  if (m < 1 || m > 5 || n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }
  constexpr int kPerIter = kWarpSize * kElementsPerThreadPerIteration;
  if (kPerIter % block_size != 0 || k % block_size != 0) {
    return false;
  }

  if (m >= 2) {
    return TryMatMul8BitsBatched(output, a_data, b_data_quant, scales_data, zero_points,
                                 m, n, k, block_size, stream);
  }

  return TryMatMul8BitsM1(output, a_data, b_data_quant, scales_data, zero_points,
                          n, k, block_size, shared_mem_per_block, stream);
}

template bool TryMatMul8Bits<float>(
    float*, const float*, const uint8_t*, const float*, const uint8_t*, int, int, int, int, size_t, cudaStream_t);

template bool TryMatMul8Bits<half>(
    half*, const half*, const uint8_t*, const half*, const uint8_t*, int, int, int, int, size_t, cudaStream_t);

template bool TryMatMul8Bits<nv_bfloat16>(
    nv_bfloat16*, const nv_bfloat16*, const uint8_t*, const nv_bfloat16*, const uint8_t*, int, int, int, int,
    size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
