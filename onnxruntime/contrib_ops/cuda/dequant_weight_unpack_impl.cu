// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kBlockSize = 256;

#define FETCH_FLOAT2(pointer) (reinterpret_cast<const float2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<const half2*>(&(pointer))[0])

template <typename T>
__global__ void kDequantizeAndUnpackWeight(T* out, const int32_t* qweight, const T* scale, const int32_t* zeros,
                                           const int group_size, const int m, const int n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int half_n = n / 2;
  if (tid >= m * half_n)
    return;
  float2 weight_int2 = FETCH_FLOAT2(qweight[tid * 2]);
  uint32_t weight_v1 = *reinterpret_cast<uint32_t*>(&weight_int2.x);
  uint32_t weight_v2 = *reinterpret_cast<uint32_t*>(&weight_int2.y);

  int col_ind = (tid % half_n) * 2;
  int weight_in_row = tid / half_n * 8;
  half2 scale_v = FETCH_HALF2(scale[weight_in_row / group_size * n + col_ind]);
  uint32_t zero_v = zeros[weight_in_row / group_size * (n / 8) + (col_ind) / 8];
  int zero_ind = col_ind % 8;
  uint8_t zv1 = (zero_v >> (zero_ind * 4)) & 15;
  uint8_t zv2 = (zero_v >> (zero_ind * 4 + 4)) & 15;
  half2 scale_zeros = __hmul2(__halves2half2(__short2half_rn(zv1), __short2half_rn(zv2)), scale_v);
  half2* out_h2 = reinterpret_cast<half2*>(out);
#pragma unroll
  for (int i = 0; i < 32 / 4; i++) {
    uint8_t wv1 = (weight_v1 >> (i * 4)) & 15;
    uint8_t wv2 = (weight_v2 >> (i * 4)) & 15;
    half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
    out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
  }
}

void DequantWeightNbit(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const int32_t* zeros_data,
    void* weight_out,
    uint32_t MATRIX_K,
    uint32_t MATRIX_N,
    uint32_t groupsize) {
  dim3 gridDim = {(MATRIX_N * MATRIX_K + kBlockSize * 2 - 1) / kBlockSize / 2};
  dim3 blockDim = {kBlockSize};

  kDequantizeAndUnpackWeight<half><<<gridDim, blockDim, 0, stream>>>(
      (half*)weight_out, qweight_i32, (half*)scales_data, zeros_data,
      groupsize, MATRIX_K, MATRIX_N);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
