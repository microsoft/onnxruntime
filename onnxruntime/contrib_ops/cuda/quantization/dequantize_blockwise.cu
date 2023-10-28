// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "dequantize_blockwise.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, half scale, half zp, half* output) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) half2 results[4];
  half v0 = __uint2half_rn(values_quant & 0xF);
  half v1 = __uint2half_rn((values_quant >> 4) & 0xF);
  results[0] = __halves2half2(v0, v1) * scale_half2 + zp_adjust2;

  half v2 = __uint2half_rn((values_quant >> 8) & 0xF);
  half v3 = __uint2half_rn((values_quant >> 12) & 0xF);
  results[1] = __halves2half2(v2, v3) * scale_half2 + zp_adjust2;

  half v4 = __uint2half_rn((values_quant >> 16) & 0xF);
  half v5 = __uint2half_rn((values_quant >> 20) & 0xF);
  results[2] = __halves2half2(v4, v5) * scale_half2 + zp_adjust2;

  half v6 = __uint2half_rn((values_quant >> 24) & 0xF);
  half v7 = __uint2half_rn((values_quant >> 28) & 0xF);
  results[3] = __halves2half2(v6, v7) * scale_half2 + zp_adjust2;
  *(reinterpret_cast<float4*>(output)) = *(reinterpret_cast<float4*>(results));
}

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, float scale, float zp, float* output) {
  float zp_adjust = -scale * zp;
  output[0] = float(values_quant & 0xF) * scale + zp_adjust;
  output[1] = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  output[2] = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  output[3] = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  output[4] = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  output[5] = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  output[6] = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  output[7] = float((values_quant >> 28) & 0xF) * scale + zp_adjust;
}

template <class T>
__global__ void Dequantize4BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const uint8_t* zero_points,
    int block_size,
    int blocks_per_threadblock,
    int shift) {
  int block_id = blockIdx.x * blocks_per_threadblock + ((threadIdx.x * 8) >> shift);
  int element_offset = block_id * block_size + ((threadIdx.x * 8) & ((1 << shift) - 1));
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / 2));
  T scale = *(scale_data + block_id);
  uint8_t zp = 8;
  if (zero_points) {
    zp = (block_id & 0x01) ? (zero_points[block_id / 2] >> 4) : (zero_points[block_id / 2] & 0x0f);
  }

  output = output + element_offset;
  DequantizeEightElements(quant_value, scale, static_cast<T>(zp), output);
}

template <class T>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const uint8_t* zero_points,  // shape: [N, (block_per_K + 1)/2]
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  // k is padded and equal to block_per_K * block_size
  ORT_ENFORCE(k % block_size == 0, "k must be a multiplier of block_size");
  constexpr int element_per_thread = 8;
  int blocks_per_threadblock = GridDim::maxThreadsPerBlock * element_per_thread / block_size;
  int blocks_per_K = k / block_size;
  int blocks_per_grid = static_cast<int>(CeilDiv(n * blocks_per_K, blocks_per_threadblock));
  int shift = static_cast<int>(log2f(float(block_size)));

  Dequantize4BitsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      output,
      quant_data,
      scales_data,
      zero_points,
      block_size,
      blocks_per_threadblock,
      shift);

  return Status::OK();
}

template Status Dequantize4Bits<float>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<half>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
