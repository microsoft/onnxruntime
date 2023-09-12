// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_with_quant_weight.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

inline __device__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  const half2* a_half2 = reinterpret_cast<const half2*>(a);
  half2 v0 = __hfma2(__halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF)), scale_half2, zp_adjust2);
  half2 v1 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF)), scale_half2, zp_adjust2);
  half2 v2 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF)), scale_half2, zp_adjust2);
  half2 v3 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF)), scale_half2, zp_adjust2);
  v0 = __hmul2(v0, a_half2[0]);
  v1 = __hmul2(v1, a_half2[1]);
  v2 = __hfma2(v2, a_half2[2], v0);
  v3 = __hfma2(v3, a_half2[3], v1);
  v3 = __hadd2(v2, v3);
  return float(v3.x) + float(v3.y);
}

inline __device__ float AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a) {
  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;
  v0 = v0 * a[0];
  v1 = v1 * a[1];
  v2 = v2 * a[2];
  v3 = v3 * a[3];
  v4 = v4 * a[4] + v0;
  v5 = v5 * a[5] + v1;
  v6 = v6 * a[6] + v2;
  v7 = v7 * a[7] + v3;
  return v4 + v5 + v6 + v7;
}

constexpr int BLOCKSIZEN = 8;

template <class T, int group_size>
__global__ void MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k) {
  int n_block_id = blockIdx.x;
  int m_id = blockIdx.y;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int n_id = n_block_id * BLOCKSIZEN + warp_id;
  int group_count = (k + group_size - 1) / group_size;
  int thread_id = warp_id * 32 + lane_id;
  int k_iter = k / 256;

  extern __shared__ char shared_buffer[];

  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + BLOCKSIZEN * group_count);
  int offset = n_block_id * BLOCKSIZEN * group_count;
  for (int i = thread_id; i < BLOCKSIZEN * group_count; i += 256) {
    b_scale_vec[i] = scales_data[offset + i];
    b_zp_vec[i] = zero_points != nullptr ? zero_points[offset + i] : uint8_t(8);
  }
  __syncthreads();

  a_data += m_id * k;
  b_data_quant += n_id * group_count * (group_size / 2);

  float sum = 0.f;
  for (int k_step = 0; k_step < k_iter; k_step++) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_step * 128 + lane_id * 4));
    T scale = b_scale_vec[warp_id * group_count + (k_step * 256 + lane_id * 8) / group_size];
    uint8_t zp = b_zp_vec[warp_id * group_count + (k_step * 256 + lane_id * 8) / group_size];
    sum += AccumulateEightElements(value, scale, zp, a_data + (lane_id << 3));
    a_data += 256;
  }

  // handle reminder
  int k_id = k_iter * 256;
  int k_remainder = k - k_iter * 256;
  if (k_remainder > 0) {
    if (lane_id * 8 < k_remainder) {
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_iter * 128 + lane_id * 4));
      T scale = b_scale_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
      uint8_t zp = b_zp_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
      sum += AccumulateEightElements(value, scale, zp, a_data + (lane_id << 3));
    }
  }

  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4BitsWeight(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int group_size,
    cudaStream_t stream) {
  if (n % BLOCKSIZEN != 0 || k % 8 != 0) {
    return false;
  }
  dim3 blocks((n + BLOCKSIZEN - 1) / BLOCKSIZEN, m);
  dim3 threads(32, 8);
  int shared_mem_size = (sizeof(T) + 1) * ((k + group_size - 1) / group_size * 8);

  // printf("group size %d\n", group_size);
  // printf("shared_mem_size %d\n", shared_mem_size);
  if (16 == group_size) {
    MatMulFloatInt4Kernel<T, 16><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (32 == group_size) {
    MatMulFloatInt4Kernel<T, 32><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (64 == group_size) {
    MatMulFloatInt4Kernel<T, 64><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (128 == group_size) {
    MatMulFloatInt4Kernel<T, 128><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else {
    ORT_THROW("block size ", group_size, " is not supported");
  }

  return true;
}

template bool TryMatMul4BitsWeight<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

template bool TryMatMul4BitsWeight<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
