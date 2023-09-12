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

template <class Scalar>
struct EightElementsDequant;

template <>
struct EightElementsDequant<float> {
  float2 values[4];
  inline __device__ void Dequant(uint32_t values_quant, float2 scale, float2 scale_x_zp) {
    values[0] = {float(values_quant & 0xF) * scale.x + scale_x_zp.x, float((values_quant >> 4) & 0xF) * scale.y + scale_x_zp.y};
    values[1] = {float((values_quant >> 8) & 0xF) * scale.x + scale_x_zp.x, float((values_quant >> 12) & 0xF) * scale.y + scale_x_zp.y};
    values[2] = {float((values_quant >> 16) & 0xF) * scale.x + scale_x_zp.x, float((values_quant >> 20) & 0xF) * scale.y + scale_x_zp.y};
    values[3] = {float((values_quant >> 24) & 0xF) * scale.x + scale_x_zp.x, float((values_quant >> 28) & 0xF) * scale.y + scale_x_zp.y};
  }

  inline __device__ void Dequant(uint32_t values_quant, float2 scale) {
    values[0] = {float(values_quant & 0xF) * scale.x, float((values_quant >> 4) & 0xF) * scale.y};
    values[1] = {float((values_quant >> 8) & 0xF) * scale.x, float((values_quant >> 12) & 0xF) * scale.y};
    values[2] = {float((values_quant >> 16) & 0xF) * scale.x, float((values_quant >> 20) & 0xF) * scale.y};
    values[3] = {float((values_quant >> 24) & 0xF) * scale.x, float((values_quant >> 28) & 0xF) * scale.y};
  }
};

template <>
struct EightElementsDequant<half> {
  half2 values[4];
  inline __device__ void Dequant(uint32_t values_quant, half2 scales, half2 scale_x_zp) {
    values[0] = __hfma2(__halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF)), scales, scale_x_zp);
    values[1] = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF)), scales, scale_x_zp);
    values[2] = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF)), scales, scale_x_zp);
    values[3] = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF)), scales, scale_x_zp);
  }

  inline __device__ void Dequant(uint32_t values_quant, half2 scales) {
    values[0] = __hmul2(__halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF)), scales);
    values[1] = __hmul2(__halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF)), scales);
    values[2] = __hmul2(__halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF)), scales);
    values[3] = __hmul2(__halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF)), scales);
  }
};

template <class Scalar>
struct Scalar2;

template <>
struct Scalar2<float> {
  using type = float2;
  inline __device__ static float2 MakeScalar2(float f) {
    return float2{f, f};
  }

  inline __device__ static float2 MulAdd(float2 a, float2 b, float2 c) {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
  }
};

template <>
struct Scalar2<half> {
  using type = half2;
  inline __device__ static half2 MakeScalar2(half h) {
    return __halves2half2(h, h);
  }

  inline __device__ static half2 MulAdd(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
  }
};

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

  if (n_id >= n) {
    return;
  }

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
  typename Scalar2<T>::type res_pair = Scalar2<T>::MakeScalar2(static_cast<T>(0.f));
  __shared__ T a_data_vec[256];
  const typename Scalar2<T>::type* a_data_vec_2 = reinterpret_cast<const typename Scalar2<T>::type*>(a_data_vec);
  for (int k_step = 0; k_step < k_iter; k_step++) {
    a_data_vec[thread_id] = a_data[k_step * 256 + thread_id];
    __syncthreads();
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_step * 128 + lane_id * 4));
    EightElementsDequant<T> eight_elements;
    typename Scalar2<T>::type scale_pair = Scalar2<T>::MakeScalar2(b_scale_vec[warp_id * group_count + (k_step * 256 + lane_id * 8) / group_size]);
    typename Scalar2<T>::type scale_zp_pair = Scalar2<T>::MakeScalar2(-scale_pair.x * static_cast<T>(b_zp_vec[warp_id * group_count + (k_step * 256 + lane_id * 8) / group_size]));
    eight_elements.Dequant(value, scale_pair, scale_zp_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[0], a_data_vec_2[(lane_id << 2) + 0], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[1], a_data_vec_2[(lane_id << 2) + 1], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[2], a_data_vec_2[(lane_id << 2) + 2], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[3], a_data_vec_2[(lane_id << 2) + 3], res_pair);
    __syncthreads();
  }

  // handle reminder
  int k_id = k_iter * 256;
  int k_remainder = k - k_iter * 256;
  if (k_remainder > 0) {
    if (thread_id < k_remainder) {
      a_data_vec[thread_id] = a_data[k_id + thread_id];
    } else {
      a_data_vec[thread_id] = static_cast<T>(.0f);
    }
    __syncthreads();

    if (lane_id * 8 < k_remainder) {
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_iter * 128 + lane_id * 4));
      EightElementsDequant<T> eight_elements;
      typename Scalar2<T>::type scale_pair = Scalar2<T>::MakeScalar2(b_scale_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size]);
      typename Scalar2<T>::type scale_zp_pair = Scalar2<T>::MakeScalar2(-scale_pair.x * static_cast<T>(b_zp_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size]));
      eight_elements.Dequant(value, scale_pair, scale_zp_pair);
      res_pair = Scalar2<T>::MulAdd(eight_elements.values[0], a_data_vec_2[(lane_id << 2) + 0], res_pair);
      res_pair = Scalar2<T>::MulAdd(eight_elements.values[1], a_data_vec_2[(lane_id << 2) + 1], res_pair);
      res_pair = Scalar2<T>::MulAdd(eight_elements.values[2], a_data_vec_2[(lane_id << 2) + 2], res_pair);
      res_pair = Scalar2<T>::MulAdd(eight_elements.values[3], a_data_vec_2[(lane_id << 2) + 3], res_pair);
    }
  }

  // warp reduction
  T sum = res_pair.x + res_pair.y;
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
  if (n % BLOCKSIZEN != 0) {
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
