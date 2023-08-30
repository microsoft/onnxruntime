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
struct EightElementsDequant<float>{
  float2 values[4];
  inline __device__ void Dequant(uint32_t values_quant, float2 scale, float2 scale_x_zp){
    values[0] = {float(values_quant & 0xF) * scale.x + scale_x_zp.x, float((values_quant>>4) & 0xF) * scale.y + scale_x_zp.y};
    values[1] = {float((values_quant>>8) & 0xF) * scale.x + scale_x_zp.x, float((values_quant>>12) & 0xF) * scale.y + scale_x_zp.y};
    values[1] = {float((values_quant>>16) & 0xF) * scale.x + scale_x_zp.x, float((values_quant>>20) & 0xF) * scale.y + scale_x_zp.y};
    values[1] = {float((values_quant>>24) & 0xF) * scale.x + scale_x_zp.x, float((values_quant>>28) & 0xF) * scale.y + scale_x_zp.y};
  }
};

template <>
struct EightElementsDequant<half>{
  half2 values[4];
  inline __device__ void Dequant(uint32_t values_quant, half2 scales, half2 scale_x_zp){
    values[0] = __hfma2(__halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant>>4) & 0xF)), scales, scale_x_zp);
    values[1] = __hfma2(__halves2half2(__uint2half_rn((values_quant>>8) & 0xF), __uint2half_rn((values_quant>>12) & 0xF)), scales, scale_x_zp);
    values[2] = __hfma2(__halves2half2(__uint2half_rn((values_quant>>16) & 0xF), __uint2half_rn((values_quant>>20) & 0xF)), scales, scale_x_zp);
    values[3] = __hfma2(__halves2half2(__uint2half_rn((values_quant>>24) & 0xF), __uint2half_rn((values_quant>>28) & 0xF)), scales, scale_x_zp);
  }
};

template<class Scalar>
struct Scalar2;

template<>
struct Scalar2<float>{

  using type = float2;
  inline __device__ static float2 MakeScalar2(float f){
    return float2{f, f};
  }

    inline __device__ static float2 MulAdd(float2 a, float2 b, float2 c){
    return {a.x*b.x + c.x, a.y*b.y+c.y};
  }
};

template<>
struct Scalar2<half>{
  using type = half2;
  inline __device__ static half2 MakeScalar2(half h){
    return __halves2half2(h, h);
  }

  inline __device__ static half2 MulAdd(half2 a, half2 b, half2 c){
    return __hfma2(a, b, c);
  }

};

template <class T, int block_size>
__global__ void MatMul4BitsWeightKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k) {
  int k_block_id = blockIdx.x;
  int n_block_id = blockIdx.y;
  int m_id = blockIdx.z;
  int n_id = n_block_id * blockDim.x + threadIdx.x;
  int k_id = k_block_id * block_size;

  if (n_id >= n) {
    return;
  }

  // load A to share
  __shared__ T a_data_vec[block_size];
  a_data += m_id * k + k_id;

  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    //if (i + k_id < k) {
      a_data_vec[i] = a_data[i];
    //} else {
    //  a_data_vec[i] = static_cast<T>(0.f);
    //}
  }
  __syncthreads();
  // dequantize a blob
  T weight[block_size];
  const uint8_t* quant_data_cur = b_data_quant + (n_id * gridDim.x + k_block_id) * block_size / 2;
  T scale = *(scales_data + n_id * gridDim.x + k_block_id);
  T zero_point = static_cast<T>(zero_points ? float(zero_points[n_id * gridDim.x + k_block_id]) : 8.f);
  T scale_zero_point = -scale * zero_point;
  typename Scalar2<T>::type scale_zero_point_pair = Scalar2<T>::MakeScalar2(scale_zero_point);
  typename Scalar2<T>::type scale_pair = Scalar2<T>::MakeScalar2(scale);

  typename Scalar2<T>::type res_pair;
  const typename Scalar2<T>::type* a_data_vec_2 = reinterpret_cast<const typename Scalar2<T>::type*>(a_data_vec);
  for (int kk = 0; kk < block_size; kk += 8) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(quant_data_cur+kk));
    EightElementsDequant<T> eight_elements;
    eight_elements.Dequant(value, scale_pair, scale_zero_point_pair);

    res_pair = Scalar2<T>::MulAdd(eight_elements.values[0], a_data_vec_2[kk/2], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[1], a_data_vec_2[kk/2 + 1], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[2], a_data_vec_2[kk/2 + 1], res_pair);
    res_pair = Scalar2<T>::MulAdd(eight_elements.values[3], a_data_vec_2[kk/2 + 1], res_pair);
  }

  atomicAdd(output + m_id * n + n_id, res_pair.x + res_pair.y);
}

template <class T>
Status MatMul4BitsWeight(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  constexpr int n_block_size = 128;
  dim3 blocks((k + block_size - 1) / block_size, (n + n_block_size - 1) / n_block_size, m);
  dim3 threads(min(n, n_block_size));

  if (16 == block_size) {
    MatMul4BitsWeightKernel<T, 16><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (32 == block_size) {
    MatMul4BitsWeightKernel<T, 32><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (64 == block_size) {
    MatMul4BitsWeightKernel<T, 64><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (128 == block_size) {
    MatMul4BitsWeightKernel<T, 128><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block size ", block_size, " is not supported");
  }

  return Status::OK();
}

template Status MatMul4BitsWeight<float>(
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

template Status MatMul4BitsWeight<half>(
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
