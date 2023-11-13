// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "matmul_bnb4.cuh"
#include "dequantize_blockwise_bnb4.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
__device__ inline float ScalarMulFloatOut(T a, T b);

template <>
__device__ inline float ScalarMulFloatOut(float a, float b) {
  return a * b;
}

template <>
__device__ inline float ScalarMulFloatOut(half a, half b) {
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
    return static_cast<float>(a * b);
  #else
    // half multiplication not supported
    return static_cast<float>(a) * static_cast<float>(b);
  #endif
}

template <>
__device__ inline float ScalarMulFloatOut(BFloat16 a, BFloat16 b) {
  return a * b;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// will use the native bfloat16 multiply instruction on sm_80+
template <>
__device__ inline float ScalarMulFloatOut(nv_bfloat16 a, nv_bfloat16 b) {
  return static_cast<float>(a * b);
}
#endif

#define num_values_4bit 32
template <class T, int THREADS, int BITS>
__global__ void kgemm_4bit_inference_naive(
    int M,
    int N,
    int K,
    const T* __restrict__ A,
    const uint8_t* B,
    const T* absmax,
    const T* datatype,
    T* out,
    int lda,
    int ldb,
    int ldc,
    int block_size) {
  // per threadblock:
  // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
  // 4 warps -> 4 loads per iter
  // 1x32 * 32x4 -> 1x4 outputs per thread block
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[THREADS / 32];

  const int warp_idx = threadIdx.x / 32;
  const int warp_lane = threadIdx.x % 32;
  const int row_B = (THREADS / 32) * blockIdx.x + warp_idx;
  const int num_values_8bit = num_values_4bit / 2;
  float local_C = 0.0f;

  uint8_t local_B_4bit[num_values_8bit];
  T local_B[num_values_4bit / 4];
  T local_A[num_values_4bit / 4];
  __shared__ T quant_map[16];
  T local_absmax = T(0.0f);

  for (int i = threadIdx.x; i < 16; i++) quant_map[i] = T(datatype[i]);
  __syncthreads();

  // A: [1, K]
  // B: [N, K]
  for (int inner_idx = warp_lane * num_values_4bit; inner_idx < K; inner_idx += 32 * num_values_4bit) {
    int inner_idx_halved = inner_idx / 2;
    int offset_B = ldb * row_B;
    int absidx = ((2 * offset_B) + inner_idx) / block_size;
    local_absmax = absmax[absidx];

    if (row_B < N) {
      if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
        // this is the most important for performance considerations
        reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] =
            reinterpret_cast<const int4*>(B)[(offset_B + (inner_idx_halved)) / (num_values_8bit)];
      } else {
        #pragma unroll
        for (int j = 0; j < (num_values_8bit); j++)
          if ((inner_idx_halved) + j < (K / 2))
            local_B_4bit[j] = B[offset_B + inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    } else {
      #pragma unroll
      for (int j = 0; j < (num_values_8bit); j++) local_B_4bit[j] = 0b01110111;
    }

    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int k = 0; k < num_values_8bit / 4; k++) {
        local_B[k * 2] = ScalarMul(quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4], local_absmax);
        local_B[k * 2 + 1] = ScalarMul(quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F], local_absmax);
      }

      if (inner_idx + (num_values_4bit / 4) + (i * num_values_4bit / 4) < K) {
        // this is also relatively important for performance
        if (BITS == 16) {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] =
              reinterpret_cast<const int4*>(A)[inner_idx / (num_values_4bit / 4) + i];
        } else {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] =
              reinterpret_cast<const int4*>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 0];
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] =
              reinterpret_cast<const int4*>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 1];
        }
      } else {
        #pragma unroll
        for (int k = 0; k < num_values_4bit / 4; k++) {
          if (inner_idx + (i * num_values_4bit / 4) + k < K)
            local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
          else
            local_A[k] = T(0.0f);
        }
      }

      // accumulate in float; small performance hit for Ampere, but lower error for outputs
      #pragma unroll
      for (int k = 0; k < num_values_4bit / 4; k++) {
        local_C += ScalarMulFloatOut(local_A[k], local_B[k]);
      }
    }
  }

  local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

  if (row_B < N && warp_lane == 0) out[row_B] = T(local_C);
}

bool CheckDims(int m, int k, int block_size) {
  if (k % block_size != 0 || m > 1) {
    return false;
  }
  // supported block_sizes are [4096, 2048, 1024, 512, 256, 128, 64, 32]
  if (block_size % 32 != 0 || block_size > 4096) {
    return false;
  }
  return true;
}

template <class T>
void Callkgemm_4bit_inference_naive(
    const T* quant_map,
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  int lda = k;
  int ldb = (k + 1) / 2;
  int ldc = n;
  int num_blocks = (n + 3) / 4;

  constexpr int bits = std::is_same_v<T, float> ? 32 : 16;
  kgemm_4bit_inference_naive<T, 128, bits><<<num_blocks, 128, 0, stream>>>(
      m, n, k, a_data, b_data_quant, absmax, quant_map, output, lda, ldb, ldc, block_size);
}

template <class T>
bool TryMatMulBnb4(
    const T* quant_map,
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  if (!CheckDims(m, k, block_size)) {
    return false;
  }

  Callkgemm_4bit_inference_naive<T>(
      quant_map, output, a_data, b_data_quant, absmax, m, n, k, block_size, stream);

  return true;
}

template bool TryMatMulBnb4<float>(
    const float* quant_map,
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

template bool TryMatMulBnb4<half>(
    const half* quant_map,
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

template <>
bool TryMatMulBnb4<BFloat16>(
    const BFloat16* quant_map,
    BFloat16* output,
    const BFloat16* a_data,
    const uint8_t* b_data_quant,
    const BFloat16* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  if (!CheckDims(m, k, block_size)) {
    return false;
  }

  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    Callkgemm_4bit_inference_naive<nv_bfloat16>(
        reinterpret_cast<const nv_bfloat16*>(quant_map),
        reinterpret_cast<nv_bfloat16*>(output),
        reinterpret_cast<const nv_bfloat16*>(a_data),
        b_data_quant,
        reinterpret_cast<const nv_bfloat16*>(absmax),
        m,
        n,
        k,
        block_size,
        stream);
  #else
    Callkgemm_4bit_inference_naive<BFloat16>(
        quant_map, output, a_data, b_data_quant, absmax, m, n, k, block_size, stream);
  #endif

  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
