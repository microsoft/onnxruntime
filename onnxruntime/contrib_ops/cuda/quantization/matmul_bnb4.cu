// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "matmul_bnb4.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define num_values_4bit 32
template <typename T, int THREADS, int BITS>
__global__ void kgemm_4bit_inference_naive(int M, int N, int K, const T* __restrict__ A, const unsigned char *B, const T *absmax, const T *datatype, T * out,  int lda, int ldb, int ldc, int block_size)
{

  // per threadblock:
  // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
  // 4 warps -> 4 loads per iter
  // 1x32 * 32x4 -> 1x4 outputs per thread block
  typedef cub::WarpReduce<float> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[THREADS/32];

  const int warp_idx = threadIdx.x / 32;
  const int warp_lane = threadIdx.x % 32;
  const int row_B = (THREADS/32)*blockIdx.x + warp_idx;
  const int num_values_8bit = num_values_4bit/2;
  float local_C = 0.0f;

  unsigned char local_B_4bit[num_values_8bit];
  T local_B[num_values_4bit/4];
  T local_A[num_values_4bit/4];
  __shared__ T quant_map[16];
	T local_absmax = T(0.0f);

  for(int i = threadIdx.x; i < 16; i++)
    quant_map[i] = T(datatype[i]);
  __syncthreads();

  // A: [1, K]
  // B: [N, K]
  for(int inner_idx = warp_lane*num_values_4bit; inner_idx < K; inner_idx += 32*num_values_4bit)
  {
    int inner_idx_halved = inner_idx/2;
    int offset_B = ldb*row_B;
    int absidx = ((2*offset_B)+inner_idx)/block_size;
	  local_absmax = __ldg(&(absmax[absidx]));

    if(row_B < N)
    {
      if((inner_idx_halved + num_values_8bit) < (K/2))
      {
        // this is the most important for performance considerations
        reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] = reinterpret_cast<const int4*>(B)[(offset_B+(inner_idx_halved))/(num_values_8bit)];
      }
      else
      {
        #pragma unroll
        for(int j = 0; j < (num_values_8bit); j++)
          if((inner_idx_halved) + j < (K/2))
            local_B_4bit[j] = B[offset_B+inner_idx_halved + j];
          else
            local_B_4bit[j] = 0b01110111;
      }
    }
    else
    {
      #pragma unroll
      for(int j = 0; j < (num_values_8bit); j++)
          local_B_4bit[j] = 0b01110111;
    }

    for(int i = 0; i < 4; i++)
    {
      #pragma unroll
      for(int k = 0; k < num_values_8bit/4; k++)
      {
        local_B[k*2] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] >> 4]*local_absmax;
        local_B[k*2 + 1] = quant_map[local_B_4bit[(i*num_values_8bit/4) + k] & 0x0F]*local_absmax;
      }

      if(inner_idx+(num_values_4bit/4) + (i*num_values_4bit/4) < K)
      {
        // this is also relatively important for performance
        if(BITS==16)
        {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<const int4*>(A)[inner_idx/(num_values_4bit/4) + i];
        }
        else
        {
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<const int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 0];
          reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] = reinterpret_cast<const int4*>(A)[inner_idx/(num_values_4bit/8) + (2*i) + 1];
        }

      }
      else
        #pragma unroll
        for(int k = 0; k < num_values_4bit/4; k++)
          if(inner_idx + (i*num_values_4bit/4) + k < K)
            local_A[k] = A[inner_idx + k + (i*num_values_4bit/4)];
          else
            local_A[k] = T(0.0f);


      // accumulate in float; small performance hit for Ampere, but lower error for outputs
      #pragma unroll
      for(int k = 0; k < num_values_4bit/4; k++)
      {
        local_C += (float)(local_A[k]*local_B[k]);
      }
    }
  }

  local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

  if(row_B < N && warp_lane == 0)
    out[row_B] = T(local_C);

}


template <class T>
bool TryMatMulBnb4(
    const T* quant_map,
    T* output,
    const T* a_data,
    const unsigned char* b_data_quant,
    const T* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream) {
  if (k % block_size != 0 || m > 1) {
    return false;
  }
  // supported block_sizes are [4096, 2048, 1024, 512, 256, 128, 64, 32]
  if (block_size % 32 != 0 || block_size > 4096) {
    return false;
  }

  int lda = k;
  int ldb = (k+1)/2;
  int ldc = n;
  int num_blocks = (n + 3) / 4;

  constexpr int bits = ::cuda::std::is_same_v<T, half> ? 16 : 32;
  kgemm_4bit_inference_naive<T, 128, bits><<<num_blocks, 128, 0, stream>>>(m, n, k, a_data, b_data_quant, absmax, quant_map, output, lda, ldb, ldc, block_size);

  return true;
}

template bool TryMatMulBnb4<float>(
    const float* quant_map,
    float* output,
    const float* a_data,
    const unsigned char* b_data_quant,
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
    const unsigned char* b_data_quant,
    const half* absmax,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
