// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "quant_nbit_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
const int width_element_per_block = 32 * 2;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if (WarpSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (WarpSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (WarpSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (WarpSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (WarpSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}
template <typename T>
__global__ void BatchGemv(T* out, const T* inA, const uint32_t* inB, const T* scales, const uint32_t* qzeros,
                          uint32_t groupsize, const uint32_t MATRIX_M, const uint32_t MATRIX_K, const uint32_t MATRIX_N) {
  const uint32_t block_k = ((MATRIX_K + 31) / 32 + 7) / 8 * 8;

  int bid = blockIdx.x;
  __shared__ float bsum[2][32][32 + 1];
  float sum[2] = {0, 0};
  int y_start = threadIdx.y * block_k;

  half2 res2 = {};
  half2 res2_1 = {};

  const half2* inA_start = (const half2*)(inA + blockIdx.y * MATRIX_K + y_start);

  int n_offset_x = bid * width_element_per_block + threadIdx.x * 2;

  int start_group_id = (y_start / groupsize);
  int compressed_idx = threadIdx.x % 4;
  half2 scale = ((const half2*)(scales + start_group_id * MATRIX_N + n_offset_x))[0];
  uint32_t qzero_p = qzeros == nullptr ? 0x88888888:((qzeros + n_offset_x / 8 +
                                                     start_group_id * ((MATRIX_N + 7) / 8)))[0];
  half2 hzero = __halves2half2(
      __int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
      __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
  half2 scale_h0 = __half2half2(scale.x);
  half2 scale_h1 = __half2half2(scale.y);
  half2 hzero_scale_0 = __half2half2(hzero.x * scale.x);
  half2 hzero_scale_1 = __half2half2(hzero.y * scale.y);

#pragma unroll
  for (int i = 0; i < block_k / 2; i += 4) {  // read half2 * 4
    res2 = {};
    res2_1 = {};
    int k_offset = y_start + i * 2;
    int g_id = k_offset / groupsize;

    if (g_id > start_group_id) {
      scale = ((const half2*)(scales + g_id * MATRIX_N + n_offset_x))[0];
      qzero_p = ((qzeros + n_offset_x / 8 +
                  g_id * ((MATRIX_N + 7) / 8)))[0];
      hzero = __halves2half2(
          __int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
          __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
      scale_h0 = __half2half2(scale.x);
      scale_h1 = __half2half2(scale.y);
      hzero_scale_0 = __half2half2(hzero.x * scale.x);
      hzero_scale_1 = __half2half2(hzero.y * scale.y);
      start_group_id = g_id;
    }

    const uint32_t* hinB = inB + n_offset_x + k_offset / 8 * MATRIX_N;
    uint32_t vbInt1 =
        (n_offset_x < MATRIX_N && (k_offset < MATRIX_K)) ? hinB[0] : int32_t(0);
    uint32_t vbInt2 = (n_offset_x + 1 < MATRIX_N && (k_offset < MATRIX_K))
                          ? (hinB)[1]
                          : int32_t(0);
    half2 vb[8];
    const uint8_t* qweight_p1 = (const uint8_t*)&vbInt1;
    const uint8_t* qweight_p2 = (const uint8_t*)&vbInt2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // vb[j] = __halves2half2(__int2half_rn(((vbInt1 >> (j * 8))) & 0xF),
      //                        __int2half_rn(((vbInt1) >> (j*8+4)) & 0xF));
      // vb[j + 4] = __halves2half2(__int2half_rn(((vbInt2)>>(j*8)) & 0xF),
      //                            __int2half_rn((((vbInt2) >> (j*8+4))) &
      //                            0xF));
      vb[j] = __halves2half2(__int2half_rn(((*(qweight_p1 + j))) & 0xF),
                             __int2half_rn(((*(qweight_p1 + j)) >> 4) & 0xF));
      vb[j + 4] =
          __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                         __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
    }

    half2 va[4];
    va[0] = (k_offset < MATRIX_K) ? ((inA_start))[i] : res2;
    va[1] = (k_offset + 1 < MATRIX_K) ? ((inA_start))[i + 1] : res2;
    va[2] = (k_offset + 2 < MATRIX_K) ? ((inA_start))[i + 2] : res2;
    va[3] = (k_offset + 3 < MATRIX_K) ? ((inA_start))[i + 3] : res2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      vb[j] = __hfma2(scale_h0, vb[j], -hzero_scale_0);
      res2 = __hfma2(va[j], vb[j], res2);
      vb[4 + j] = __hfma2(scale_h1, vb[4 + j], -hzero_scale_1);
      res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
    }

    sum[0] += __half2float(res2.x) + __half2float(res2.y);
    sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
  }
  // sum[0] += __half2float(res2.x);
  // sum[1] +=  __half2float(res2.y);
  bsum[0][threadIdx.x][threadIdx.y] = sum[0];
  bsum[1][threadIdx.x][threadIdx.y] = sum[1];

  __syncthreads();
  sum[0] = 0;
  sum[1] = 0;

#pragma unroll
  for (int i = 0; i < 2; i++) {
    sum[i] = bsum[i][threadIdx.y][threadIdx.x];
    __syncthreads();
    sum[i] = warpReduceSum<32>(sum[i]);
    if (threadIdx.x == 0) {
      out[+blockIdx.y * MATRIX_N + bid * width_element_per_block +
          threadIdx.y * 2 + i] = __float2half_rn(sum[i]);
    }
  }
}


constexpr int kBlockOutput = 32;
constexpr int kMaxInputBatchInThread = 1;

template <typename scalar_t, int WBITS>
__global__ void Gemv_g(const scalar_t* __restrict__ input,
                       const int* __restrict__ qweight, scalar_t* __restrict__ output,
                       const scalar_t* __restrict__ scales,
                       const int* __restrict__ qzeros,
                       const int* __restrict__ g_idx, uint32_t mat_m,
                       uint32_t mat_k, uint32_t mat_n, uint32_t zero_width) {
  const int num_thread_group = kBlockSize / kNumWaves;
  const int thread_num_k = (mat_k + num_thread_group - 1) / num_thread_group;
  const int thread_idx_group = threadIdx.y;
  const int thread_group_start = thread_idx_group * thread_num_k;

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  const int weight_x = blockIdx.x * kBlockOutput + threadIdx.x;

  __shared__ float blocksum[kMaxInputBatchInThread][num_thread_group]
                           [kBlockOutput];
  float sum[kMaxInputBatchInThread];
#pragma unroll
  for (int bid = 0; bid < kMaxInputBatchInThread; bid++) {
    sum[bid] = 0;
  }
  const int end_k = min(mat_k, thread_group_start + thread_num_k);
  int input_start_y = blockIdx.y * kMaxInputBatchInThread;
  int input_end_y = min(mat_m, input_start_y + kMaxInputBatchInThread);
  int len_input_y = input_end_y - input_start_y;
  for (int weight_y = thread_group_start; weight_y < end_k; weight_y++) {
    scalar_t input_vec[kMaxInputBatchInThread];
    for (int bid = 0; bid < len_input_y; bid++) {
      input_vec[bid] = input[(input_start_y + bid) * mat_k + weight_y];
    }
    int scale_row = g_idx[weight_y];

    scalar_t scale_v = scales[scale_row * mat_n + weight_x];
    uint32_t zero_v =
        qzeros == nullptr
            ? 0x88888888
            : qzeros[scale_row * zero_width + (weight_x / compress_group_size)];
    int zero_ind = weight_x % compress_group_size;
    uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;

    scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1));

    uint32_t weight_int = qweight[(weight_y / compress_group_size) * mat_n + weight_x];
    int weight_ind = (weight_y) % compress_group_size;
    uint8_t wv1 = (weight_int >> (weight_ind * WBITS)) & max_num_in_bits;
    scalar_t wv = __ushort2half_rn(wv1);
    scalar_t weight = __hfma(wv, scale_v, -scale_zeros);
    // sum = __hfma(weight, input_v, sum);
    for (int bid = 0; bid < len_input_y; bid++) {
      sum[bid] += __half2float(weight * input_vec[bid]);
    }
  }
  for (int bid = 0; bid < len_input_y; bid++) {
    if constexpr (!std::is_same<scalar_t, float>::value) {
      blocksum[bid][thread_idx_group][threadIdx.x] = sum[bid];  //__half2float(sum);
    } else {
      blocksum[bid][thread_idx_group][threadIdx.x] = sum[bid];
    }
  }
  for (unsigned int s = 1; s < num_thread_group; s *= 2) {
    __syncthreads();
    int index = 2 * s * thread_idx_group;
    if (index < num_thread_group) {
      for (int bid = 0; bid < len_input_y; bid++) {
        blocksum[bid][index][threadIdx.x] +=
            blocksum[bid][index + s][threadIdx.x];
      }
    }
  }
  for (int bid = 0; bid < len_input_y; bid++) {
    if (thread_idx_group == 0) {
      if constexpr (!std::is_same<scalar_t, float>::value) {
        output[(input_start_y + bid) * mat_n + blockIdx.x * kBlockOutput +
               threadIdx.x] = __float2half_rn(blocksum[bid][0][threadIdx.x]);
      } else {
        output[(input_start_y + bid) * mat_n + blockIdx.x * kBlockOutput +
               threadIdx.x] = blocksum[bid][0][threadIdx.x];
      }
    }
  }
}

void Q4bitGemv(
    cudaStream_t stream,
    const void* vec_data,
    const int32_t* mat_data,
    void* mul_out_data,
    const void* scales_data,
    const int32_t* zeros_data,
    uint32_t MATRIX_M,
    uint32_t MATRIX_K,
    uint32_t MATRIX_N,
    uint32_t groupsize) {
  const int block_k = ((MATRIX_K + 31) / 32 + 7) / 8 * 8;

  dim3 gridDim = {(MATRIX_N + width_element_per_block - 1) / width_element_per_block, MATRIX_M};
  dim3 blockDim = {32, (MATRIX_K + block_k - 1) / block_k};
  BatchGemv<half><<<gridDim, blockDim, 0, stream>>>(
      static_cast<half*>(mul_out_data), static_cast<const half*>(vec_data),
      reinterpret_cast<const uint32_t*>(mat_data), static_cast<const half*>(scales_data),
      reinterpret_cast<const uint32_t*>(zeros_data), groupsize, MATRIX_M, MATRIX_K, MATRIX_N);
}


template <typename T>
__forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

void NbitGemvGidx(
    cudaStream_t stream,
    const void* input,
    const int32_t* qweight,
    void* output,
    const void* scales,
    const int32_t* qzeros,
    const int32_t* g_idx,
    const int64_t* shapes) {
  auto matricx_m = static_cast<uint32_t>(shapes[0]);
  auto matricx_k = static_cast<uint32_t>(shapes[1]);
  auto matricx_n = static_cast<uint32_t>(shapes[2]);
  auto zero_width = static_cast<uint32_t>(shapes[3]);

  dim3 blocks(ceil_div<uint32_t>(matricx_n, kBlockOutput),
              ceil_div<uint32_t>(matricx_m, kMaxInputBatchInThread));
  dim3 threads(kBlockOutput, kBlockSize / kBlockOutput);

  using scalar_t = __half;
  Gemv_g<scalar_t, 4><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const scalar_t*>(input), qweight, reinterpret_cast<scalar_t*> (output),
      reinterpret_cast<const scalar_t*>(scales), qzeros, g_idx,matricx_m, matricx_k, matricx_n, zero_width);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
