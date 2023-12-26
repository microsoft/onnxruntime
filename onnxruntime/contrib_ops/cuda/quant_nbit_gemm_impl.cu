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

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 700
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
//__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
//    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
//    unsigned int old = *address_as_ui;
//    unsigned int assumed;
//
//    do {
//        assumed = old;
//        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
//        hsum += val;
//        old = reinterpret_cast<size_t>(address) & 2
//                 ? (old & 0xffff) | (hsum << 16)
//                 : (old & 0xffff0000) | hsum;
//        old = atomicCAS(address_as_ui, assumed, old);
//
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//}
#endif
#endif

const int BLOCKWIDTH = 64;
const int BLOCKHEIGHT4 = 8;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const scalar_t* __restrict__ vec,
    const int* __restrict__ mat,
    scalar_t* __restrict__ mul,
    const scalar_t* __restrict__ scales,
    const int* __restrict__ zeros,
    const int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width) {
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 8;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  scalar_t weight[BLOCKWIDTH];

  for (k = 0; k < BLOCKWIDTH; ++k) {
    int k_w = (k / 8);
    int k_bit = (k % 8) * 4;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);

    weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b) {
    res = 0;

    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
    for (k = 0; k < BLOCKWIDTH; ++k) {
      res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}
void vecquant4matmul_cuda(
    cudaStream_t stream,
    const void* vec,
    const int* mat,
    void* mul,
    const void* scales,
    const int* zeros,
    const int* g_idx,
    int64_t* shape) {
  int batch = shape[0];//vec.size(0);
  int vec_height = shape[1];//vec.size(1);
  int height = shape[2];    // mat.size(0);
  int width = shape[3];     // mat.size(1);
  int zero_width = shape[4];  // zeros.size(1);

  dim3 blocks(
      (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
      (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
  dim3 threads(BLOCKWIDTH);

using scalar = half;
VecQuant4MatMulKernel<<<blocks, threads, 0, stream>>>(
    (const scalar*)vec, (const int*)mat, (scalar*)mul,
    (const scalar*)scales, (const int*)zeros, (const int*)g_idx,
    batch, vec_height, height, width, zero_width);
}

  }  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
