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

#define FETCH_UINT2(pointer) (reinterpret_cast<const uint2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<const half2*>(&(pointer))[0])

template <typename T, int WBITS>
__global__ void kDequantizeAndUnpackWeight248(T* out, const int32_t* qweight, const T* scale, const int32_t* zeros,
                                              const int group_size, const int in_features, const int n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  //const int qweight_rows = (in_features * WBITS + 31) / 32;
  const int half_n = n / 2;

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  uint2 weight_int2 = FETCH_UINT2(qweight[tid * 2]);
  uint32_t weight_v1 = weight_int2.x;
  uint32_t weight_v2 = weight_int2.y;

  int col_ind = (tid % half_n) * 2;
  int weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v = FETCH_HALF2(scale[weight_in_row / group_size * n + col_ind]);
  uint32_t zero_v = zeros[weight_in_row / group_size * (n / compress_group_size) + (col_ind) / compress_group_size];
  int zero_ind = col_ind % compress_group_size;
  uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;
  uint8_t zv2 = (zero_v >> (zero_ind * WBITS + WBITS)) & max_num_in_bits;
  half2 scale_zeros = __hmul2(__halves2half2(__short2half_rn(zv1), __short2half_rn(zv2)), scale_v);

  half2* out_h2 = reinterpret_cast<half2*>(out);
  // decompress weights
  int remains = in_features - weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  } else {
    for (int i = 0; i < remains; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  }
}

template <typename T, int WBITS>
__device__ __forceinline__ uchar2 iterator_qweight_v2(const T* ptr, int idx) {
  int start_bits = idx * WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  uchar2 res;
  if (first == second) {
    res.x = (ptr[first].x >> (start_bits)) & ((1 << WBITS) - 1);
    res.y = (ptr[first].y >> (start_bits)) & ((1 << WBITS) - 1);
    return res;
  } else {
    res.x = (ptr[first].x >> (start_bits));
    res.y = (ptr[first].y >> (start_bits));

    res.x |= ((ptr[second].x) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
    res.y |= ((ptr[second].y) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
    return res;
  }
}

template <typename T, int WBITS>
__global__ void DequantizeAndUnpackWeight3567_v2(T* out, const uint32_t* qweight, const T* scale, const uint32_t* zeros,
                                                 int group_size, const int in_features, const int row_n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  __shared__ uint2 qweight_shared[WBITS * kBlockSize];
  const int half_n = row_n / 2;

  const int group_row_n = half_n * (WBITS == 6 ? 3 : WBITS);
  int total_qw = qweight_rows * half_n;

  uint2* qweight_thread = qweight_shared + WBITS * threadIdx.x;

  int qweight_start = tid / half_n * group_row_n + tid % half_n;
  const uint2* qweigh2 = (const uint2*)qweight;
#pragma unroll
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + half_n * j;
    qweight_thread[j] = ind < total_qw ? (qweigh2[ind]) : uint2();
  }

  const int max_num_in_bits = (1 << WBITS) - 1;
  const int col_ind = (tid % half_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = (fp16_weight_in_row + compress_group_size) / group_size;

  // decompress scales
  const half2* scale2 = reinterpret_cast<const half2*>(scale);
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = (scale2[scale_zero_from_i * half_n + col_ind]);
  }

  // decompress zeros
  uchar2 zv1[4];
  int half_col_ind = col_ind * 2;
  const int zero_col_from = half_col_ind * WBITS / 32;
  const int zero_col_to = ((half_col_ind + 1) * WBITS - 1) / 32;
  const int zero_col_to_2 = ((half_col_ind + 2) * WBITS - 1) / 32;
  const int qzero_width = (row_n * WBITS + 32 - 1) / 32;
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    uint32_t zero_v = zeros[scale_zero_from_i * qzero_width + zero_col_from];
    const int zero_bits_last = (((half_col_ind)*WBITS) % 32);
    zv1[i].x = (zero_v >> zero_bits_last) & max_num_in_bits;
    if (zero_col_from != zero_col_to) {
      const int zero_bits_first = ((half_col_ind + 1) * WBITS) % 32;
      uint32_t zero_v1 = zeros[scale_zero_from * qzero_width + zero_col_to];
      zv1[i].x |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last);

      zv1[i].y = (zero_v1 >> zero_bits_first) & max_num_in_bits;
    } else {
      zv1[i].y = (zero_v >> (zero_bits_last + WBITS)) & max_num_in_bits;
    }

    if (zero_col_to != zero_col_to_2) {
      const int zero_bits_first = ((half_col_ind + 2) * WBITS) % 32;
      uint32_t zero_v1 = zeros[scale_zero_from * qzero_width + zero_col_to_2];
      zv1[i].y |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last - WBITS);
    }
  }

  half2 scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul2(__halves2half2(__ushort2half_rn(zv1[i].x), __ushort2half_rn(zv1[i].y)), scale_v[i]);
  }
  half2 scale_2 = scale_v[0];
  half2 scale_zeros_2 = scale_zeros[0];

  const int out_offset = ((fp16_weight_in_row)*half_n + col_ind);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size / 2; i++) {
      uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);
      uchar2 wv2 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, 16 + i);

      half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        half2 scale_2 = scale_v[i / group_size];
        half2 scale_zeros_2 = scale_zeros[i / group_size];
      }
      half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;

      wv = __halves2half2(__ushort2half_rn(wv2.x), __ushort2half_rn(wv2.y));
      if (group_size < 32) {
        half2 scale_2 = scale_v[(i + 16) / group_size];
        half2 scale_zeros_2 = scale_zeros[(i + 16) / group_size];
      }
      res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[(out_offset + (i + 16) * half_n)] = res;
    }
  } else {
    // decompress weights
    for (int i = 0; i < remains; i++) {
      uchar2 wv1 = iterator_qweight_v2<uint2, WBITS>(qweight_thread, i);

      half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        scale_2 = scale_v[i / group_size];
        scale_zeros_2 = scale_zeros[i / group_size];
      }
      half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;
    }
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
    uint32_t bits,
    uint32_t groupsize) {
  uint32_t compress_ratio = 32 / bits;
  if (bits != 2 && bits != 4 && bits != 8) {
    compress_ratio = 32;
  }
  dim3 gridDim = {(MATRIX_N / 2 * ((MATRIX_K + compress_ratio - 1) / compress_ratio) + kBlockSize - 1) / kBlockSize};
  dim3 blockDim = {kBlockSize};
  switch (bits) {
    case 2:
      kDequantizeAndUnpackWeight248<half, 2><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, qweight_i32, (half*)scales_data, zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 4:
      kDequantizeAndUnpackWeight248<half, 4><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, qweight_i32, (half*)scales_data, zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 8:
      kDequantizeAndUnpackWeight248<half, 8><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, qweight_i32, (half*)scales_data, zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 3:
      DequantizeAndUnpackWeight3567_v2<half, 3><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, (const uint32_t*)qweight_i32, (half*)scales_data, (const uint32_t*)zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 5:
      DequantizeAndUnpackWeight3567_v2<half, 5><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, (const uint32_t*)qweight_i32, (half*)scales_data, (const uint32_t*)zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 6:
      DequantizeAndUnpackWeight3567_v2<half, 6><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, (const uint32_t*)qweight_i32, (half*)scales_data, (const uint32_t*)zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    case 7:
      DequantizeAndUnpackWeight3567_v2<half, 7><<<gridDim, blockDim, 0, stream>>>(
          (half*)weight_out, (const uint32_t*)qweight_i32, (half*)scales_data, (const uint32_t*)zeros_data,
          groupsize, MATRIX_K, MATRIX_N);
      break;
    default:
      break;
  }

}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
