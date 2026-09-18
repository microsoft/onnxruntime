// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_2bits_common.cuh"
#include "contrib_ops/cuda/quantization/matmul_2bits_m1.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// One warp per output channel, one row of A. Scales and zero points for the eight channels owned
// by the thread block are staged in shared memory; the zero points are unpacked to one byte per
// block there so the main loop indexes them without any bit arithmetic.
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize2b* kColsPerThreadBlock2b) MatMulFloat2bKernelM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int blocks_per_K) {
  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int n_block_id = blockIdx.x;
  const int n_id = n_block_id * kColsPerThreadBlock2b + warp_id;
  const int meta_count = kColsPerThreadBlock2b * blocks_per_K;

  extern __shared__ char shared_buffer[];
  T* b_scale_vec = reinterpret_cast<T*>(shared_buffer);
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + meta_count);

  const int64_t scale_offset = static_cast<int64_t>(n_block_id) * kColsPerThreadBlock2b * blocks_per_K;
  for (int i = warp_id * kWarpSize2b + lane_id; i < meta_count; i += kColsPerThreadBlock2b * kWarpSize2b) {
    b_scale_vec[i] = scales_data[scale_offset + i];
  }
  if constexpr (has_zero_point) {
    const int zp_bytes_per_col = (blocks_per_K + kElementsPerByte2b - 1) / kElementsPerByte2b;
    for (int i = warp_id * kWarpSize2b + lane_id; i < meta_count; i += kColsPerThreadBlock2b * kWarpSize2b) {
      const int col = i / blocks_per_K;
      const int blk = i % blocks_per_K;
      const uint8_t* zp_row =
          zero_points + static_cast<int64_t>(n_block_id * kColsPerThreadBlock2b + col) * zp_bytes_per_col;
      b_zp_vec[i] = UnpackZeroPoint2b(zp_row, blk);
    }
  }
  __syncthreads();

  const T* scale_thread = b_scale_vec + warp_id * blocks_per_K;
  const uint8_t* zp_thread = b_zp_vec + warp_id * blocks_per_K;
  const int lane_offset = lane_id * kElementsPerThreadPerIteration2b;
  const uint8_t* b_column =
      b_data_quant + static_cast<int64_t>(n_id) * blocks_per_K * (block_size / kElementsPerByte2b);

  typename Traits2b<T>::Acc acc{};
  constexpr int k_per_iter = kWarpSize2b * kElementsPerThreadPerIteration2b;

  int k_id = 0;
  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    const int k_offset = lane_offset + k_id;
    const int blk = k_offset / block_size;
    typename Traits2b<T>::Weights w;
    typename Traits2b<T>::Acts av;
    DequantizeSixteen2b(*reinterpret_cast<const uint32_t*>(b_column + k_offset / kElementsPerByte2b),
                        scale_thread[blk], has_zero_point ? zp_thread[blk] : kDefaultZeroPoint2b, w);
    LoadSixteen2b(a_data + k_offset, av);
    DotAccum2b(w, av, acc);
  }
  if (lane_offset + k_id < k) {
    const int k_offset = lane_offset + k_id;
    const int blk = k_offset / block_size;
    typename Traits2b<T>::Weights w;
    typename Traits2b<T>::Acts av;
    DequantizeSixteen2b(*reinterpret_cast<const uint32_t*>(b_column + k_offset / kElementsPerByte2b),
                        scale_thread[blk], has_zero_point ? zp_thread[blk] : kDefaultZeroPoint2b, w);
    LoadSixteen2b(a_data + k_offset, av);
    DotAccum2b(w, av, acc);
  }

  float sum = HorizontalAdd2b(acc);
  for (int offset = kWarpSize2b / 2; offset > 0; offset /= 2) {
    sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, offset);
  }
  if (lane_id == 0) {
    output[n_id] = static_cast<T>(sum);
  }
}

template <class T>
bool TryMatMul2BitsM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock2b != 0 || k % kElementsPerThreadPerIteration2b != 0) {
    return false;
  }
  // A thread's 16 codes must lie inside one quantization block, and the trailing partial block of
  // a K that is not a multiple of block_size is left to the dequantize + cuBLAS fallback.
  constexpr int k_per_iter = kWarpSize2b * kElementsPerThreadPerIteration2b;
  if (block_size < kElementsPerThreadPerIteration2b || k_per_iter % block_size != 0 || k % block_size != 0) {
    return false;
  }

  const int blocks_per_K = k / block_size;
  const size_t shared_mem_size = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) *
                                 static_cast<size_t>(blocks_per_K) * kColsPerThreadBlock2b;
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock2b);
  dim3 blocks(n / kColsPerThreadBlock2b, 1);

#define MATMUL_FLOAT2B_M1_DISPATCH(bs)                                                 \
  if (zero_points != nullptr) {                                                        \
    MatMulFloat2bKernelM1<T, bs, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, n, k, blocks_per_K);   \
  } else {                                                                             \
    MatMulFloat2bKernelM1<T, bs, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, n, k, blocks_per_K);       \
  }

  if (block_size == 16) {
    MATMUL_FLOAT2B_M1_DISPATCH(16)
  } else if (block_size == 32) {
    MATMUL_FLOAT2B_M1_DISPATCH(32)
  } else if (block_size == 64) {
    MATMUL_FLOAT2B_M1_DISPATCH(64)
  } else if (block_size == 128) {
    MATMUL_FLOAT2B_M1_DISPATCH(128)
  } else if (block_size == 256) {
    MATMUL_FLOAT2B_M1_DISPATCH(256)
  } else if (block_size == 512) {
    MATMUL_FLOAT2B_M1_DISPATCH(512)
  } else {
    return false;
  }
#undef MATMUL_FLOAT2B_M1_DISPATCH
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
