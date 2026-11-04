// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "contrib_ops/cuda/quantization/matmul_8bits_m1.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kM1ColsPerThreadBlock8Bits = 8;
constexpr int kM1ElementsPerThreadPerIteration8Bits = 8;
constexpr int kM1WarpSize8Bits = onnxruntime::cuda::GPU_WARP_SIZE;
constexpr uint8_t kM1DefaultZeroPoint8Bits = 128;

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, half scale, uint8_t zp,
                                                          const half* a, float* sums_f) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
  const uint32_t magic = 0x64646464u;
  const uint32_t lo = static_cast<uint32_t>(values_quant);
  const uint32_t hi = static_cast<uint32_t>(values_quant >> 32);
  const half2 scale_h2 = __half2half2(scale);
  const half2 zp_h2 = __half2half2(__ushort_as_half(static_cast<uint16_t>(0x6400u | zp)));
  const half2 q[4] = {bit_cast<half2>(__byte_perm(lo, magic, 0x4140)),
                      bit_cast<half2>(__byte_perm(lo, magic, 0x4342)),
                      bit_cast<half2>(__byte_perm(hi, magic, 0x4140)),
                      bit_cast<half2>(__byte_perm(hi, magic, 0x4342))};
  const half2* a_h2 = reinterpret_cast<const half2*>(a);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 av = __half22float2(a_h2[i]);
    const float2 bv = __half22float2(__hmul2(__hsub2(q[i], zp_h2), scale_h2));
    sums_f[2 * i] = fmaf(av.x, bv.x, sums_f[2 * i]);
    sums_f[2 * i + 1] = fmaf(av.y, bv.y, sums_f[2 * i + 1]);
  }
#else
  const float scale_f = __half2float(scale);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint8_t q = static_cast<uint8_t>(values_quant >> (i * 8));
    const float b = (static_cast<float>(q) - static_cast<float>(zp)) * scale_f;
    sums_f[i] = fmaf(__half2float(a[i]), b, sums_f[i]);
  }
#endif
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, float scale, uint8_t zp,
                                                          const float* a, float* sums_f) {
  const float4 a0 = *reinterpret_cast<const float4*>(a);
  const float4 a1 = *reinterpret_cast<const float4*>(a + 4);
  const float zp_adjust = -scale * static_cast<float>(zp);
  const float4 av[2] = {a0, a1};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float ai = reinterpret_cast<const float*>(av)[i];
    const uint8_t q = static_cast<uint8_t>(values_quant >> (i * 8));
    const float b = static_cast<float>(q) * scale + zp_adjust;
    sums_f[i] = fmaf(ai, b, sums_f[i]);
  }
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, nv_bfloat16 scale, uint8_t zp,
                                                          const nv_bfloat16* a, float* sums_f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const float scale_f = __bfloat162float(scale);
  const float zp_adjust = -scale_f * static_cast<float>(zp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint8_t q = static_cast<uint8_t>(values_quant >> (i * 8));
    const float b = static_cast<float>(q) * scale_f + zp_adjust;
    sums_f[i] = fmaf(__bfloat162float(a[i]), b, sums_f[i]);
  }
#else
  (void)values_quant;
  (void)scale;
  (void)zp;
  (void)a;
  (void)sums_f;
#endif
}

template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kM1WarpSize8Bits* kM1ColsPerThreadBlock8Bits) MatMulFloat8bKernelM1(
    T* output, const T* a_data, const uint8_t* b_data_quant, const T* scales_data, const uint8_t* zero_points,
    int n, int k, int blocks_per_K) {
  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int n_id = blockIdx.x * kM1ColsPerThreadBlock8Bits + warp_id;
  if (n_id >= n) return;

  extern __shared__ char shared_buffer[];
  T* scales = reinterpret_cast<T*>(shared_buffer);
  [[maybe_unused]] uint8_t* zps = nullptr;
  if constexpr (has_zero_point) zps = reinterpret_cast<uint8_t*>(scales + kM1ColsPerThreadBlock8Bits * blocks_per_K);

  for (int i = threadIdx.y * kM1WarpSize8Bits + threadIdx.x; i < kM1ColsPerThreadBlock8Bits * blocks_per_K;
       i += kM1ColsPerThreadBlock8Bits * kM1WarpSize8Bits) {
    const int column = i / blocks_per_K;
    const int block = i % blocks_per_K;
    const int current_n = blockIdx.x * kM1ColsPerThreadBlock8Bits + column;
    if (current_n < n) {
      const int64_t index = static_cast<int64_t>(current_n) * blocks_per_K + block;
      scales[i] = scales_data[index];
      if constexpr (has_zero_point) zps[i] = zero_points[index];
    }
  }
  __syncthreads();

  const int lane_offset = lane_id * kM1ElementsPerThreadPerIteration8Bits;
  const T* a_thread = a_data + lane_offset;
  const uint8_t* b_column = b_data_quant + static_cast<int64_t>(n_id) * blocks_per_K * block_size;
  const T* scale_thread = scales + warp_id * blocks_per_K;
  [[maybe_unused]] const uint8_t* zp_thread = nullptr;
  if constexpr (has_zero_point) zp_thread = zps + warp_id * blocks_per_K;

  float sums[kM1ElementsPerThreadPerIteration8Bits] = {0.0f};
  constexpr int k_per_iter = kM1WarpSize8Bits * kM1ElementsPerThreadPerIteration8Bits;
  int k_id = 0;
  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    const int block = (lane_offset + k_id) / block_size;
    const uint8_t* b = b_column + lane_offset + k_id;
    const uint8_t zp = has_zero_point ? zp_thread[block] : kM1DefaultZeroPoint8Bits;
    AccumulateEightElements8b(*reinterpret_cast<const uint64_t*>(b), scale_thread[block], zp, a_thread + k_id, sums);
  }
  if (lane_offset + k_id < k) {
    const int block = (lane_offset + k_id) / block_size;
    const uint8_t zp = has_zero_point ? zp_thread[block] : kM1DefaultZeroPoint8Bits;
    AccumulateEightElements8b(*reinterpret_cast<const uint64_t*>(b_column + lane_offset + k_id), scale_thread[block],
                              zp, a_thread + k_id, sums);
  }

  float total = 0.0f;
#pragma unroll
  for (float sum : sums) total += sum;
  using BlockReduce = cub::WarpReduce<float>;
  __shared__ typename BlockReduce::TempStorage temp_storage[kM1ColsPerThreadBlock8Bits];
  total = BlockReduce(temp_storage[warp_id]).Sum(total);
  if (lane_id == 0) output[n_id] = static_cast<T>(total);
}

template <class T>
bool TryMatMul8BitsM1(T* output, const T* a_data, const uint8_t* b_data_quant, const T* scales_data,
                      const uint8_t* zero_points, int n, int k, int block_size, size_t shared_mem_per_block,
                      cudaStream_t stream) {
  if (n % kM1ColsPerThreadBlock8Bits != 0 || k % kM1ElementsPerThreadPerIteration8Bits != 0) return false;
  constexpr int k_per_iter = kM1WarpSize8Bits * kM1ElementsPerThreadPerIteration8Bits;
  if (k_per_iter % block_size != 0 || k % block_size != 0) return false;

  const int blocks_per_K = k / block_size;
  const size_t scale_zp_shared_mem = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) *
                                     static_cast<size_t>(blocks_per_K) * kM1ColsPerThreadBlock8Bits;
  const size_t total_shared_mem = scale_zp_shared_mem +
                                  static_cast<size_t>(kM1ColsPerThreadBlock8Bits) *
                                      sizeof(typename cub::WarpReduce<float>::TempStorage);
  if (total_shared_mem > shared_mem_per_block) return false;

  dim3 threads(kM1WarpSize8Bits, kM1ColsPerThreadBlock8Bits);
  dim3 blocks((n + kM1ColsPerThreadBlock8Bits - 1) / kM1ColsPerThreadBlock8Bits, 1);
#define MATMUL_FLOAT8B_M1_DISPATCH(bs)                                                  \
  if (zero_points != nullptr) {                                                         \
    MatMulFloat8bKernelM1<T, bs, true><<<blocks, threads, total_shared_mem, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, n, k, blocks_per_K);    \
  } else {                                                                              \
    MatMulFloat8bKernelM1<T, bs, false><<<blocks, threads, total_shared_mem, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, n, k, blocks_per_K);        \
  }

  if (block_size == 16) {
    MATMUL_FLOAT8B_M1_DISPATCH(16)
  } else if (block_size == 32) {
    MATMUL_FLOAT8B_M1_DISPATCH(32)
  } else if (block_size == 64) {
    MATMUL_FLOAT8B_M1_DISPATCH(64)
  } else if (block_size == 128) {
    MATMUL_FLOAT8B_M1_DISPATCH(128)
  } else if (block_size == 256) {
    MATMUL_FLOAT8B_M1_DISPATCH(256)
  } else {
    return false;
  }
#undef MATMUL_FLOAT8B_M1_DISPATCH
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
