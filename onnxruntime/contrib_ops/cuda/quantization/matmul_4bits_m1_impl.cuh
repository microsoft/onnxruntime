// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1.cuh"
// Include after common.cuh defines SHARED_PROVIDER.
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T, int block_size, bool has_zero_point, bool TailN = false,
          bool direct_scales = false, int max_unroll = 16>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat4BitsKernelM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  static_assert(!direct_scales || (std::is_same_v<T, half> && block_size == 32 && !has_zero_point && !TailN));
  static_assert(max_unroll == 4 || max_unroll == 16);
  const T* b_scale_vec;
  uint8_t* b_zp_vec = nullptr;
  if constexpr (direct_scales) {
    b_scale_vec = scales_data + n_id * blocks_per_K;
  } else {
    extern __shared__ char shared_buffer[];
    T* staged_scales = reinterpret_cast<T*>(shared_buffer);
    const int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
    for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K;
         i += kColsPerThreadBlock * kWarpSize) {
      if constexpr (TailN) {
        const int remaining_cols = n - n_block_id * kColsPerThreadBlock;
        const int valid_cols = remaining_cols < kColsPerThreadBlock ? remaining_cols : kColsPerThreadBlock;
        staged_scales[i] = i < valid_cols * blocks_per_K ? scales_data[offset + i] : T(0.f);
      } else {
        staged_scales[i] = scales_data[offset + i];
      }
    }

    if constexpr (has_zero_point) {
      b_zp_vec = reinterpret_cast<uint8_t*>(staged_scales + kColsPerThreadBlock * blocks_per_K);
      const int b_zp_k = (blocks_per_K + 1) / 2;
      const int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
      for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k;
           i += kColsPerThreadBlock * kWarpSize) {
        b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
        b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
      }
      b_zp_vec += warp_id * b_zp_k * 2;
    }
    __syncthreads();
    if constexpr (TailN) {
      static_assert(!has_zero_point, "TailN is scale-only");
      // Every thread participates in the scale-loading barrier before an invalid output warp exits.
      if (n_id >= n) {
        return;
      }
    }
    b_scale_vec = staged_scales + warp_id * blocks_per_K;
  }

  a_data += m_id * k + (lane_id << 3);

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollStep = kUnroll * k_per_iter;                                             \
    const int k_unroll_bound = k - k % kUnrollStep;                                               \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {                                          \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        AccumulateEightElements4b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);        \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(max_unroll);
  if constexpr (max_unroll == 16) {
    UnRollReduction(4);
  }
  UnRollReduction(1);
#undef UnRollReduction

  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements4b(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4BitsM1(
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
  if (n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }

  const int blocks_per_K = (k + block_size - 1) / block_size;
  const size_t shared_mem_size =
      sizeof(T) * blocks_per_K * kColsPerThreadBlock +
      static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, 1);
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  if constexpr (std::is_same_v<T, half>) {
    // Cache the mode to keep launches consistent with captured graphs.
    static const int experiment = ParseEnvironmentVariableWithDefault<int>("ORT_MATMUL4BITS_M1_EXPERIMENT", 0);
    const int direct_unroll = SelectMatMul4BitsM1DirectUnroll(experiment, n, k, block_size, zero_points != nullptr);
    if (direct_unroll == 4) {
      MatMulFloat4BitsKernelM1<T, 32, false, false, true, 4><<<blocks, threads, 0, stream>>>(
          output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, blocks_per_K);
      return true;
    }
    if (direct_unroll == 16) {
      MatMulFloat4BitsKernelM1<T, 32, false, false, true, 16><<<blocks, threads, 0, stream>>>(
          output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, blocks_per_K);
      return true;
    }
  }

#define MATMUL_FLOAT4B_M1_DISPATCH(bs)                                                    \
  if (zero_points != nullptr) {                                                           \
    MatMulFloat4BitsKernelM1<T, bs, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, 1, n, k, blocks_per_K);   \
  } else {                                                                                \
    MatMulFloat4BitsKernelM1<T, bs, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, blocks_per_K);       \
  }

  if (block_size == 16) {
    MATMUL_FLOAT4B_M1_DISPATCH(16)
  } else if (block_size == 32) {
    MATMUL_FLOAT4B_M1_DISPATCH(32)
  } else if (block_size == 64) {
    MATMUL_FLOAT4B_M1_DISPATCH(64)
  } else if (block_size == 128) {
    MATMUL_FLOAT4B_M1_DISPATCH(128)
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }
#undef MATMUL_FLOAT4B_M1_DISPATCH
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
