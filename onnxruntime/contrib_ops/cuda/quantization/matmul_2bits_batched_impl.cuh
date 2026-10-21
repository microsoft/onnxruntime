// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_2bits_batched.cuh"
#include "contrib_ops/cuda/quantization/matmul_2bits_common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kSmallMMax2b = 8;

// Register-tiled small-M kernel: each warp owns CtaN output channels and CtaM rows of A, so one
// dequantized weight chunk is reused across CtaM rows and one activation chunk across CtaN
// channels. Scales and zero points are read straight from global memory; unlike the m1 kernel
// there is no reuse across warps to justify staging them.
template <class T, int block_size, bool has_zero_point, int CtaM, int CtaN>
__global__ void __launch_bounds__(kWarpSize2b* kColsPerThreadBlock2b, 3) MatMulFloat2bKernelBatched(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  using Acc = typename Traits2b<T>::Acc;
  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int col_base = (blockIdx.x * kColsPerThreadBlock2b + warp_id) * CtaN;
  const int m_base = blockIdx.y * CtaM;
  const int valid = m - m_base;
  constexpr int k_per_iter = kWarpSize2b * kElementsPerThreadPerIteration2b;
  const int lane_offset = lane_id * kElementsPerThreadPerIteration2b;
  const int zp_bytes_per_col = (blocks_per_K + kElementsPerByte2b - 1) / kElementsPerByte2b;

  const T* a_base = a_data + static_cast<size_t>(m_base) * k + lane_offset;
  const uint8_t* b_ptr[CtaN];
#pragma unroll
  for (int c = 0; c < CtaN; ++c) {
    b_ptr[c] = b_data_quant +
               static_cast<size_t>(col_base + c) * blocks_per_K * (block_size / kElementsPerByte2b) + lane_offset / kElementsPerByte2b;
  }

  Acc acc[CtaM][CtaN];
#pragma unroll
  for (int r = 0; r < CtaM; ++r) {
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      acc[r][c] = Acc{};
    }
  }

  int k_id = 0;
  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    const int blk = (lane_offset + k_id) / block_size;
    typename Traits2b<T>::Weights w[CtaN];
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      uint8_t zp = kDefaultZeroPoint2b;
      if constexpr (has_zero_point) {
        zp = UnpackZeroPoint2b(zero_points + static_cast<size_t>(col_base + c) * zp_bytes_per_col, blk);
      }
      DequantizeSixteen2b(*reinterpret_cast<const uint32_t*>(b_ptr[c] + k_id / kElementsPerByte2b),
                          scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + blk], zp, w[c]);
    }
#pragma unroll
    for (int r = 0; r < CtaM; ++r) {
      if (r < valid) {
        typename Traits2b<T>::Acts av;
        LoadSixteen2b(a_base + static_cast<size_t>(r) * k + k_id, av);
#pragma unroll
        for (int c = 0; c < CtaN; ++c) {
          DotAccum2b(w[c], av, acc[r][c]);
        }
      }
    }
  }

  if (lane_offset + k_id < k) {
    const int blk = (lane_offset + k_id) / block_size;
    typename Traits2b<T>::Weights w[CtaN];
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      uint8_t zp = kDefaultZeroPoint2b;
      if constexpr (has_zero_point) {
        zp = UnpackZeroPoint2b(zero_points + static_cast<size_t>(col_base + c) * zp_bytes_per_col, blk);
      }
      DequantizeSixteen2b(*reinterpret_cast<const uint32_t*>(b_ptr[c] + k_id / kElementsPerByte2b),
                          scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + blk], zp, w[c]);
    }
#pragma unroll
    for (int r = 0; r < CtaM; ++r) {
      if (r < valid) {
        typename Traits2b<T>::Acts av;
        LoadSixteen2b(a_base + static_cast<size_t>(r) * k + k_id, av);
#pragma unroll
        for (int c = 0; c < CtaN; ++c) {
          DotAccum2b(w[c], av, acc[r][c]);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < CtaM; ++r) {
    if (r >= valid) continue;
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      float sum = HorizontalAdd2b(acc[r][c]);
      for (int offset = kWarpSize2b / 2; offset > 0; offset /= 2) {
        sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, offset);
      }
      if (lane_id == 0) {
        output[static_cast<size_t>(m_base + r) * n + (col_base + c)] = static_cast<T>(sum);
      }
    }
  }
}

}  // namespace

template <class T>
bool TryMatMul2BitsBatched(
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
  if (m < 2 || m > kSmallMMax2b) {
    return false;
  }
  const int cta_m = (m <= 2) ? 2 : (m <= 4) ? 4
                                            : 8;
  const int cta_n = (n % (kColsPerThreadBlock2b * 2) == 0) ? 2 : 1;
  const int blocks_per_K = k / block_size;
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock2b);
  dim3 blocks(n / (kColsPerThreadBlock2b * cta_n), (m + cta_m - 1) / cta_m);

#define MatMulFloat2bBatchedDispatch(bs, cm, cn)                                        \
  if (nullptr != zero_points) {                                                         \
    MatMulFloat2bKernelBatched<T, bs, true, cm, cn><<<blocks, threads, 0, stream>>>(    \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K); \
  } else {                                                                              \
    MatMulFloat2bKernelBatched<T, bs, false, cm, cn><<<blocks, threads, 0, stream>>>(   \
        output, a_data, b_data_quant, scales_data, nullptr, m, n, k, blocks_per_K);     \
  }
#define MatMulFloat2bBatchedDispatchN(cm, cn) \
  if (16 == block_size) {                     \
    MatMulFloat2bBatchedDispatch(16, cm, cn)  \
  } else if (32 == block_size) {              \
    MatMulFloat2bBatchedDispatch(32, cm, cn)  \
  } else if (64 == block_size) {              \
    MatMulFloat2bBatchedDispatch(64, cm, cn)  \
  } else if (128 == block_size) {             \
    MatMulFloat2bBatchedDispatch(128, cm, cn) \
  } else if (256 == block_size) {             \
    MatMulFloat2bBatchedDispatch(256, cm, cn) \
  } else {                                    \
    return false;                             \
  }
#define MatMulFloat2bBatchedDispatchM(cn)         \
  switch (cta_m) {                                \
    case 2:                                       \
      MatMulFloat2bBatchedDispatchN(2, cn) break; \
    case 4:                                       \
      MatMulFloat2bBatchedDispatchN(4, cn) break; \
    default:                                      \
      MatMulFloat2bBatchedDispatchN(8, cn) break; \
  }

  if (cta_n == 2) {
    MatMulFloat2bBatchedDispatchM(2)
  } else {
    MatMulFloat2bBatchedDispatchM(1)
  }

#undef MatMulFloat2bBatchedDispatchM
#undef MatMulFloat2bBatchedDispatchN
#undef MatMulFloat2bBatchedDispatch
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
