// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "contrib_ops/cuda/quantization/matmul_8bits_batched.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = onnxruntime::cuda::GPU_WARP_SIZE;
constexpr uint8_t kDefaultZeroPoint = 128;

template <class T>
__device__ __forceinline__ float ToFloat8b(T v);
template <>
__device__ __forceinline__ float ToFloat8b<float>(float v) { return v; }
template <>
__device__ __forceinline__ float ToFloat8b<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float ToFloat8b<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <class T>
__device__ __forceinline__ void DequantizeEight8b(uint64_t values_quant, T scale, uint8_t zp, float dq[8]) {
  float scale_f = ToFloat8b<T>(scale);
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
  const uint32_t kMagicBytes = 0x64646464u;
  const uint32_t lo32 = static_cast<uint32_t>(values_quant);
  const uint32_t hi32 = static_cast<uint32_t>(values_quant >> 32);
  const half2 bias_h2 = __half2half2(
      __ushort_as_half(static_cast<uint16_t>(0x6400u | static_cast<uint32_t>(zp))));

  uint32_t packed[4];
  packed[0] = __byte_perm(lo32, kMagicBytes, 0x4140);
  packed[1] = __byte_perm(lo32, kMagicBytes, 0x4342);
  packed[2] = __byte_perm(hi32, kMagicBytes, 0x4140);
  packed[3] = __byte_perm(hi32, kMagicBytes, 0x4342);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 d = __half22float2(__hsub2(bit_cast<half2>(packed[i]), bias_h2));
    dq[2 * i] = d.x * scale_f;
    dq[2 * i + 1] = d.y * scale_f;
  }
#else
  float zp_f = static_cast<float>(zp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t q = (values_quant >> (i * 8)) & 0xFF;
    dq[i] = (static_cast<float>(q) - zp_f) * scale_f;
  }
#endif
}

template <class T>
__device__ __forceinline__ void LoadEightActivations8b(const T* a, float out[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[j] = ToFloat8b<T>(a[j]);
  }
}

template <class T, int block_size, bool has_zero_point, int CtaM, int CtaN>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock, 3) MatMulFloat8bKernelBatched(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int col_base = (blockIdx.x * kColsPerThreadBlock + warp_id) * CtaN;
  const int m_base = blockIdx.y * CtaM;
  const int valid = m - m_base;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
  const int lane_offset = lane_id * kElementsPerThreadPerIteration;

  const T* a_base = a_data + static_cast<size_t>(m_base) * k + lane_offset;
  const uint8_t* b_ptr[CtaN];
#pragma unroll
  for (int c = 0; c < CtaN; ++c) {
    b_ptr[c] = b_data_quant + static_cast<size_t>(col_base + c) * blocks_per_K * block_size + lane_offset;
  }

  float acc[CtaM][CtaN];
#pragma unroll
  for (int r = 0; r < CtaM; ++r) {
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      acc[r][c] = 0.0f;
    }
  }

  int k_id = 0;
  int t_meta_k = lane_offset / block_size;
  constexpr int kWork = CtaM * CtaN;
  constexpr int kMainUnroll = (kWork >= 20) ? 1 : (kWork >= 12) ? 2
                                                                : 4;

#define BATCHED8_BODY(i)                                                                              \
  do {                                                                                                \
    float dq[CtaN][8];                                                                                \
    const int bk = t_meta_k + k_per_iter / block_size * (i);                                          \
    _Pragma("unroll") for (int c = 0; c < CtaN; ++c) {                                                \
      uint64_t value = *reinterpret_cast<const uint64_t*>(b_ptr[c] + k_per_iter * (i));               \
      T scale = scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + bk];                   \
      uint8_t zp = kDefaultZeroPoint;                                                                 \
      if constexpr (has_zero_point) {                                                                 \
        zp = zero_points[static_cast<size_t>(col_base + c) * blocks_per_K + bk];                      \
      }                                                                                               \
      DequantizeEight8b<T>(value, scale, zp, dq[c]);                                                  \
    }                                                                                                 \
    _Pragma("unroll") for (int r = 0; r < CtaM; ++r) {                                                \
      if (r < valid) {                                                                                \
        float av[8];                                                                                  \
        LoadEightActivations8b<T>(a_base + static_cast<size_t>(r) * k + k_id + (i) * k_per_iter, av); \
        _Pragma("unroll") for (int c = 0; c < CtaN; ++c) {                                            \
          float s = 0.0f;                                                                             \
          _Pragma("unroll") for (int j = 0; j < 8; ++j) {                                             \
            s = fmaf(av[j], dq[c][j], s);                                                             \
          }                                                                                           \
          acc[r][c] += s;                                                                             \
        }                                                                                             \
      }                                                                                               \
    }                                                                                                 \
  } while (false)

#define BATCHED8_UNROLL(unroll_size)                        \
  do {                                                      \
    constexpr int kUnroll = unroll_size;                    \
    constexpr int kUnrollStep = kUnroll * k_per_iter;       \
    const int k_unroll_bound = k - k % kUnrollStep;         \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {    \
      _Pragma("unroll") for (int i = 0; i < kUnroll; ++i) { \
        BATCHED8_BODY(i);                                   \
      }                                                     \
      _Pragma("unroll") for (int c = 0; c < CtaN; ++c) {    \
        b_ptr[c] += kUnrollStep;                            \
      }                                                     \
      t_meta_k += k_per_iter / block_size * kUnroll;        \
    }                                                       \
  } while (false)

  BATCHED8_UNROLL(kMainUnroll);
  BATCHED8_UNROLL(1);
#undef BATCHED8_UNROLL

  if (lane_offset + k_id < k) {
    float dq[CtaN][8];
    const int bk = t_meta_k;
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      uint64_t value = *reinterpret_cast<const uint64_t*>(b_ptr[c]);
      T scale = scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + bk];
      uint8_t zp = kDefaultZeroPoint;
      if constexpr (has_zero_point) {
        zp = zero_points[static_cast<size_t>(col_base + c) * blocks_per_K + bk];
      }
      DequantizeEight8b<T>(value, scale, zp, dq[c]);
    }
#pragma unroll
    for (int r = 0; r < CtaM; ++r) {
      if (r < valid) {
        float av[8];
        LoadEightActivations8b<T>(a_base + static_cast<size_t>(r) * k + k_id, av);
#pragma unroll
        for (int c = 0; c < CtaN; ++c) {
          float s = 0.0f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            s = fmaf(av[j], dq[c][j], s);
          }
          acc[r][c] += s;
        }
      }
    }
  }
#undef BATCHED8_BODY

#pragma unroll
  for (int r = 0; r < CtaM; ++r) {
    if (r >= valid) continue;
#pragma unroll
    for (int c = 0; c < CtaN; ++c) {
      float sum = acc[r][c];
      for (int off = kWarpSize / 2; off > 0; off /= 2) {
        sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, off);
      }
      if (lane_id == 0) {
        output[static_cast<size_t>(m_base + r) * n + (col_base + c)] = static_cast<T>(sum);
      }
    }
  }
}

}  // namespace

template <class T>
bool TryMatMul8BitsBatched(
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
  const int cta_m = (m <= 2) ? 2 : (m <= 4) ? 4
                                            : 8;
  const int cta_n = (n % (kColsPerThreadBlock * 2) == 0) ? 2 : 1;
  const int blocks_per_K = k / block_size;
  dim3 threads(kWarpSize, kColsPerThreadBlock);
  dim3 blocks(n / (kColsPerThreadBlock * cta_n), (m + cta_m - 1) / cta_m);

#define MatMulFloat8bBatchedDispatch(bs, cm, cn)                                        \
  if (nullptr != zero_points) {                                                         \
    MatMulFloat8bKernelBatched<T, bs, true, cm, cn><<<blocks, threads, 0, stream>>>(    \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K); \
  } else {                                                                              \
    MatMulFloat8bKernelBatched<T, bs, false, cm, cn><<<blocks, threads, 0, stream>>>(   \
        output, a_data, b_data_quant, scales_data, nullptr, m, n, k, blocks_per_K);     \
  }
#define MatMulFloat8bBatchedDispatchN(cm, cn) \
  if (16 == block_size) {                     \
    MatMulFloat8bBatchedDispatch(16, cm, cn)  \
  } else if (32 == block_size) {              \
    MatMulFloat8bBatchedDispatch(32, cm, cn)  \
  } else if (64 == block_size) {              \
    MatMulFloat8bBatchedDispatch(64, cm, cn)  \
  } else if (128 == block_size) {             \
    MatMulFloat8bBatchedDispatch(128, cm, cn) \
  } else if (256 == block_size) {             \
    MatMulFloat8bBatchedDispatch(256, cm, cn) \
  } else {                                    \
    return false;                             \
  }
#define MatMulFloat8bBatchedDispatchM(cn)         \
  switch (cta_m) {                                \
    case 2:                                       \
      MatMulFloat8bBatchedDispatchN(2, cn) break; \
    case 4:                                       \
      MatMulFloat8bBatchedDispatchN(4, cn) break; \
    default:                                      \
      MatMulFloat8bBatchedDispatchN(8, cn) break; \
  }
  if (cta_n == 2) {
    MatMulFloat8bBatchedDispatchM(2)
  } else {
    MatMulFloat8bBatchedDispatchM(1)
  }
#undef MatMulFloat8bBatchedDispatchM
#undef MatMulFloat8bBatchedDispatchN
#undef MatMulFloat8bBatchedDispatch

  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime