// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_batched.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kSmallMMax = 16;

template <class T>
struct DequantizedEight;

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
template <>
struct DequantizedEight<half> {
  half2 v[4];
};

__device__ __forceinline__ void DequantizeEight(
    uint32_t values_quant, half scale, uint8_t zp, DequantizedEight<half>& output) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  half2 elements[4];
  Convert8xInt4To8xHalfs(values_quant, elements);
  output.v[0] = elements[0] * scale_half2 + zp_adjust2;
  output.v[1] = elements[1] * scale_half2 + zp_adjust2;
  output.v[2] = elements[2] * scale_half2 + zp_adjust2;
  output.v[3] = elements[3] * scale_half2 + zp_adjust2;
}
#else
template <>
struct DequantizedEight<half> {
  half v[8];
};

__device__ __forceinline__ void DequantizeEight(
    uint32_t values_quant, half scale, uint8_t zp, DequantizedEight<half>& output) {
  half zp_adjust = -scale * __short2half_rn(zp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    output.v[i] = __uint2half_rn((values_quant >> (4 * i)) & 0xF) * scale + zp_adjust;
  }
}
#endif

template <>
struct DequantizedEight<nv_bfloat16> {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 v[4];
#else
  nv_bfloat16 v[8];
#endif
};

__device__ __forceinline__ void DequantizeEight(
    uint32_t values_quant, nv_bfloat16 scale, uint8_t zp, DequantizedEight<nv_bfloat16>& output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale_bf162 = __bfloat162bfloat162(scale);
  nv_bfloat16 zp_adjust = -scale * __uint2bfloat16_rn(zp);
  __nv_bfloat162 zp_adjust2 = __bfloat162bfloat162(zp_adjust);
  __nv_bfloat162 elements[4];
  Convert8xInt4To8xBF16s(values_quant, elements);
  output.v[0] = __hfma2(elements[0], scale_bf162, zp_adjust2);
  output.v[1] = __hfma2(elements[1], scale_bf162, zp_adjust2);
  output.v[2] = __hfma2(elements[2], scale_bf162, zp_adjust2);
  output.v[3] = __hfma2(elements[3], scale_bf162, zp_adjust2);
#endif
}

template <class T>
struct Acc2;
template <>
struct Acc2<half> {
  using type = half2;
};
template <>
struct Acc2<nv_bfloat16> {
  using type = __nv_bfloat162;
};

template <class T>
struct WPack {
  typename Acc2<T>::type v[4];
};

__device__ __forceinline__ WPack<half> PackNatural(const DequantizedEight<half>& dequantized) {
  WPack<half> packed;
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  uint32_t d0 = *reinterpret_cast<const uint32_t*>(&dequantized.v[0]);
  uint32_t d1 = *reinterpret_cast<const uint32_t*>(&dequantized.v[1]);
  uint32_t d2 = *reinterpret_cast<const uint32_t*>(&dequantized.v[2]);
  uint32_t d3 = *reinterpret_cast<const uint32_t*>(&dequantized.v[3]);
  constexpr uint32_t kLo = 0x5410;
  constexpr uint32_t kHi = 0x7632;
  uint32_t value;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d0), "r"(d1), "r"(kLo));
  packed.v[0] = *reinterpret_cast<half2*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d2), "r"(d3), "r"(kLo));
  packed.v[1] = *reinterpret_cast<half2*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d0), "r"(d1), "r"(kHi));
  packed.v[2] = *reinterpret_cast<half2*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d2), "r"(d3), "r"(kHi));
  packed.v[3] = *reinterpret_cast<half2*>(&value);
#endif
  return packed;
}

__device__ __forceinline__ void DotAccum(const WPack<half>& weights, const half2* activations, half2& acc) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  acc = __hfma2(weights.v[0], activations[0], acc);
  acc = __hfma2(weights.v[1], activations[1], acc);
  acc = __hfma2(weights.v[2], activations[2], acc);
  acc = __hfma2(weights.v[3], activations[3], acc);
#endif
}

__device__ __forceinline__ float HorizontalAdd(half2 acc) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  return static_cast<float>(acc.x) + static_cast<float>(acc.y);
#else
  return 0.0f;
#endif
}

__device__ __forceinline__ WPack<nv_bfloat16> PackNatural(
    const DequantizedEight<nv_bfloat16>& dequantized) {
  WPack<nv_bfloat16> packed;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t d0 = *reinterpret_cast<const uint32_t*>(&dequantized.v[0]);
  uint32_t d1 = *reinterpret_cast<const uint32_t*>(&dequantized.v[1]);
  uint32_t d2 = *reinterpret_cast<const uint32_t*>(&dequantized.v[2]);
  uint32_t d3 = *reinterpret_cast<const uint32_t*>(&dequantized.v[3]);
  constexpr uint32_t kLo = 0x5410;
  constexpr uint32_t kHi = 0x7632;
  uint32_t value;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d0), "r"(d1), "r"(kLo));
  packed.v[0] = *reinterpret_cast<__nv_bfloat162*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d2), "r"(d3), "r"(kLo));
  packed.v[1] = *reinterpret_cast<__nv_bfloat162*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d0), "r"(d1), "r"(kHi));
  packed.v[2] = *reinterpret_cast<__nv_bfloat162*>(&value);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(value) : "r"(d2), "r"(d3), "r"(kHi));
  packed.v[3] = *reinterpret_cast<__nv_bfloat162*>(&value);
#endif
  return packed;
}

__device__ __forceinline__ void DotAccum(
    const WPack<nv_bfloat16>& weights, const __nv_bfloat162* activations, __nv_bfloat162& acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  acc = __hfma2(weights.v[0], activations[0], acc);
  acc = __hfma2(weights.v[1], activations[1], acc);
  acc = __hfma2(weights.v[2], activations[2], acc);
  acc = __hfma2(weights.v[3], activations[3], acc);
#endif
}

__device__ __forceinline__ float HorizontalAdd(__nv_bfloat162 acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return static_cast<float>(acc.x) + static_cast<float>(acc.y);
#else
  return 0.0f;
#endif
}

template <class T, int block_size, bool has_zero_point, int CtaM, int CtaN>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock, 3) MatMulFloat4BatchedKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  using AccT = typename Acc2<T>::type;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int col_base = (blockIdx.x * kColsPerThreadBlock + warp_id) * CtaN;
  const int m_base = blockIdx.y * CtaM;
  const int valid = m - m_base;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
  const int zp_blocks = (blocks_per_K + 1) / 2;

  const T* a_base = a_data + static_cast<size_t>(m_base) * k + (lane_id << 3);
  const uint8_t* b_ptr[CtaN];
#pragma unroll
  for (int c = 0; c < CtaN; ++c) {
    b_ptr[c] = b_data_quant + static_cast<size_t>(col_base + c) * blocks_per_K * (block_size / 2) + lane_id * 4;
  }

  AccT acc[CtaM][CtaN];
#pragma unroll
  for (int row = 0; row < CtaM; ++row) {
#pragma unroll
    for (int col = 0; col < CtaN; ++col) {
      acc[row][col] = AccT{};
    }
  }

  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  constexpr int kWork = CtaM * CtaN;
  constexpr int kMainUnroll = (kWork >= 20) ? 1 : (kWork >= 12) ? 2
                                                                : 4;

#define BATCHED_BODY(i)                                                                                    \
  do {                                                                                                     \
    WPack<T> weights[CtaN];                                                                                \
    const int block_k = t_meta_k + k_per_iter / block_size * (i);                                          \
    _Pragma("unroll") for (int col = 0; col < CtaN; ++col) {                                               \
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_ptr[col] + k_per_iter / 2 * (i)));            \
      T scale = scales_data[static_cast<size_t>(col_base + col) * blocks_per_K + block_k];                 \
      uint8_t zero_point = 8;                                                                              \
      if constexpr (has_zero_point) {                                                                      \
        uint8_t packed_zp = zero_points[static_cast<size_t>(col_base + col) * zp_blocks + (block_k >> 1)]; \
        zero_point = (block_k & 1) ? (packed_zp >> 4) : (packed_zp & 0x0f);                                \
      }                                                                                                    \
      DequantizedEight<T> dequantized;                                                                     \
      DequantizeEight(value, scale, zero_point, dequantized);                                              \
      weights[col] = PackNatural(dequantized);                                                             \
    }                                                                                                      \
    _Pragma("unroll") for (int row = 0; row < CtaM; ++row) {                                               \
      if (row < valid) {                                                                                   \
        AccT activations[4];                                                                               \
        *reinterpret_cast<uint4*>(activations) = *reinterpret_cast<const uint4*>(                          \
            a_base + static_cast<size_t>(row) * k + k_id + (i) * k_per_iter);                              \
        _Pragma("unroll") for (int col = 0; col < CtaN; ++col) {                                           \
          DotAccum(weights[col], activations, acc[row][col]);                                              \
        }                                                                                                  \
      }                                                                                                    \
    }                                                                                                      \
  } while (false)

#define BATCHED_UNROLL(unroll_size)                            \
  do {                                                         \
    constexpr int kUnroll = unroll_size;                       \
    constexpr int kUnrollStep = kUnroll * k_per_iter;          \
    const int k_unroll_bound = k - k % kUnrollStep;            \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {       \
      _Pragma("unroll") for (int i = 0; i < kUnroll; ++i) {    \
        BATCHED_BODY(i);                                       \
      }                                                        \
      _Pragma("unroll") for (int col = 0; col < CtaN; ++col) { \
        b_ptr[col] += k_per_iter / 2 * kUnroll;                \
      }                                                        \
      t_meta_k += k_per_iter / block_size * kUnroll;           \
    }                                                          \
  } while (false)

  BATCHED_UNROLL(kMainUnroll);
  BATCHED_UNROLL(1);
#undef BATCHED_UNROLL

  if (k_id + lane_id * 8 < k) {
    WPack<T> weights[CtaN];
    const int block_k = t_meta_k;
#pragma unroll
    for (int col = 0; col < CtaN; ++col) {
      uint32_t value = *reinterpret_cast<const uint32_t*>(b_ptr[col]);
      T scale = scales_data[static_cast<size_t>(col_base + col) * blocks_per_K + block_k];
      uint8_t zero_point = 8;
      if constexpr (has_zero_point) {
        uint8_t packed_zp = zero_points[static_cast<size_t>(col_base + col) * zp_blocks + (block_k >> 1)];
        zero_point = (block_k & 1) ? (packed_zp >> 4) : (packed_zp & 0x0f);
      }
      DequantizedEight<T> dequantized;
      DequantizeEight(value, scale, zero_point, dequantized);
      weights[col] = PackNatural(dequantized);
    }
#pragma unroll
    for (int row = 0; row < CtaM; ++row) {
      if (row < valid) {
        AccT activations[4];
        *reinterpret_cast<uint4*>(activations) =
            *reinterpret_cast<const uint4*>(a_base + static_cast<size_t>(row) * k + k_id);
#pragma unroll
        for (int col = 0; col < CtaN; ++col) {
          DotAccum(weights[col], activations, acc[row][col]);
        }
      }
    }
  }
#undef BATCHED_BODY

#pragma unroll
  for (int row = 0; row < CtaM; ++row) {
    if (row >= valid) continue;
#pragma unroll
    for (int col = 0; col < CtaN; ++col) {
      float sum = HorizontalAdd(acc[row][col]);
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, offset);
      }
      if (lane_id == 0) {
        output[static_cast<size_t>(m_base + row) * n + col_base + col] = static_cast<T>(sum);
      }
    }
  }
}

}  // namespace

template <class T>
bool TryMatMulBatched4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t /*shared_mem_size*/,
    cudaStream_t stream) {
  if (m < 2 || m > kSmallMMax) {
    return false;
  }

  const int cta_m = (m <= 2) ? 2 : (m <= 4) ? 4
                               : (m <= 8)   ? 8
                                            : 16;
  const int cta_n = (n % (kColsPerThreadBlock * 2) == 0) ? 2 : 1;
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  dim3 blocks(n / (kColsPerThreadBlock * cta_n), (m + cta_m - 1) / cta_m);

#define BatchedDispatch(BS, CM, CN)                                                          \
  if (nullptr != zero_points) {                                                              \
    MatMulFloat4BatchedKernel<T, BS, true, CM, CN><<<blocks, threads, 0, stream>>>(          \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS); \
  } else {                                                                                   \
    MatMulFloat4BatchedKernel<T, BS, false, CM, CN><<<blocks, threads, 0, stream>>>(         \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS); \
  }
#define BatchedDispatchN(CM, CN)  \
  if (16 == block_size) {         \
    BatchedDispatch(16, CM, CN)   \
  } else if (32 == block_size) {  \
    BatchedDispatch(32, CM, CN)   \
  } else if (64 == block_size) {  \
    BatchedDispatch(64, CM, CN)   \
  } else if (128 == block_size) { \
    BatchedDispatch(128, CM, CN)  \
  } else {                        \
    return false;                 \
  }
#define BatchedDispatchM(CN)          \
  switch (cta_m) {                    \
    case 2:                           \
      BatchedDispatchN(2, CN) break;  \
    case 4:                           \
      BatchedDispatchN(4, CN) break;  \
    case 8:                           \
      BatchedDispatchN(8, CN) break;  \
    default:                          \
      BatchedDispatchN(16, CN) break; \
  }

  if (cta_n == 2) {
    BatchedDispatchM(2)
  } else {
    BatchedDispatchM(1)
  }

#undef BatchedDispatchM
#undef BatchedDispatchN
#undef BatchedDispatch
  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime