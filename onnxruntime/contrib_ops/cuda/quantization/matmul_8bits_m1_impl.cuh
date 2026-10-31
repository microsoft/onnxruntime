// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

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

// ---------------------------------------------------------------------------------------------
// Wide fp16 M=1 kernel.
//
// MatMulFloat8bKernelM1 above reads 8 bytes of B per lane per loop iteration and stages the scales
// in shared memory behind a __syncthreads(), so a warp keeps little memory in flight and pays one
// extra global round trip before its first weight load. For decode-sized GEMVs (a few MB of
// weights) that caps it far below DRAM bandwidth: on A100 it reads an 8.4 MB (N=2048, K=4096)
// weight at ~0.6 TB/s.
//
// This kernel gives each warp kCols output columns, has every lane load 16 bytes of B per step and
// issues kUnroll steps of loads before consuming any of them. Scales and zero points are read
// straight from global memory (neighbouring lanes share their cache lines), and A is read with
// 16-byte loads that hit L1. One warp per block keeps the grid fine-grained.
// ---------------------------------------------------------------------------------------------
constexpr int kWide8bLaneBytes = 16;                              // bytes of B per lane per step
constexpr int kWide8bStep = kM1WarpSize8Bits * kWide8bLaneBytes;  // elements of K per warp step

// sum over 16 elements of (q - zp) * a, fp32 accumulation. (q - zp) is exact in fp16.
__device__ __forceinline__ float Dot16Elements8b(uint4 q, uint8_t zp, const uint4& a0, const uint4& a1) {
  const uint32_t magic = 0x64646464u;  // fp16 1024.0 + byte, see AccumulateEightElements8b
  const half2 zp_h2 = __half2half2(__ushort_as_half(static_cast<uint16_t>(0x6400u | zp)));
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
  const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 b_lo = __half22float2(__hsub2(bit_cast<half2>(__byte_perm(w[i], magic, 0x4140)), zp_h2));
    const float2 b_hi = __half22float2(__hsub2(bit_cast<half2>(__byte_perm(w[i], magic, 0x4342)), zp_h2));
    const float2 a_lo = __half22float2(bit_cast<half2>(a[2 * i]));
    const float2 a_hi = __half22float2(bit_cast<half2>(a[2 * i + 1]));
    acc = fmaf(a_lo.x, b_lo.x, acc);
    acc = fmaf(a_lo.y, b_lo.y, acc);
    acc = fmaf(a_hi.x, b_hi.x, acc);
    acc = fmaf(a_hi.y, b_hi.y, acc);
  }
  return acc;
}

template <int block_size, bool has_zero_point, int kCols, int kUnroll>
__global__ void __launch_bounds__(kM1WarpSize8Bits) MatMulFloat8bKernelM1Wide(
    half* output, const half* a_data, const uint8_t* b_data_quant, const half* scales_data,
    const uint8_t* zero_points, int k, int blocks_per_K) {
  static_assert(block_size % kWide8bLaneBytes == 0, "a 16-byte chunk must not straddle quantization blocks");
  const int lane = threadIdx.x;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * kCols;
  float acc[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) acc[c] = 0.f;

  for (int k0 = 0; k0 < k; k0 += kWide8bStep * kUnroll) {
    uint4 q[kUnroll][kCols];
    half scale[kUnroll][kCols];
    uint8_t zp[kUnroll][kCols];
    uint4 a0[kUnroll], a1[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int kk = k0 + u * kWide8bStep + lane * kWide8bLaneBytes;
      const int block = kk / block_size;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int64_t col = col0 + c;
        q[u][c] = __ldg(reinterpret_cast<const uint4*>(b_data_quant + col * k + kk));
        scale[u][c] = scales_data[col * blocks_per_K + block];
        zp[u][c] = has_zero_point ? zero_points[col * blocks_per_K + block] : kM1DefaultZeroPoint8Bits;
      }
      a0[u] = __ldg(reinterpret_cast<const uint4*>(a_data + kk));
      a1[u] = __ldg(reinterpret_cast<const uint4*>(a_data + kk + 8));
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        acc[c] = fmaf(Dot16Elements8b(q[u][c], zp[u][c], a0[u], a1[u]), __half2float(scale[u][c]), acc[c]);
      }
    }
  }

#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    float v = acc[c];
#pragma unroll
    for (int offset = kM1WarpSize8Bits / 2; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    if (lane == 0) output[col0 + c] = __float2half(v);
  }
}

template <int kCols, int kUnroll>
void LaunchMatMulFloat8bKernelM1Wide(half* output, const half* a_data, const uint8_t* b_data_quant,
                                     const half* scales_data, const uint8_t* zero_points, int n, int k,
                                     int block_size, cudaStream_t stream) {
  const int blocks_per_K = k / block_size;
  const dim3 blocks(n / kCols);
#define MATMUL_FLOAT8B_M1_WIDE_DISPATCH(bs)                                                        \
  if (zero_points != nullptr) {                                                                    \
    MatMulFloat8bKernelM1Wide<bs, true, kCols, kUnroll><<<blocks, kM1WarpSize8Bits, 0, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, k, blocks_per_K);                  \
  } else {                                                                                         \
    MatMulFloat8bKernelM1Wide<bs, false, kCols, kUnroll><<<blocks, kM1WarpSize8Bits, 0, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, k, blocks_per_K);                      \
  }
  switch (block_size) {
    case 16:
      MATMUL_FLOAT8B_M1_WIDE_DISPATCH(16)
      break;
    case 32:
      MATMUL_FLOAT8B_M1_WIDE_DISPATCH(32)
      break;
    case 64:
      MATMUL_FLOAT8B_M1_WIDE_DISPATCH(64)
      break;
    case 128:
      MATMUL_FLOAT8B_M1_WIDE_DISPATCH(128)
      break;
    case 256:
      MATMUL_FLOAT8B_M1_WIDE_DISPATCH(256)
      break;
  }
#undef MATMUL_FLOAT8B_M1_WIDE_DISPATCH
}

// Returns false (and launches nothing) when the shape is not covered; the caller then falls back.
inline bool TryMatMul8BitsM1Wide(half* output, const half* a_data, const uint8_t* b_data_quant,
                                 const half* scales_data, const uint8_t* zero_points, int n, int k, int block_size,
                                 cudaStream_t stream) {
  if (block_size != 16 && block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256) {
    return false;
  }
  if (k % kWide8bStep != 0 || k % block_size != 0) return false;
  // A and B are read with 16-byte vector loads. B is a prepacked initializer, but A can be a
  // user-bound buffer at an arbitrary offset, so an unaligned pointer falls back to the original kernel.
  if (reinterpret_cast<uintptr_t>(a_data) % kWide8bLaneBytes != 0 ||
      reinterpret_cast<uintptr_t>(b_data_quant) % kWide8bLaneBytes != 0) {
    return false;
  }
  // Long K: one column per warp already keeps enough bytes in flight. Short K: two columns per warp
  // share each A load. Chosen from A100 measurements at N=2048/K=4096 and N=5120/K=2048.
  if (k >= 4096) {
    if (k % (kWide8bStep * 4) == 0) {
      LaunchMatMulFloat8bKernelM1Wide<1, 4>(output, a_data, b_data_quant, scales_data, zero_points, n, k,
                                            block_size, stream);
    } else {
      LaunchMatMulFloat8bKernelM1Wide<1, 1>(output, a_data, b_data_quant, scales_data, zero_points, n, k,
                                            block_size, stream);
    }
    return true;
  }
  // Defensive only: TryMatMul8Bits already requires N % 8 == 0.
  if (n % 2 != 0) return false;
  if (k % (kWide8bStep * 4) == 0) {
    LaunchMatMulFloat8bKernelM1Wide<2, 4>(output, a_data, b_data_quant, scales_data, zero_points, n, k, block_size,
                                          stream);
  } else if (k % (kWide8bStep * 2) == 0) {
    LaunchMatMulFloat8bKernelM1Wide<2, 2>(output, a_data, b_data_quant, scales_data, zero_points, n, k, block_size,
                                          stream);
  } else {
    LaunchMatMulFloat8bKernelM1Wide<2, 1>(output, a_data, b_data_quant, scales_data, zero_points, n, k, block_size,
                                          stream);
  }
  return true;
}

template <class T>
bool TryMatMul8BitsM1(T* output, const T* a_data, const uint8_t* b_data_quant, const T* scales_data,
                      const uint8_t* zero_points, int n, int k, int block_size, size_t shared_mem_per_block,
                      cudaStream_t stream) {
  if constexpr (std::is_same_v<T, half>) {
    // ORT_MATMUL_NBITS_8BIT_M1_WIDE=0 keeps the original kernel (A/B comparison). std::getenv rather
    // than the platform env-var helper, whose header clashes with provider_api.h macros here.
    static const bool use_wide = [] {
      const char* v = std::getenv("ORT_MATMUL_NBITS_8BIT_M1_WIDE");
      return v == nullptr || std::atoi(v) != 0;
    }();
    if (use_wide &&
        TryMatMul8BitsM1Wide(output, a_data, b_data_quant, scales_data, zero_points, n, k, block_size, stream)) {
      return true;
    }
  }
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
