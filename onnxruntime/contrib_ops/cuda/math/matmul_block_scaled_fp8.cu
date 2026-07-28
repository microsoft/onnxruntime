// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

// cuda_fp8.h only ships with CUDA 11.8+. Guard the include so older toolkits (or
// DISABLE_FLOAT8_TYPES builds) still compile. CUDA_VERSION is provided by <cuda.h>,
// which is pulled in by common.cuh above.
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

namespace {

template <typename T>
__device__ __forceinline__ T FromFloat(float v);

template <>
__device__ __forceinline__ half FromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float ToFloat(T v);

template <>
__device__ __forceinline__ float ToFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Dequantizes FP8 E4M3 weights with per-block FP32 scales into FP16/BF16.
// b_fp8 is [N, K] row-major FP8 E4M3, weight_scale is [N, k_blocks] fp32. Scalar per-element
// fallback used when K is not a multiple of 16 (the vectorized kernel below handles K % 16 == 0).
template <typename T>
__global__ void DequantizeBlockScaledFp8Kernel(T* __restrict__ out,
                                               const __nv_fp8_e4m3* __restrict__ b_fp8,
                                               const float* __restrict__ weight_scale,
                                               int n,
                                               int k,
                                               int k_blocks,
                                               int block_size) {
  const long long total = static_cast<long long>(n) * k;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / k);
  const int col = static_cast<int>(idx - static_cast<long long>(row) * k);
  const int blk = col / block_size;
  const float scale = weight_scale[row * k_blocks + blk];
  out[idx] = FromFloat<T>(static_cast<float>(b_fp8[idx]) * scale);
}

// Vectorized dequantization for K % 16 == 0. Each thread converts one 16-element K chunk of a
// single row: a coalesced 16-byte FP8 load and a coalesced 32-byte (2 x uint4) FP16/BF16 store.
// The row index comes from a 2D grid (blockIdx.y), avoiding the expensive 64-bit idx / k division
// of the scalar kernel. When block_size is a multiple of 16 the whole chunk shares one scale, so a
// single scale value is loaded per chunk instead of per element.
template <typename T>
__global__ void DequantizeBlockScaledFp8Vec16Kernel(T* __restrict__ out,
                                                    const __nv_fp8_e4m3* __restrict__ b_fp8,
                                                    const float* __restrict__ weight_scale,
                                                    int n,
                                                    int k,
                                                    int k_blocks,
                                                    int block_size,
                                                    int kv,
                                                    bool block_aligned16) {
  const int g = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;  // 16-element chunk in a row
  if (g >= kv) {
    return;
  }
  const int col0 = g * 16;
  for (int row = blockIdx.y; row < n; row += gridDim.y) {
    const size_t base = static_cast<size_t>(row) * k + col0;
    const uint4 raw = *reinterpret_cast<const uint4*>(b_fp8 + base);
    const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
    const float* srow = weight_scale + static_cast<size_t>(row) * k_blocks;

    T outv[16];
    if (block_aligned16) {
      const float scale = srow[col0 / block_size];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        outv[i] = FromFloat<T>(static_cast<float>(bp[i]) * scale);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        outv[i] = FromFloat<T>(static_cast<float>(bp[i]) * srow[(col0 + i) / block_size]);
      }
    }

    uint4* op = reinterpret_cast<uint4*>(out + base);
    op[0] = *reinterpret_cast<const uint4*>(&outv[0]);
    op[1] = *reinterpret_cast<const uint4*>(&outv[8]);
  }
}

template <typename T>
__global__ void AddBiasKernel(T* __restrict__ y, const T* __restrict__ bias, int m, int n) {
  const long long total = static_cast<long long>(m) * n;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % n);
  y[idx] = FromFloat<T>(ToFloat<T>(y[idx]) + ToFloat<T>(bias[col]));
}

// Statically quantizes a FP16/BF16 activation to FP8 E4M3 using a single per-tensor scale and then
// dequantizes it back to the activation type. This realizes W8A8 activation numerics: the FP8
// rounding error is intentionally introduced so the result matches native W8A8 execution, while the
// downstream GEMM stays in the activation type (architecture independent). a_scale is the dequant
// scale, so the quantized value is fp8_e4m3(a / a_scale) and the emitted activation is
// fp8_e4m3(a / a_scale) * a_scale.
template <typename T>
__global__ void QuantizeDequantizeActivationFp8Kernel(T* __restrict__ out,
                                                      const T* __restrict__ in,
                                                      const float* __restrict__ a_scale,
                                                      long long total) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float scale = *a_scale;
  const float inv_scale = scale != 0.f ? 1.0f / scale : 0.f;
  const float x = ToFloat<T>(in[idx]);
  const float q = static_cast<float>(__nv_fp8_e4m3(x * inv_scale));
  out[idx] = FromFloat<T>(q * scale);
}

// -----------------------------------------------------------------------------
// Fused FP8 weight-only GEMV fast path for the decode phase (small M).
//
// Each warp computes RowsPerWarp x ColsPerWarp output elements Y[row, col]. The 32
// lanes of the warp cooperatively reduce over K using 16-wide vectorized loads, so
// the FP8 weight B is streamed exactly once with fully coalesced warp transactions.
// Block scales are applied once per K-block (not per element): a lane's 16-element
// chunk is guaranteed to lie inside a single K-block whenever block_size is a
// multiple of 16, so weight_scale is folded in per chunk. Runs on SM80+.
//
// ColsPerWarp / Unroll exist because the decode GEMV is bound by the number of
// *outstanding* loads, not by bandwidth or by instruction count. Nsight Compute on
// an 8192x2048 decode GEMV (H200) reports only 1.0 full waves, L1 hit rate 79.6%
// (the A row is already resident, so staging it in shared memory buys nothing), and
// a third of the warp stall cycles waiting on L1TEX. With one 16-element B chunk in
// flight per thread, each thread only moves K/32 bytes of B and eats the full L1
// latency every iteration.
//
//   * Unroll      issues `Unroll` independent B/A loads before consuming any of them.
//   * ColsPerWarp lets one A load feed `ColsPerWarp` independent FMA chains, which
//                 both raises arithmetic intensity and adds independent B streams.
//
// Both trade occupancy (already register-capped at 8 blocks/SM) for per-thread
// memory-level parallelism, which is the right trade when there is only one wave.
// <RowsPerWarp, 1, 1> reproduces the original one-column-per-warp geometry.

// Vector-2 companion type and float2 widening for the accumulate type.
template <typename T>
struct Fp8GemvVec2;
template <>
struct Fp8GemvVec2<half> {
  using type = __half2;
};
template <>
struct Fp8GemvVec2<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

__device__ __forceinline__ float2 Fp8GemvToFloat2(const __half2& v) { return __half22float2(v); }
__device__ __forceinline__ float2 Fp8GemvToFloat2(const __nv_bfloat162& v) { return __bfloat1622float2(v); }

// Convert 16 packed FP8 bytes into 8 half2 (one `cvt.rn.f16x2.e4m3x2` per pair
// instead of 16 scalar converts). FP8 E4M3 is exactly representable in FP16, so
// this is lossless for both the half and the bfloat16 instantiation.
//
// Written as a macro rather than a helper taking `__half2 (&)[8]`: an array passed
// by reference is placed in local memory, which costs ~2x here and much more at
// RowsPerWarp > 1 where the spill lands in the inner loop.
#define ORT_FP8_GEMV_CVT16(raw, bh)                                                         \
  do {                                                                                      \
    const __nv_fp8x2_storage_t* _p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&(raw)); \
    _Pragma("unroll") for (int _i = 0; _i < 8; ++_i) {                                      \
      (bh)[_i] = __half2(__nv_cvt_fp8x2_to_halfraw2(_p[_i], __NV_E4M3));                    \
    }                                                                                       \
  } while (0)

// acc += dot(A[0:16], B[0:16]) with FP32 accumulation, matching the scalar version bit for bit.
#define ORT_FP8_GEMV_DOT16(a0, a1, bh, acc)                   \
  do {                                                        \
    const AVec2* _a0 = reinterpret_cast<const AVec2*>(&(a0)); \
    const AVec2* _a1 = reinterpret_cast<const AVec2*>(&(a1)); \
    _Pragma("unroll") for (int _i = 0; _i < 4; ++_i) {        \
      const float2 _av = Fp8GemvToFloat2(_a0[_i]);            \
      const float2 _bv = __half22float2((bh)[_i]);            \
      (acc) = fmaf(_av.x, _bv.x, (acc));                      \
      (acc) = fmaf(_av.y, _bv.y, (acc));                      \
    }                                                         \
    _Pragma("unroll") for (int _i = 0; _i < 4; ++_i) {        \
      const float2 _av = Fp8GemvToFloat2(_a1[_i]);            \
      const float2 _bv = __half22float2((bh)[4 + _i]);        \
      (acc) = fmaf(_av.x, _bv.x, (acc));                      \
      (acc) = fmaf(_av.y, _bv.y, (acc));                      \
    }                                                         \
  } while (0)

template <int RowsPerWarp, int ColsPerWarp, int Unroll, typename AType>
__global__ void MatMulBlockScaledFp8GemvKernel(AType* __restrict__ output,
                                               const AType* __restrict__ input_a,
                                               const __nv_fp8_e4m3* __restrict__ input_b,
                                               const float* __restrict__ weight_scale,
                                               const AType* __restrict__ bias,
                                               int m,
                                               int n,
                                               int k,
                                               int block_size,
                                               int k_blocks) {
  using AVec2 = typename Fp8GemvVec2<AType>::type;

  constexpr int kElemsPerLane = 16;
  constexpr int kStride = 32 * kElemsPerLane;  // 512 elements per warp sub-chunk

  const int lane = threadIdx.x;  // 0..31
  const int col_base = (blockIdx.x * blockDim.y + threadIdx.y) * ColsPerWarp;
  const int row_base = blockIdx.y * RowsPerWarp;
  if (row_base >= m || col_base >= n) {
    return;
  }

  float acc[RowsPerWarp][ColsPerWarp] = {};

  for (int base = 0; base < k; base += kStride * Unroll) {
    // Phase 1: issue every load for this iteration back to back. Nothing here consumes a
    // loaded value, so all Unroll * (ColsPerWarp + 2 * RowsPerWarp) requests are in flight
    // at once instead of one at a time.
    uint4 b_raw[Unroll][ColsPerWarp];
    uint4 a_lo[Unroll][RowsPerWarp];
    uint4 a_hi[Unroll][RowsPerWarp];
    int koff[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      koff[u] = base + u * kStride + lane * kElemsPerLane;
      const bool k_ok = koff[u] < k;
#pragma unroll
      for (int c = 0; c < ColsPerWarp; ++c) {
        const int col = col_base + c;
        b_raw[u][c] = (k_ok && col < n)
                          ? *reinterpret_cast<const uint4*>(input_b + static_cast<size_t>(col) * k + koff[u])
                          : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = row_base + r;
        if (k_ok && row < m) {
          const uint4* ap = reinterpret_cast<const uint4*>(input_a + static_cast<size_t>(row) * k + koff[u]);
          a_lo[u][r] = ap[0];
          a_hi[u][r] = ap[1];
        } else {
          a_lo[u][r] = make_uint4(0, 0, 0, 0);
          a_hi[u][r] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // Phase 2: consume.
#pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      if (koff[u] >= k) {
        continue;
      }
      const int kb = koff[u] / block_size;
#pragma unroll
      for (int c = 0; c < ColsPerWarp; ++c) {
        const int col = col_base + c;
        if (col >= n) {
          continue;
        }
        __half2 b_half[8];
        ORT_FP8_GEMV_CVT16(b_raw[u][c], b_half);
        const float b_scale = weight_scale[static_cast<size_t>(col) * k_blocks + kb];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
          if (row_base + r >= m) {
            continue;
          }
          float partial = 0.0f;
          ORT_FP8_GEMV_DOT16(a_lo[u][r], a_hi[u][r], b_half, partial);
          acc[r][c] += partial * b_scale;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
    for (int c = 0; c < ColsPerWarp; ++c) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][c] += __shfl_down_sync(0xffffffffu, acc[r][c], offset);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      const int row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (int c = 0; c < ColsPerWarp; ++c) {
        const int col = col_base + c;
        if (col >= n) {
          continue;
        }
        float result = acc[r][c];
        if (bias != nullptr) {
          result += ToFloat<AType>(bias[col]);
        }
        output[static_cast<size_t>(row) * n + col] = FromFloat<AType>(result);
      }
    }
  }
}

}  // namespace

#endif  // !DISABLE_FLOAT8_TYPES && defined(CUDA_VERSION) && CUDA_VERSION >= 11080

Status LaunchDequantizeBlockScaledFp8(void* b_dequant,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(n) * k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(b_fp8);

  // Fast path: K is a multiple of 16, so each thread owns one aligned 16-element chunk and both the
  // FP8 load and the FP16/BF16 store are fully coalesced. This is the common prefill layout.
  if (k % 16 == 0) {
    const int kv = k / 16;  // 16-element chunks per row
    constexpr int kThreads = 256;
    const unsigned int grid_x = static_cast<unsigned int>((kv + kThreads - 1) / kThreads);
    const unsigned int grid_y = static_cast<unsigned int>(n < 65535 ? n : 65535);
    const dim3 blocks{grid_x, grid_y};
    const bool block_aligned16 = (block_size % 16 == 0);
    if (is_bf16) {
      DequantizeBlockScaledFp8Vec16Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, kv,
          block_aligned16);
    } else {
      DequantizeBlockScaledFp8Vec16Kernel<half><<<blocks, kThreads, 0, stream>>>(
          reinterpret_cast<half*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size, kv, block_aligned16);
    }
    return CUDA_CALL(cudaGetLastError());
  }

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    DequantizeBlockScaledFp8Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size);
  } else {
    DequantizeBlockScaledFp8Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(b_dequant), b, weight_scale, n, k, k_blocks, block_size);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(b_dequant);
  ORT_UNUSED_PARAMETER(b_fp8);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}

Status LaunchAddBiasBlockScaledFp8(void* y,
                                   const void* bias,
                                   int m,
                                   int n,
                                   bool is_bf16,
                                   cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(m) * n;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    AddBiasKernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(bias), m, n);
  } else {
    AddBiasKernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(y), reinterpret_cast<const half*>(bias), m, n);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}
Status LaunchQuantizeDequantizeActivationFp8(void* a_out,
                                             const void* a_in,
                                             const float* a_scale,
                                             int m,
                                             int k,
                                             bool is_bf16,
                                             cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(m) * k;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    QuantizeDequantizeActivationFp8Kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(a_out), reinterpret_cast<const __nv_bfloat16*>(a_in), a_scale, total);
  } else {
    QuantizeDequantizeActivationFp8Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(a_out), reinterpret_cast<const half*>(a_in), a_scale, total);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(a_out);
  ORT_UNUSED_PARAMETER(a_in);
  ORT_UNUSED_PARAMETER(a_scale);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}
Status LaunchMatMulBlockScaledFp8Gemv(void* y,
                                      const void* a,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      const void* bias,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  if (m <= 0 || n <= 0 || k <= 0) {
    return Status::OK();
  }
  // This kernel assumes K is a multiple of 16 (each lane loads a full 16-element slice) and that
  // block_size is a multiple of 16 (a lane's slice lies inside a single K-block). Guard against
  // misuse if this helper is ever reused outside the callers that already check these.
  ORT_RETURN_IF_NOT(k % 16 == 0, "MatMulBlockQuantizedFp8Weight GEMV requires K divisible by 16, got ", k, ".");
  ORT_RETURN_IF_NOT(block_size % 16 == 0,
                    "MatMulBlockQuantizedFp8Weight GEMV requires block_size divisible by 16, got ", block_size, ".");
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kWarpsPerBlock = 8;
  const dim3 threads{32, kWarpsPerBlock};
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(b_fp8);
  const auto launch = [&]<int RowsPerWarp, int ColsPerWarp, int Unroll>() {
    const int cols_per_block = kWarpsPerBlock * ColsPerWarp;
    const dim3 blocks{static_cast<unsigned int>((n + cols_per_block - 1) / cols_per_block),
                      static_cast<unsigned int>((m + RowsPerWarp - 1) / RowsPerWarp)};
    if (is_bf16) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp, ColsPerWarp, Unroll><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(a), b,
          weight_scale, reinterpret_cast<const __nv_bfloat16*>(bias), m, n, k, block_size, k_blocks);
    } else {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp, ColsPerWarp, Unroll><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<half*>(y), reinterpret_cast<const half*>(a), b,
          weight_scale, reinterpret_cast<const half*>(bias), m, n, k, block_size, k_blocks);
    }
  };

  // M == 1 is the decode-with-batch-1 case and by far the hottest. There the grid is a
  // single wave, so widening each warp (ColsPerWarp) and pre-issuing loads (Unroll) buys
  // memory-level parallelism at no real occupancy cost. It only pays once N is large
  // enough that dividing the column count still leaves the GPU full -- below N == 4096
  // the wider tiles measured slower than one column per warp on H200, and for M > 1 the
  // extra live registers per warp (RowsPerWarp * ColsPerWarp accumulators plus the
  // pre-issued loads) cost more than the extra parallelism is worth.
  if (m == 1) {
    if (n >= 8192) {
      launch.template operator()<1, 4, 2>();
    } else if (n >= 4096) {
      launch.template operator()<1, 2, 2>();
    } else {
      launch.template operator()<1, 1, 1>();
    }
  } else if (m <= 2) {
    launch.template operator()<2, 1, 1>();
  } else if (m <= 4) {
    launch.template operator()<4, 1, 1>();
  } else {
    launch.template operator()<8, 1, 1>();
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(b_fp8);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires CUDA 11.8 or later.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
