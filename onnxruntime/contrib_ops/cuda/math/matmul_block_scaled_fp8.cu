// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080

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
// b_fp8 is [N, K] row-major FP8 E4M3, weight_scale is [N, k_blocks] fp32.
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
// Each warp computes RowsPerWarp output elements Y[row, col]. The 32 lanes of the
// warp cooperatively reduce over K using 16-wide vectorized loads, so the FP8
// weight B is streamed exactly once with fully coalesced warp transactions. Block
// scales are applied once per K-block (not per element): a lane's 16-element chunk
// is guaranteed to lie inside a single K-block whenever block_size is a multiple of
// 16, so weight_scale is folded in per chunk. Runs on any architecture (SM80+).
template <typename AType>
__device__ __forceinline__ void LoadFp8Gemv16A(const AType* ptr, float (&out)[16]);

template <>
__device__ __forceinline__ void LoadFp8Gemv16A<half>(const half* ptr, float (&out)[16]) {
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
  const uint4 lo = p[0];
  const uint4 hi = p[1];
  const half* vlo = reinterpret_cast<const half*>(&lo);
  const half* vhi = reinterpret_cast<const half*>(&hi);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __half2float(vlo[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[8 + i] = __half2float(vhi[i]);
  }
}

template <>
__device__ __forceinline__ void LoadFp8Gemv16A<__nv_bfloat16>(const __nv_bfloat16* ptr, float (&out)[16]) {
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
  const uint4 lo = p[0];
  const uint4 hi = p[1];
  const __nv_bfloat16* vlo = reinterpret_cast<const __nv_bfloat16*>(&lo);
  const __nv_bfloat16* vhi = reinterpret_cast<const __nv_bfloat16*>(&hi);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __bfloat162float(vlo[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[8 + i] = __bfloat162float(vhi[i]);
  }
}

template <int RowsPerWarp, typename AType>
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
  const int lane = threadIdx.x;                           // 0..31
  const int col = blockIdx.x * blockDim.y + threadIdx.y;  // n
  const int row_base = blockIdx.y * RowsPerWarp;          // m
  if (row_base >= m || col >= n) {
    return;
  }

  const __nv_fp8_e4m3* b_row = input_b + static_cast<size_t>(col) * k;
  const float* sb_row = weight_scale + static_cast<size_t>(col) * k_blocks;

  constexpr int kElemsPerLane = 16;
  const int stride = 32 * kElemsPerLane;  // 512 elements per warp iteration

  float acc[RowsPerWarp] = {};
  for (int base = 0; base < k; base += stride) {
    const int koff = base + lane * kElemsPerLane;
    if (koff < k) {
      const uint4 b_raw = *reinterpret_cast<const uint4*>(b_row + koff);
      const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&b_raw);
      const int kb = koff / block_size;
      const float b_scale = sb_row[kb];
#pragma unroll
      for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
        const int row = row_base + row_offset;
        if (row < m) {
          const AType* a_row = input_a + static_cast<size_t>(row) * k;
          float a_vals[16];
          LoadFp8Gemv16A<AType>(a_row + koff, a_vals);

          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < kElemsPerLane; ++i) {
            partial += a_vals[i] * static_cast<float>(bp[i]);
          }
          acc[row_offset] += partial * b_scale;
        }
      }
    }
  }

#pragma unroll
  for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[row_offset] += __shfl_down_sync(0xffffffffu, acc[row_offset], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
      const int row = row_base + row_offset;
      if (row < m) {
        float result = acc[row_offset];
        if (bias != nullptr) {
          result += ToFloat<AType>(bias[col]);
        }
        output[static_cast<size_t>(row) * n + col] = FromFloat<AType>(result);
      }
    }
  }
}

}  // namespace

#endif  // !DISABLE_FLOAT8_TYPES && CUDA_VERSION >= 11080

Status LaunchDequantizeBlockScaledFp8(void* b_dequant,
                                      const void* b_fp8,
                                      const float* weight_scale,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
  const long long total = static_cast<long long>(n) * k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(b_fp8);
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
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
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
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
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
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
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
  const auto launch = [&]<int RowsPerWarp>() {
    const dim3 blocks{static_cast<unsigned int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock),
                      static_cast<unsigned int>((m + RowsPerWarp - 1) / RowsPerWarp)};
    if (is_bf16) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(y), reinterpret_cast<const __nv_bfloat16*>(a), b,
          weight_scale, reinterpret_cast<const __nv_bfloat16*>(bias), m, n, k, block_size, k_blocks);
    } else {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<half*>(y), reinterpret_cast<const half*>(a), b,
          weight_scale, reinterpret_cast<const half*>(bias), m, n, k, block_size, k_blocks);
    }
  };

  if (m == 1) {
    launch.template operator()<1>();
  } else if (m <= 2) {
    launch.template operator()<2>();
  } else if (m <= 4) {
    launch.template operator()<4>();
  } else {
    launch.template operator()<8>();
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
