// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

namespace {

template <typename T>
__device__ __forceinline__ T FromFloat(float v);

template <>
__device__ __forceinline__ half FromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 FromFloat<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float ToFloat(T v);

template <>
__device__ __forceinline__ float ToFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float ToFloat<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__global__ void DequantizeNvFp4Kernel(T* __restrict__ out,
                                      const uint8_t* __restrict__ b_packed,
                                      const uint8_t* __restrict__ weight_scale,
                                      const float* __restrict__ weight_scale_2,
                                      int n,
                                      int k,
                                      int k_blocks,
                                      int block_size) {
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int row = static_cast<int>(idx / half_k);
  const int pair = static_cast<int>(idx - static_cast<long long>(row) * half_k);
  const int k0 = pair << 1;

  const uint8_t packed = b_packed[idx];
  const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packed), __NV_E2M1);
  const __half2 h2 = __half2(hr);
  const float2 v = __half22float2(h2);

  const float g = *weight_scale_2;
  const int blk0 = k0 / block_size;
  const int blk1 = (k0 + 1) / block_size;
  const float s0 = __half2float(__nv_cvt_fp8_to_halfraw(
                       static_cast<__nv_fp8_storage_t>(weight_scale[row * k_blocks + blk0]), __NV_E4M3)) *
                   g;
  const float s1 = __half2float(__nv_cvt_fp8_to_halfraw(
                       static_cast<__nv_fp8_storage_t>(weight_scale[row * k_blocks + blk1]), __NV_E4M3)) *
                   g;

  const long long out_base = static_cast<long long>(row) * k + k0;
  out[out_base] = FromFloat<T>(v.x * s0);
  out[out_base + 1] = FromFloat<T>(v.y * s1);
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

// -----------------------------------------------------------------------------
// Fused NVFP4 weight-only GEMV fast path for the decode phase (small M).
//
// Each warp computes one output element Y[row, col]. The 32 lanes cooperatively
// reduce over K reading the packed NVFP4 weight directly (two E2M1 values per
// byte) with 16-byte coalesced loads, so the weight is streamed exactly once and
// no [N, K] dequantized buffer is materialized. Each lane consumes 32 contiguous
// K elements = 16 packed bytes, which span exactly two 16-element blocks; the two
// per-block E4M3 scales are folded in per half. The global fp32 scale is applied
// once after the warp reduction. Runs on any architecture with NVFP4 conversion
// intrinsics (CUDA >= 12.8), including SM90 and SM120.
template <typename T>
__device__ __forceinline__ void LoadFp4Gemv32A(const T* ptr, float (&out)[32]);

template <>
__device__ __forceinline__ void LoadFp4Gemv32A<half>(const half* ptr, float (&out)[32]) {
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 raw = p[j];
    const half* v = reinterpret_cast<const half*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[j * 8 + i] = __half2float(v[i]);
    }
  }
}

template <>
__device__ __forceinline__ void LoadFp4Gemv32A<nv_bfloat16>(const nv_bfloat16* ptr, float (&out)[32]) {
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 raw = p[j];
    const nv_bfloat16* v = reinterpret_cast<const nv_bfloat16*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[j * 8 + i] = __bfloat162float(v[i]);
    }
  }
}

template <typename T>
__global__ void MatMulBlockQuantizedFp4WeightGemvKernel(T* __restrict__ y,
                                               const T* __restrict__ a,
                                               const uint8_t* __restrict__ b_packed,
                                               const uint8_t* __restrict__ weight_scale,
                                               const float* __restrict__ weight_scale_2,
                                               const T* __restrict__ bias,
                                               int m,
                                               int n,
                                               int k,
                                               int k_blocks) {
  const int lane = threadIdx.x;                           // 0..31
  const int col = blockIdx.x * blockDim.y + threadIdx.y;  // n
  const int row = blockIdx.y;                             // m
  if (row >= m || col >= n) {
    return;
  }

  const T* a_row = a + static_cast<size_t>(row) * k;
  const uint8_t* b_row = b_packed + static_cast<size_t>(col) * (k >> 1);
  const uint8_t* ws_row = weight_scale + static_cast<size_t>(col) * k_blocks;

  constexpr int kBlockSize = 16;
  constexpr int kElemsPerLane = 32;       // two 16-element blocks
  const int stride = 32 * kElemsPerLane;  // 1024 elements per warp iteration

  float acc = 0.0f;
  for (int base = 0; base < k; base += stride) {
    const int koff = base + lane * kElemsPerLane;
    if (koff < k) {
      const uint4 packed = *reinterpret_cast<const uint4*>(b_row + (koff >> 1));
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);
      float b_vals[32];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(
            static_cast<__nv_fp4x2_storage_t>(bytes[i]), __NV_E2M1);
        const float2 f = __half22float2(__half2(hr));
        b_vals[i * 2] = f.x;
        b_vals[i * 2 + 1] = f.y;
      }

      float a_vals[32];
      LoadFp4Gemv32A<T>(a_row + koff, a_vals);

      const int kb0 = koff / kBlockSize;
      const int kb1 = kb0 + 1;
      float p0 = 0.0f;
      float p1 = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        p0 += a_vals[i] * b_vals[i];
      }
#pragma unroll
      for (int i = 16; i < 32; ++i) {
        p1 += a_vals[i] * b_vals[i];
      }
      const float s0 = __half2float(__nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(ws_row[kb0]), __NV_E4M3));
      const float s1 = __half2float(__nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(ws_row[kb1]), __NV_E4M3));
      acc += p0 * s0 + p1 * s1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) {
    float result = acc * (*weight_scale_2);
    if (bias != nullptr) {
      result += ToFloat<T>(bias[col]);
    }
    y[static_cast<size_t>(row) * n + col] = FromFloat<T>(result);
  }
}

}  // namespace

#endif  // CUDA_VERSION >= 12080

Status LaunchDequantizeNvFp4(void* b_dequant,
                             const void* b_packed,
                             const void* weight_scale,
                             const float* weight_scale_2,
                             int n,
                             int k,
                             int block_size,
                             bool is_bf16,
                             cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_packed);
  const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);

  if (is_bf16) {
    DequantizeNvFp4Kernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  } else {
    DequantizeNvFp4Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(b_dequant);
  ORT_UNUSED_PARAMETER(b_packed);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(weight_scale_2);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

Status LaunchAddBiasNvFp4(void* y,
                          const void* bias,
                          int m,
                          int n,
                          bool is_bf16,
                          cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const long long total = static_cast<long long>(m) * n;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    AddBiasKernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(y), reinterpret_cast<const nv_bfloat16*>(bias), m, n);
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
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

Status LaunchMatMulBlockQuantizedFp4WeightGemv(void* y,
                                      const void* a,
                                      const void* b_packed,
                                      const void* weight_scale,
                                      const float* weight_scale_2,
                                      const void* bias,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool is_bf16,
                                      cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  if (m <= 0 || n <= 0 || k <= 0) {
    return Status::OK();
  }
  // This kernel is hard-coded for block_size == 16 and assumes K is a multiple of 32 so that each
  // warp lane always owns a full 32-element slice (and one E4M3 scale per 16-element block). Guard
  // against misuse if this helper is ever reused outside the callers that already check these.
  ORT_RETURN_IF_NOT(block_size == 16, "MatMulBlockQuantizedFp4Weight GEMV requires block_size == 16, got ", block_size, ".");
  ORT_RETURN_IF_NOT(k % 32 == 0, "MatMulBlockQuantizedFp4Weight GEMV requires K divisible by 32, got ", k, ".");
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kWarpsPerBlock = 8;
  const dim3 threads{32, kWarpsPerBlock};
  const dim3 blocks{static_cast<unsigned int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock),
                    static_cast<unsigned int>(m)};
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_packed);
  const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);
  if (is_bf16) {
    MatMulBlockQuantizedFp4WeightGemvKernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(y), reinterpret_cast<const nv_bfloat16*>(a), bp, ws, weight_scale_2,
        reinterpret_cast<const nv_bfloat16*>(bias), m, n, k, k_blocks);
  } else {
    MatMulBlockQuantizedFp4WeightGemvKernel<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(y), reinterpret_cast<const half*>(a), bp, ws, weight_scale_2,
        reinterpret_cast<const half*>(bias), m, n, k, k_blocks);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(b_packed);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(weight_scale_2);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
