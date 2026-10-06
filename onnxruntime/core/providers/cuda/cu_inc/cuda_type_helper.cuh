// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>

// cuda_fp8.h only ships with CUDA 11.8+. Guard the include so older toolkits (or builds that
// explicitly disable the float8 types) still compile; the FP8 helpers below are guarded to match.
#if !defined(DISABLE_FLOAT8_TYPES) && defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#define ORT_CUDA_TYPE_HELPER_HAS_FP8 1
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Reinterprets the bits of `from` as a `To`. memcpy is the only strict-aliasing-safe way to do
// this; nvcc lowers it to a plain register move, so it costs nothing versus a reinterpret_cast.
template <typename To, typename From>
__device__ __forceinline__ To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast requires types of identical size");
  To to;
  memcpy(&to, &from, sizeof(To));
  return to;
}

// Convert half/bfloat16 to float.
//
// The bfloat16 conversions are declared __CUDA_HOSTDEVICE_BF16_DECL__ by the CUDA headers, i.e.
// they are emulated below sm_80 rather than being sm_80-only like the bfloat16 *arithmetic*
// intrinsics, so they need no __CUDA_ARCH__ guard on any toolkit ORT supports (>= 11.8).
template <typename T>
__device__ __forceinline__ float to_float(T val) = delete;

template <>
__device__ __forceinline__ float to_float(float val) { return val; }

template <>
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

// Convert float to half/bfloat16/float
template <typename T>
__device__ __forceinline__ T from_float(float val) = delete;

template <>
__device__ __forceinline__ float from_float(float val) { return val; }

template <>
__device__ __forceinline__ half from_float(float val) { return __float2half(val); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// Vector-2 companion type of a 16-bit floating point type, plus the packed math that goes with it.
// Kernels that process two elements per lane use this to stay generic over half/bfloat16.
template <typename T>
struct Vec2Traits;

template <>
struct Vec2Traits<half> {
  using Type = half;
  using Type2 = half2;

  __device__ __forceinline__ static float2 to_float2(const Type2& v) { return __half22float2(v); }
  __device__ __forceinline__ static Type2 mul2(const Type2& a, const Type2& b) { return __hmul2(a, b); }
  // Broadcasts a scalar float into both lanes.
  __device__ __forceinline__ static Type2 splat(float v) { return __float2half2_rn(v); }
};

template <>
struct Vec2Traits<__nv_bfloat16> {
  using Type = __nv_bfloat16;
  using Type2 = __nv_bfloat162;

  __device__ __forceinline__ static float2 to_float2(const Type2& v) { return __bfloat1622float2(v); }
  __device__ __forceinline__ static Type2 mul2(const Type2& a, const Type2& b) { return __hmul2(a, b); }
  // Broadcasts a scalar float into both lanes.
  __device__ __forceinline__ static Type2 splat(float v) { return __float2bfloat162_rn(v); }
};

template <typename T>
using Vec2 = typename Vec2Traits<T>::Type2;

// Overloads for call sites that already hold the packed type and do not need the trait.
__device__ __forceinline__ float2 to_float2(const half2& v) { return __half22float2(v); }
__device__ __forceinline__ float2 to_float2(const __nv_bfloat162& v) { return __bfloat1622float2(v); }

#if defined(ORT_CUDA_TYPE_HELPER_HAS_FP8)
// Raw E4M3 byte <-> float. Block-scale tensors store E4M3 as plain uint8_t rather than as
// __nv_fp8_e4m3, so the conversion has to go through the bit-level intrinsics.
__device__ __forceinline__ uint8_t float_to_e4m3(float value) {
  return bit_cast<uint8_t>(__nv_fp8_e4m3(value));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t raw) {
  return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(raw), __NV_E4M3));
}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
