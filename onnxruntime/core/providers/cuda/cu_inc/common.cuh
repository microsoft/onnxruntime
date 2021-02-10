// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <mutex>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

namespace onnxruntime {
namespace cuda {

// float16 arithmetic is supported after sm5.3 with intrinsics, and cuda does not provide fallback for lower versions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
__device__ __forceinline__ half operator+(const half& lh, const half& rh) { return half((float)lh + (float)rh); }
__device__ __forceinline__ half operator-(const half& lh, const half& rh) { return half((float)lh - (float)rh); }
__device__ __forceinline__ half operator*(const half& lh, const half& rh) { return half((float)lh * (float)rh); }
__device__ __forceinline__ half operator/(const half& lh, const half& rh) { return half((float)lh / (float)rh); }

__device__ __forceinline__ half& operator+=(half& lh, const half& rh) {
  lh = half((float)lh + (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator-=(half& lh, const half& rh) {
  lh = half((float)lh - (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator*=(half& lh, const half& rh) {
  lh = half((float)lh * (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator/=(half& lh, const half& rh) {
  lh = half((float)lh / (float)rh);
  return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00 equating to half(1.0f), to avoid the extra conversion */
__device__ __forceinline__ __half& operator++(__half& h) {
  h = half((float)h + 1.0f);
  return h;
}
__device__ __forceinline__ __half& operator--(__half& h) {
  h = half((float)h - 1.0f);
  return h;
}
__device__ __forceinline__ __half operator++(__half& h, int) {
  half ret = h;
  h = half((float)h + 1);
  return ret;
}
__device__ __forceinline__ __half operator--(__half& h, int) {
  half ret = h;
  h = half((float)h - 1);
  return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ half operator+(const half& h) { return h; }
__device__ __forceinline__ half operator-(const half& h) { return half(-(float)h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const half& lh, const half& rh) { return (float)lh == (float)rh; }
__device__ __forceinline__ bool operator!=(const half& lh, const half& rh) { return (float)lh != (float)rh; }
__device__ __forceinline__ bool operator>(const half& lh, const half& rh) { return (float)lh > (float)rh; }
__device__ __forceinline__ bool operator<(const half& lh, const half& rh) { return (float)lh < (float)rh; }
__device__ __forceinline__ bool operator>=(const half& lh, const half& rh) { return (float)lh >= (float)rh; }
__device__ __forceinline__ bool operator<=(const half& lh, const half& rh) { return (float)lh <= (float)rh; }
#endif

template <typename T>
__device__ __inline__ T _Ceil(T a);

template <>
__device__ __inline__ float _Ceil(float a) { return ceilf(a); }

template <>
__device__ __inline__ double _Ceil(double a) { return ceil(a); }

template <>
__device__ __inline__ half _Ceil(half a) { return half(ceilf((float)a)); }

template <typename T>
__device__ __inline__ T _Floor(T a);

template <>
__device__ __inline__ float _Floor(float a) { return floorf(a); }

template <>
__device__ __inline__ double _Floor(double a) { return floor(a); }

template <>
__device__ __inline__ half _Floor(half a) { return half(floorf((float)a)); }

template <typename T>
__device__ __inline__ T _Sqrt(T a);

template <>
__device__ __inline__ float _Sqrt(float a) { return sqrtf(a); }

template <>
__device__ __inline__ double _Sqrt(double a) { return sqrt(a); }

template <>
__device__ __inline__ half _Sqrt(half a) { return half(sqrtf((float)a)); }

template <typename T>
__device__ __inline__ T _Erf(T a);

template <>
__device__ __inline__ float _Erf(float a) { return erff(a); }

template <>
__device__ __inline__ double _Erf(double a) { return erf(a); }

template <>
__device__ __inline__ half _Erf(half a) { return half(erff((float)a)); }

template <typename T>
__device__ __inline__ T _Round(T a);

template <>
__device__ __inline__ float _Round(float a) { return rintf(a); }

template <>
__device__ __inline__ double _Round(double a) { return rint(a); }

template <>
__device__ __inline__ half _Round(half a) {
#if __CUDA_ARCH__ < 530
  return half(rintf((float)a));
#else
  return hrint(a);
#endif
}

template <typename T>
__device__ __inline__ T _Cos(T a);

template <>
__device__ __inline__ float _Cos(float a) { return cosf(a); }

template <>
__device__ __inline__ double _Cos(double a) { return cos(a); }

template <>
__device__ __inline__ half _Cos(half a) {
#if __CUDA_ARCH__ < 530
  return half(cosf((float)a));
#else
  return hcos(a);
#endif
}

template <typename T>
__device__ __inline__ T _Sin(T a);

template <>
__device__ __inline__ float _Sin(float a) { return sinf(a); }

template <>
__device__ __inline__ double _Sin(double a) { return sin(a); }

template <>
__device__ __inline__ half _Sin(half a) {
#if __CUDA_ARCH__ < 530
  return half(sinf((float)a));
#else
  return hsin(a);
#endif
}

template <typename T>
__device__ __inline__ T _Exp(T a);

template <>
__device__ __inline__ float _Exp(float a) { return expf(a); }

template <>
__device__ __inline__ double _Exp(double a) { return exp(a); }

template <>
__device__ __inline__ half _Exp(half a) { return half(expf((float)a)); }

template <typename T>
__device__ __inline__ T _Log(T a);

template <>
__device__ __inline__ float _Log(float a) { return logf(a); }

template <>
__device__ __inline__ double _Log(double a) { return log(a); }

template <>
__device__ __inline__ half _Log(half a) { return half(logf((float)a)); }

template <typename T>
__device__ __inline T _Tanh(T a);

template <>
__device__ __inline__ float _Tanh(float a) { return tanhf(a); }

template <>
__device__ __inline__ double _Tanh(double a) { return tanh(a); }

template <>
__device__ __inline__ half _Tanh(half a) { return half(tanhf((float)a)); }

template <>
__device__ __inline__ half2 _Tanh(half2 a) {
  float2 tmp = (__half22float2(a));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22half2_rn(tmp);
}

// Capture permutations of int32/64/float/double
template <typename T, typename T1>
__device__ __inline__ T _Pow(T a, T1 b) {
  return static_cast<T>(pow(static_cast<double>(a), static_cast<double>(b)));
}

template <>
__device__ __inline__ float _Pow(float a, float b) { return powf(a, b); }

template <>
__device__ __inline__ double _Pow(double a, double b) { return pow(a, b); }

template <>
__device__ __inline__ half _Pow(half a, half b) { return half(powf((float)a, (float)b)); }

template <typename T>
__device__ __inline__ T _Min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ __inline__ T _Max(T a, T b) { return a > b ? a : b; }

template <typename T>
__device__ __inline__ T _Abs(T a) { return a > (T)0 ? a : -a; }

template <typename T>
__device__ __inline__ T _Normcdf(T a);

template <>
__device__ __inline__ float _Normcdf(float a) { return normcdff(a); }

template <>
__device__ __inline__ double _Normcdf(double a) { return normcdf(a); }

template <>
__device__ __inline__ half _Normcdf(half a) { return half(normcdff((float)a)); }

template <typename T>
__device__ __inline__ T _Gelu(T a) {
  return a * _Normcdf(a);
}

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__device__ __inline__ nv_bfloat16 _Sqrt(nv_bfloat16 a) { return nv_bfloat16(sqrtf(static_cast<float>(a))); }

template <>
__device__ __inline__ nv_bfloat16 _Exp(nv_bfloat16 a) { return nv_bfloat16(expf(static_cast<float>(a))); }

template <>
__device__ __inline__ nv_bfloat16 _Log(nv_bfloat16 a) { return nv_bfloat16(logf(static_cast<float>(a))); }

template <>
__device__ __inline__ nv_bfloat16 _Tanh(nv_bfloat16 a) { return nv_bfloat16(tanhf(static_cast<float>(a))); }

template <>
__device__ __inline__ nv_bfloat162 _Tanh(nv_bfloat162 a) {
  float2 tmp = (__bfloat1622float2(a));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22bfloat162_rn(tmp);
}

template <>
__device__ __inline__ nv_bfloat16 _Normcdf(nv_bfloat16 a) { return nv_bfloat16(normcdff(static_cast<float>(a))); }
#endif

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)      \
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N)                                          \
    return;

// CUDA_KERNEL_ASSERT is a macro that wraps an assert() call inside cuda kernels.
// This is not supported by Apple platforms so we special case it.
// See http://docs.nvidia.com/cuda/cuda-c-programming-guide/#assertion
#if defined(__APPLE__) || defined(__HIP_PLATFORM_HCC__)
#define CUDA_KERNEL_ASSERT(...)
#else  // __APPLE__
#define CUDA_KERNEL_ASSERT(...) assert(__VA_ARGS__)
#endif  // __APPLE__

// WARP related definitions and functions
constexpr int GPU_WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, srcLane, width);
#else
  return __shfl(value, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(mask, value, delta, width);
#else
  return __shfl_up(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
