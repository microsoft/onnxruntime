// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <mutex>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {

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
  return hrint(a);
}

template <typename T>
__device__ __inline__ T _Cos(T a);

template <>
__device__ __inline__ float _Cos(float a) { return cosf(a); }

template <>
__device__ __inline__ double _Cos(double a) { return cos(a); }

template <>
__device__ __inline__ half _Cos(half a) { return hcos(a); }

template <typename T>
__device__ __inline__ T _Sin(T a);

template <>
__device__ __inline__ float _Sin(float a) { return sinf(a); }

template <>
__device__ __inline__ double _Sin(double a) { return sin(a); }

template <>
__device__ __inline__ half _Sin(half a) { return hsin(a); }

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


// We would like to use 64-bit integer to support large matrices. However, HIP seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : HIP_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};


#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)          \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;     \
  if (id >= N)                                              \
    return;

#define HIP_KERNEL_ASSERT(...)

// WARP related definitions and functions
constexpr int GPU_WARP_SIZE = 64;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl(value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_xor(value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_down(value, delta, width);
}

}  // namespace rocm
}  // namespace onnxruntime
