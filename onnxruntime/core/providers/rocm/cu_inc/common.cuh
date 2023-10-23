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

/// Arithmetic for BFloat16

__device__ __forceinline__ BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator-(const BFloat16& a) { return -static_cast<float>(a); }

__device__ __forceinline__ BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

__device__ __forceinline__ BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

__device__ __forceinline__ BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

__device__ __forceinline__ BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

__device__ __forceinline__ float operator+(BFloat16 a, float b) { return static_cast<float>(a) + b; }
__device__ __forceinline__ float operator-(BFloat16 a, float b) { return static_cast<float>(a) - b; }
__device__ __forceinline__ float operator*(BFloat16 a, float b) { return static_cast<float>(a) * b; }
__device__ __forceinline__ float operator/(BFloat16 a, float b) { return static_cast<float>(a) / b; }

__device__ __forceinline__ float operator+(float a, BFloat16 b) { return a + static_cast<float>(b); }
__device__ __forceinline__ float operator-(float a, BFloat16 b) { return a - static_cast<float>(b); }
__device__ __forceinline__ float operator*(float a, BFloat16 b) { return a * static_cast<float>(b); }
__device__ __forceinline__ float operator/(float a, BFloat16 b) { return a / static_cast<float>(b); }

__device__ __forceinline__ float& operator+=(float& a, const BFloat16& b) { return a += static_cast<float>(b); }
__device__ __forceinline__ float& operator-=(float& a, const BFloat16& b) { return a -= static_cast<float>(b); }
__device__ __forceinline__ float& operator*=(float& a, const BFloat16& b) { return a *= static_cast<float>(b); }
__device__ __forceinline__ float& operator/=(float& a, const BFloat16& b) { return a /= static_cast<float>(b); }

/// Arithmetic with doubles

__device__ __forceinline__ double operator+(BFloat16 a, double b) { return static_cast<double>(a) + b; }
__device__ __forceinline__ double operator-(BFloat16 a, double b) { return static_cast<double>(a) - b; }
__device__ __forceinline__ double operator*(BFloat16 a, double b) { return static_cast<double>(a) * b; }
__device__ __forceinline__ double operator/(BFloat16 a, double b) { return static_cast<double>(a) / b; }

__device__ __forceinline__ double operator+(double a, BFloat16 b) { return a + static_cast<double>(b); }
__device__ __forceinline__ double operator-(double a, BFloat16 b) { return a - static_cast<double>(b); }
__device__ __forceinline__ double operator*(double a, BFloat16 b) { return a * static_cast<double>(b); }
__device__ __forceinline__ double operator/(double a, BFloat16 b) { return a / static_cast<double>(b); }

// Overloading < and > operators

__device__ __forceinline__ bool operator==(BFloat16& lhs, BFloat16& rhs) { return float(lhs) == float(rhs); }
__device__ __forceinline__ bool operator!=(BFloat16& lhs, BFloat16& rhs) { return float(lhs) != float(rhs); }
__device__ __forceinline__ bool operator>(BFloat16& lhs, BFloat16& rhs) { return float(lhs) > float(rhs); }
__device__ __forceinline__ bool operator<(BFloat16& lhs, BFloat16& rhs) { return float(lhs) < float(rhs); }

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
__device__ __inline__ half _Cos(half a) {
  return hcos(a);
}

template <typename T>
__device__ __inline__ T _Sin(T a);

template <>
__device__ __inline__ float _Sin(float a) { return sinf(a); }

template <>
__device__ __inline__ double _Sin(double a) { return sin(a); }

template <>
__device__ __inline__ half _Sin(half a) {
  return hsin(a);
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
__device__ __inline__ T _Signum(T a, std::false_type /* is_signed */) { return T(0) < a; }

template <typename T>
__device__ __inline__ T _Signum(T a, std::true_type /* is_signed */) { return (T(0) < a) - (a < T(0)); }

template <typename T>
__device__ __inline__ T _Sign(T a) { return _Signum(a, std::is_signed<T>()); }

template <>
__device__ __inline__ half _Sign(half a) { return _Signum(a, std::true_type()); }

template <typename T>
__device__ __inline__ T _Normcdf(T a);

template <>
__device__ __inline__ float _Normcdf(float a) { return normcdff(a); }

template <>
__device__ __inline__ double _Normcdf(double a) { return normcdf(a); }

template <>
__device__ __inline__ half _Normcdf(half a) { return half(normcdff((float)a)); }

template <>
__device__ __inline__ BFloat16 _Sqrt(BFloat16 a) { return sqrtf(static_cast<float>(a)); }

template <>
__device__ __inline__ BFloat16 _Exp(BFloat16 a) { return expf(static_cast<float>(a)); }

template <>
__device__ __inline__ BFloat16 _Log(BFloat16 a) { return logf(static_cast<float>(a)); }

template <>
__device__ __inline__ BFloat16 _Tanh(BFloat16 a) { return tanhf(static_cast<float>(a)); }

template <>
__device__ __inline__ BFloat16 _Normcdf(BFloat16 a) { return normcdff(static_cast<float>(a)); }

template <typename T>
__device__ __inline__ T _Gelu(T a) {
  return a * _Normcdf(a);
}

template <>
__device__ __inline__ half _Gelu(half a) {
  const half kHalf = half(0.5);
  const half kOne = half(1.0);
  const half kAlpha = half(M_SQRT1_2);
  return a * kHalf * (kOne + _Erf(kAlpha * a));
}

template <typename T>
__device__ __inline__ T _Mod(T a, T b) {
  T r = a % b;
  T zero = T(0);
  if ((r > zero && b < zero) || (r < zero && b > zero)) {
    r += b;
  }
  return r;
}

template <typename T>
__device__ __inline__ T _Fmod(T a, T b) {
  return a % b;
}

template <>
__device__ __inline__ float _Fmod(float a, float b) {
  return fmodf(a, b);
}

template <>
__device__ __inline__ double _Fmod(double a, double b) {
  return fmod(a, b);
}

template <>
__device__ __inline__ half _Fmod(half a, half b) {
  return fmodf((float)a, (float)b);
}

template <>
__device__ __inline__ BFloat16 _Fmod(BFloat16 a, BFloat16 b) {
  return fmodf((float)a, (float)b);
}

// We would like to use 64-bit integer to support large matrices. However, ROCM seems to support only 32-bit integer
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

// aligned vector generates vectorized load/store on ROCM
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)      \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x; \
  if (id >= N)                                          \
    return;

// HIP_KERNEL_ASSERT is a macro that wraps an assert() call inside rocm kernels.
#define HIP_KERNEL_ASSERT(...) assert(__VA_ARGS__)

// WARP related definitions and functions
constexpr int GPU_WARP_SIZE = warpSize;
inline int GPU_WARP_SIZE_HOST = warpSizeDynamic();

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
  ORT_UNUSED_PARAMETER(mask);
  return __shfl(value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
  ORT_UNUSED_PARAMETER(mask);
  return __shfl_xor(value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
  ORT_UNUSED_PARAMETER(mask);
  return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
  ORT_UNUSED_PARAMETER(mask);
  return __shfl_down(value, delta, width);
}

}  // namespace rocm
}  // namespace onnxruntime
