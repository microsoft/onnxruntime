// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Compatibility shim: hipify-perl translates #include <cuda_bf16.h> to
// #include <rocm_bf16.h>. This stub re-exports the real HIP BF16 header
// and provides a hip_bfloat162 type (and CUDA __nv_bfloat162 alias) plus
// conversion helpers that hipify does not translate on its own.
//
// These helpers must be safe in BOTH:
//   - device code (hipcc, where __device__/__forceinline__ are active), and
//   - host code compiled with the GNU host compiler (/usr/bin/c++), where
//     hip_bfloat16's `operator float()` is not usable, so we must not rely
//     on static_cast<float>(hip_bfloat16). We use bit manipulation instead.

#pragma once
#include <hip/hip_bfloat16.h>

// CUDA's __nv_bfloat162 is a packed pair of bfloat16 values.
// ROCm does not have an equivalent; provide a compatible struct.
struct __attribute__((aligned(4))) hip_bfloat162 {
  hip_bfloat16 x;
  hip_bfloat16 y;
};

// Hipify leaves the CUDA type name __nv_bfloat162 untouched (it is not in the
// hipify mapping), so define it as an alias for the HIP-compatible struct.
typedef hip_bfloat162 __nv_bfloat162;

// CUDA BF16 conversion intrinsics → HIP equivalents.
// These names are left untouched by hipify, so define them here.
#ifndef __NVCC__
__device__ __forceinline__ hip_bfloat16 __float2bfloat16(float f) {
  return hip_bfloat16(f);
}

__device__ __forceinline__ float __bfloat162float(hip_bfloat16 b) {
  // hip_bfloat16::data holds the 16-bit bfloat16 bit pattern.
  // Convert to float by zero-extending to 32 bits and reinterpreting.
  union {
    uint32_t i;
    float f;
  } u;
  u.i = static_cast<uint32_t>(b.data) << 16;
  return u.f;
}

// Additional CUDA bfloat16 conversion intrinsics not mapped by hipify
__device__ __forceinline__ hip_bfloat16 __uint2bfloat16_rn(unsigned int x) {
  return hip_bfloat16(static_cast<float>(x));
}
__device__ __forceinline__ hip_bfloat16 __ushort2bfloat16_rn(unsigned short x) {
  return hip_bfloat16(static_cast<float>(x));
}
__device__ __forceinline__ hip_bfloat16 __int2bfloat16_rn(int x) {
  return hip_bfloat16(static_cast<float>(x));
}
// __hneg for bfloat16: negate via bit flip of sign bit
__device__ __forceinline__ hip_bfloat16 __hneg(hip_bfloat16 h) {
  union { uint16_t u; } u;
  u.u = h.data ^ 0x8000u;
  hip_bfloat16 r; r.data = u.u; return r;
}

// Mixed float/bfloat16 arithmetic (CUDA allows implicit promotion; HIP requires explicit).
// The full hip_bfloat16.h (under __HIPCC__) only defines hip_bfloat16 op hip_bfloat16,
// not float op hip_bfloat16, so these are needed in both host and device contexts.
// Unary operator- is already provided by amd_hip_bfloat16.h as __HOST_DEVICE__,
// so we must not redeclare it.
__device__ __forceinline__ hip_bfloat16 operator*(float a, hip_bfloat16 b) {
  return hip_bfloat16(a * __bfloat162float(b));
}
__device__ __forceinline__ hip_bfloat16 operator*(hip_bfloat16 a, float b) {
  return hip_bfloat16(__bfloat162float(a) * b);
}
__device__ __forceinline__ hip_bfloat16 operator+(hip_bfloat16 a, float b) {
  return hip_bfloat16(__bfloat162float(a) + b);
}
__device__ __forceinline__ hip_bfloat16 operator+(float a, hip_bfloat16 b) {
  return hip_bfloat16(a + __bfloat162float(b));
}

// Arithmetic operators for hip_bfloat162 (needed by hipified add_bias_transpose.cu).
// Use __bfloat162float + hip_bfloat16(float) constructor because host-only
// compilation (GCC without __HIPCC__) sees hip_bfloat16 as a plain struct
// with no arithmetic operators.
__device__ __forceinline__ hip_bfloat162 operator+(const hip_bfloat162& a, const hip_bfloat162& b) {
  return hip_bfloat162{hip_bfloat16(__bfloat162float(a.x) + __bfloat162float(b.x)),
                       hip_bfloat16(__bfloat162float(a.y) + __bfloat162float(b.y))};
}
__device__ __forceinline__ hip_bfloat162 operator-(const hip_bfloat162& a, const hip_bfloat162& b) {
  return hip_bfloat162{hip_bfloat16(__bfloat162float(a.x) - __bfloat162float(b.x)),
                       hip_bfloat16(__bfloat162float(a.y) - __bfloat162float(b.y))};
}
__device__ __forceinline__ hip_bfloat162 operator*(const hip_bfloat162& a, const hip_bfloat162& b) {
  return hip_bfloat162{hip_bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x)),
                       hip_bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y))};
}
#endif  // __NVCC__
