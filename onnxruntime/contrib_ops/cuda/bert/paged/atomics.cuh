// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::paged {

template <typename T>
__forceinline__ __device__ T
atomic_load_relaxed(const T* address) {
  const volatile T* v{address};
  return *v;
}

__forceinline__ __device__ float
atomic_max(float* address, float val) {
  union {
    uint32_t u32;
    float f32;
  } cvt;
  cvt.f32 = val;

  // https://stackoverflow.com/a/51549250/2091555
  // https://github.com/NVIDIA/cutlass/blob/ffa34e7075/include/cutlass/functional.h#L621-L623
  return (cvt.u32 & 0x80000000) == 0  // extract signbit, aka, if positive
             ? __int_as_float(atomicMax((int*)address, __float_as_int(val)))
             : __uint_as_float(atomicMin((unsigned int*)address, __float_as_uint(val)));
}

__forceinline__ __device__ float
atomic_min(float* address, float val) {
  union {
    uint32_t u32;
    float f32;
  } cvt;
  cvt.f32 = val;

  // https://stackoverflow.com/a/51549250/2091555
  return (cvt.u32 & 0x80000000) == 0  // extract signbit, aka, if positive
             ? __int_as_float(atomicMin((int*)address, __float_as_int(val)))
             : __uint_as_float(atomicMax((unsigned int*)address, __float_as_uint(val)));
}

// temporarily put them here
template <typename T>
__forceinline__ __device__ T volatile_load(const volatile T* address) {
  return *address;
}

template <>
__forceinline__ __device__ float2 volatile_load(const volatile float2* address) {
  union {
    float2 f32x2;
    uint64_t u64;
  };
  u64 = *reinterpret_cast<const volatile uint64_t*>(address);
  return f32x2;
}

template <>
__forceinline__ __device__ int2 volatile_load(const volatile int2* address) {
  union {
    int2 i32x2;
    uint64_t u64;
  };
  u64 = *reinterpret_cast<const volatile uint64_t*>(address);
  return i32x2;
}

template <typename T>
__forceinline__ __device__ void volatile_store(volatile T* address, const T& val) {
  *address = val;
}

template <>
__forceinline__ __device__ void volatile_store(volatile float2* address, const float2& val) {
  union {
    float2 f32x2;
    uint64_t u64;
  };
  f32x2 = val;
  *reinterpret_cast<volatile uint64_t*>(address) = u64;
}

template <>
__forceinline__ __device__ void volatile_store(volatile int2* address, const int2& val) {
  union {
    int2 i32x2;
    uint64_t u64;
  };
  i32x2 = val;
  *reinterpret_cast<volatile uint64_t*>(address) = u64;
}

}  // namespace onnxruntime::contrib::paged
