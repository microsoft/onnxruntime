// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/framework/float16.h"

typedef __half half;

namespace onnxruntime {
namespace rocm {

__device__ __forceinline__ void atomic_add(float *address, float value) {
    atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double *address, double value) {
  atomicAdd(address, value);
}

//
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh
//
__device__ __forceinline__ void atomic_add(half *address, half value) {
  unsigned int* base_address = (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *base_address;
  unsigned int assumed;
  unsigned short x;

  do {
    assumed = old;
    x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    x = __half_as_short(__float2half(__half2float(*reinterpret_cast<const __half*>(&x)) + __half2float(value)));
    old = (size_t)address & 2 ? (old & 0xffff) | (x << 16) : (old & 0xffff0000) | x;
    old = atomicCAS(base_address, assumed, old);
  } while (assumed != old);
}

__device__ __forceinline__ void atomic_add(BFloat16* address, BFloat16 value) {
  unsigned int* base_address =
      reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *base_address;
  unsigned int assumed;
  BFloat16 bsum;
  do {
    assumed = old;
    bsum.val = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    bsum = bsum + value;
    old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (bsum.val << 16) : (old & 0xffff0000) | bsum.val;
    old = atomicCAS(base_address, assumed, old);
  } while (assumed != old);
}

// This function is added to speed up atomic add for half/bf16 type on CUDA. For ROCM, use default implementation.
template <typename T>
__device__ __forceinline__ void AtomicAdd(T *start_addr, size_t index, const size_t numel, T value) {
  ORT_UNUSED_PARAMETER(numel);
  atomic_add(start_addr + index, value);
}

}  // namespace rocm
}  // namespace onnxruntime
