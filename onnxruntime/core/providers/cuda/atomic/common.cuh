/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/* Modifications Copyright (c) Microsoft. */

#pragma once
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "core/framework/float16.h"

namespace onnxruntime {
namespace cuda {

__device__ __forceinline__ void atomic_add(float *address, float value) {
    atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double *address, double value) {
#if __CUDA_ARCH__ < 600
  unsigned long long* raw_address = reinterpret_cast<unsigned long long*>(address);
  unsigned long long raw_old_value = 0ULL;
  unsigned long long raw_new_value = 0ULL;
  unsigned long long seen_old_value = 0ULL;
  double* const p_old_value = reinterpret_cast<double*>(&raw_old_value);
  double* const p_new_value = reinterpret_cast<double*>(&raw_new_value);
  do {
    *p_old_value = *address;
    *p_new_value = *address + value;
     seen_old_value = atomicCAS(raw_address, raw_old_value, raw_new_value);
  } while (seen_old_value != raw_old_value);
#else
  atomicAdd(address, value);
#endif
}

//
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh
//
__device__ __forceinline__ void atomic_add(half *address, half value) {
#if __CUDA_ARCH__ < 700
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
#else
  atomicAdd(address, value);
#endif
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

// CUDA's atomic_add for half type is too slow. Using half2 will be much faster. To avoid address out of bound,
// we need to pass in the numel so that the element at the edge can be handled separately.
// Ideally we need to deprecate above atomic_add function and use below one for better performance.
// But since the signature is different, we can change it for specific Op kernel once we find it is slow.
// TODO: need to add same logic for BF16.
template <typename T>
__device__ __forceinline__ void AtomicAdd(T *start_addr, size_t index, const size_t numel, T value) {
  ORT_UNUSED_PARAMETER(numel);
  atomic_add(start_addr + index, value);
}

template <>
__device__ __forceinline__ void AtomicAdd<half>(half* start_addr, size_t index, const size_t numel, half value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
  atomic_add(start_addr + index, value);
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  half* target_addr = reinterpret_cast<half*>(start_addr + index);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    atomicAdd(start_addr + index, value);
  }
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
