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

__device__ __forceinline__ void atomic_mul(half* address, half val) {
    unsigned short int* address_as_short = reinterpret_cast<unsigned short int*>(address);
    unsigned short int old = *address_as_short, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_short, assumed,
                        __half_as_short(val * __short_as_half(assumed)));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_mul(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_mul(double* address, double val) {
    unsigned long long int* address_as_long_long = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_long_long, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_long_long, assumed,
                        __double_as_longlong(val * __longlong_as_double(assumed)));
    } while (assumed != old);
}

// Obtained from pytorch/aten/src/ATen/cuda/Atomic.cuh.
//
// This function compute 8-bit atomic binary operation using 32-bit atomicCAS.
//
// E.g., Making OneByteType int8_t and BinaryFunc
// struct AddFunc {
//   __device__ __forceinline__ OneByteType operator()(OneByteType a, OneByteType b) const {
//     return a + b;
// }
// makes this function atomic_add for int8_t.
template<typename OneByteType, typename BinaryFunc>
__device__ __forceinline__ void atomic_byte_func_with_4byte_cas(OneByteType* address, OneByteType val) {
    // Number of bytes to the lower 4-byte aligned address.
    // If the current address is b1010"10", then offset = b10 = 2,
    // which means the current address is 2 bytes away from
    // the lower 4-byte aligned address b1010"00".
    size_t offset = (size_t)address & 3;
    // Find an new 4-byte aligned address `address_as_ui` lower than
    // or equal to `address`. Lower than `address` so that the actual
    // int8_t byte is in the 4-byte word that we load.
    //
    // This address has the following properties:
    //   1. It is 4-byte aligned.
    //   2. It is lower than or equal to `address`.
    //   3. De-referencing this address may return
    //      a uint32_t value that contains the same int8_t
    //      value indicated by `address`.
    //
    // E.g.,
    //  address = b101010
    //  offset = b101010 & b000011 = b10 = 2
    //  (char*)address - offset => (char*)b101010 - b000010 => b1010"00",
    // which is (32-bit aligned).
    uint32_t * address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t old = *address_as_ui;
    // E.g., offset = 2.
    // address_as_ui is an address 2 bytes lower than `address`.
    //
    // ..... byte 3 ..... | ..... byte 2 ..... | ..... byte 1 ..... | ..... byte 0 .....
    //                  ^                    ^                                         ^
    //                  |                    |                                         |
    //                  |                  address <--- offset * 8 (bit)----->  address_as_ui
    //                  |                                                              ^
    //                  |                                                              |
    //                  ------------------------- *address_as_ui -----------------------
    //
    // This visualization shows
    //  1. the 32-bit word at address_as_ui.
    //  2. the gap between address_as_ui and address.
    //  3. *address_as_ui contains the int8_t value at `address`.
    uint32_t shift = offset * 8;
    uint32_t old_byte;
    uint32_t newval;
    uint32_t assumed;
    do {
      assumed = old;
      // Select 8-bit value from 32-bit word. Assume offset = 2 (byte), so
      // we want to select the 3rd byte (byte 2 below) from the word.
      //
      // Journey of a 32-bit value:
      //
      // ..... byte 3 ..... | ..... byte 2 ..... | ..... byte 1 ..... | ..... byte 0 .....
      //
      //                                         |
      //                                         |  old >> offset * 8, where offset = 2.
      //                                         |  Effectively, push lower two bytes
      //                                         |  out of the word.
      //                                         V
      //
      //      00000000      |      00000000      | ..... byte 3 ..... | ..... byte 2 .....
      //
      //                                                              |  apply bit-wise AND,
      //                                                              |  & 0xff (i.e., & b11111111),
      //                                                              |  so that we only keep
      //                                                              |  the byte of interest.
      //                                                              |  Otherwise, overflow may
      //                                                              |  happen when casting this
      //                                                              |  32-bit value to int8_t.
      //                                                              V
      //
      //      00000000      |      00000000      |      00000000      | ..... byte 2 .....
      old_byte = (old >> shift) & 0xff;
      // Use + for atomic addition, * for atomic multiplication, / for atomic division.
      newval = static_cast<uint32_t>(val * static_cast<OneByteType>(old_byte));
      // Journey of a 32-bit value (cont'd):
      //
      // old
      // ..... byte 3 ..... | ..... byte 2 ..... | ..... byte 1 ..... | ..... byte 0 .....
      //
      // 0x000000ff
      //      00000000      |      00000000      |      00000000      |      11111111
      //
      // 0x000000ff << shift
      //      00000000      |      11111111      |      00000000      |      00000000
      //
      // ~(0x000000ff << shift)
      //      11111111      |      00000000      |      11111111      |      11111111
      //
      // old & ~(0x000000ff << shift)
      // ..... byte 3 ..... |      00000000      | ..... byte 1 ..... | ..... byte 0 .....
      //
      // newval << shift
      //      00000000      | ... new byte 2 ... |      00000000      |      00000000
      //
      // (old & ~(0x000000ff << shift)) | (newval << shift)
      // ..... byte 3 ..... | ... new byte 2 ... | ..... byte 1 ..... | ..... byte 0 .....
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
}

struct AddFunc {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

struct MulFunc {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a * b;
  }
};

struct MaxFunc {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return b > a ? b : a;
  }
};

struct MinFunc {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return b < a ? b : a;
  }
};

__device__ __forceinline__ void atomic_add(int8_t* address, int8_t value) {
  atomic_byte_func_with_4byte_cas<int8_t, AddFunc>(address, value);
}

__device__ __forceinline__ void atomic_mul(int8_t* address, int8_t value) {
  atomic_byte_func_with_4byte_cas<int8_t, MulFunc>(address, value);
}

__device__ __forceinline__ void atomic_max(int8_t* address, int8_t value) {
  atomic_byte_func_with_4byte_cas<int8_t, MaxFunc>(address, value);
}

__device__ __forceinline__ void atomic_min(int8_t* address, int8_t value) {
  atomic_byte_func_with_4byte_cas<int8_t, MinFunc>(address, value);
}

}  // namespace cuda
}  // namespace onnxruntime
