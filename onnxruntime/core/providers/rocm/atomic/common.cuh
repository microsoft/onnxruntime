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

// Disable default template instantiation.
// For every type T, we need to define a specialization
// to select the right type for calling atomicCAS.
template <typename T>
class AtomicCasType;

template<>
class AtomicCasType<int8_t> {
 public:
  using type = unsigned short int;
  static const unsigned int mask = 0xffu;
};

template<>
class AtomicCasType<half> {
 public:
  using type = unsigned short int;
  static const unsigned int mask = 0xffffu;
};

template<>
class AtomicCasType<float> {
 public:
  using type = unsigned int;
  static const unsigned int mask = 0xffffffffu;
};

template<>
class AtomicCasType<double> {
 public:
  using type = unsigned long long int;
  static const unsigned int mask = 0xffffffffu;
};

template<>
class AtomicCasType<int> {
 public:
  using type = int;
  static const unsigned int mask = 0xffffffffu;
};

template<>
class AtomicCasType<int64_t> {
 public:
  using type = unsigned long long int;
  static const unsigned int mask = 0xffffffffu;
};

// Obtained from pytorch/aten/src/ATen/cuda/Atomic.cuh.
//
// This function compute 8-bit atomic binary operation using 32-bit atomicCAS.
// It accumulate `val` into the `address` using the `func`.
// The accumulation is atomic (i.e., thread-safe).
//
// E.g., Assume ValueType is
//  int8_t
// and BinaryFunc is
//  struct AddFunc {
//    __device__ __forceinline__ int8_t operator()(int8_t a, int8_t b) const {
//      return a + b;
//  }
// This function becomes atomic_add for int8_t.
template<typename ValueType, typename BinaryFunc>
__device__ __forceinline__ void atomic_byte_func_with_unit32_cas(ValueType* address, ValueType val, BinaryFunc func) {
    // Assert to ensure the following bit-wise manipulation is correct.
    static_assert(sizeof(ValueType) == 1 | sizeof(ValueType) == 2 | sizeof(ValueType) == 4,
      "ValueType must be 1-byte, 2-byte or 4-byte large.");
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
      old_byte = (old >> shift) & AtomicCasType<ValueType>::mask;
      // Compute new int8_t value and store it to newrawvalue.
      // Journey of a 32-bit value (cont'd):
      //
      // newrawvalue
      // ... new byte 2 ...
      auto newrawvalue = func(val, reinterpret_cast<ValueType&>(old_byte));
      // Put the new int8_t value back to 32-bit word.
      // Also ensure that bits not occupied by the int8_t value are 0s.
      //
      // Journey of a 32-bit value (cont'd):
      //
      // reinterpret_cast<uint32_t&>(newrawvalue)
      //    random values   |   random values    |   random values    | ... new byte 2 ...
      //
      // reinterpret_cast<uint32_t&>(newrawvalue) & AtomicCasType<ValueType>::mask
      //      00000000      |      00000000      |      00000000      | ... new byte 2 ...
      newval = reinterpret_cast<uint32_t&>(newrawvalue) & AtomicCasType<ValueType>::mask;
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
      newval = (old & ~(AtomicCasType<ValueType>::mask << shift)) | (newval << shift);
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
}

// It accumulates `val` into the `address` using the `func`.
// This function is thread-safe (i.e., atomic).
template<typename ValueType, typename BinaryFunc>
__device__ __forceinline__ void atomic_binary_func(ValueType* address, ValueType val, BinaryFunc func) {
  ValueType observed = *address, assumed, new_value;
  using CasType = typename AtomicCasType<ValueType>::type;
  static_assert(sizeof(ValueType) == sizeof(CasType),
    "ValueType and CasType must have the same size for calling atomicCAS.");
  auto address_as_cas_type = reinterpret_cast<CasType*>(address);
  do {
      // Record the value used to compute new value.
      assumed = observed;

      // Compute expected new value.
      new_value = func(observed, val);

      // Cast to aribitrary 2-byte type to desired integer type supported by atomicCAS.
      //                    4
      //                    8
      auto observed_as_cas_type = *reinterpret_cast<CasType*>(&observed);
      auto new_value_as_cas_type = *reinterpret_cast<CasType*>(&new_value);

      // Call atomicCAS as if the 2-byte type variables are all unsigned short int.
      //                          4                             unsigned int (or int)
      //                          8                             unsigned long long int
      auto cas_observed_as_cas_type = atomicCAS(address_as_cas_type, observed_as_cas_type, new_value_as_cas_type);

      // Cast the freshly observed value in memory back to the TwoByteType.
      observed = *reinterpret_cast<ValueType*>(&cas_observed_as_cas_type);

      // Two cases:
      // 1. compare-and-swap success
      //    a. `address` holds `new_value`
      //    b. `observed` becomes the new value after the assignment.
      //       Thus, the following `observed != new_value` is false,
      //       and the loop terminates.
      //  2. compare-and-swap fails
      //     a. `address` holds a value different from `observed`, thus,
      //        the `new_value` is stale.
      //     b. `observed` becomes the fresh value observed in `address`.
      //        Thus, the following (observed != new_value) is true,
      //        and the loop continues. In the next iteration, the
      //        `new_value` is computed again using the fresh `observed`.
  } while (observed != assumed);
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
  atomic_byte_func_with_unit32_cas(address, value, AddFunc());
}
__device__ __forceinline__ void atomic_mul(int8_t* address, int8_t value) {
  atomic_byte_func_with_unit32_cas(address, value, MulFunc());
}
__device__ __forceinline__ void atomic_max(int8_t* address, int8_t value) {
  atomic_byte_func_with_unit32_cas(address, value, MaxFunc());
}
__device__ __forceinline__ void atomic_min(int8_t* address, int8_t value) {
  atomic_byte_func_with_unit32_cas(address, value, MinFunc());
}

__device__ __forceinline__ void atomic_mul(half* address, half value) {
  atomic_byte_func_with_unit32_cas(address, value, MulFunc());
}
__device__ __forceinline__ void atomic_max(half* address, half value) {
  atomic_byte_func_with_unit32_cas(address, value, MaxFunc());
}
__device__ __forceinline__ void atomic_min(half* address, half value) {
  atomic_byte_func_with_unit32_cas(address, value, MinFunc());
}

__device__ __forceinline__ void atomic_mul(float* address, float value) {
  atomic_binary_func(address, value, MulFunc());
}
__device__ __forceinline__ void atomic_max(float* address, float value) {
  atomic_binary_func(address, value, MaxFunc());
}
__device__ __forceinline__ void atomic_min(float* address, float value) {
  atomic_binary_func(address, value, MinFunc());
}

__device__ __forceinline__ void atomic_mul(double* address, double value) {
  atomic_binary_func(address, value, MulFunc());
}
__device__ __forceinline__ void atomic_max(double* address, double value) {
  atomic_binary_func(address, value, MaxFunc());
}
__device__ __forceinline__ void atomic_min(double* address, double value) {
  atomic_binary_func(address, value, MinFunc());
}


}  // namespace rocm
}  // namespace onnxruntime
