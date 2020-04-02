// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef __half half;

namespace onnxruntime {
namespace hip {

__device__ __forceinline__ void atomic_add(float *address, float value) {
    atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double *address, double value) {
  atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(half *address, half value) {
  // need to review whehter the code below is OK on AMD GPU
//   half packed_old[2];
//   half packed_new[2];
//   int* const p_packed_old = reinterpret_cast<int*>(packed_old);
//   int* const p_packed_new = reinterpret_cast<int*>(packed_new);
//   int seen_old_value = 0;
//   do {
//     packed_old[0] = *address;
//     packed_old[1] = *(address + 1);
//     packed_new[0] = __float2half(__half2float(packed_old[0]) + __half2float(value));
//     packed_new[1] = packed_old[1];
//     seen_old_value = atomicCAS(reinterpret_cast<int*>(address), *p_packed_old, *p_packed_new);
//   } while (seen_old_value != *p_packed_old);
}

}  // namespace hip
}  // namespace onnxruntime