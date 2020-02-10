// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

namespace onnxruntime {
namespace hip {

__device__ __forceinline__ void atomic_add(float *address, float value) {
    atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double *address, double value) {
#if __hip_ARCH__ < 600
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

__device__ __forceinline__ void atomic_add(half *address, half value) {
#if __hip_ARCH__ < 700
  half packed_old[2];
  half packed_new[2];
  int* const p_packed_old = reinterpret_cast<int*>(packed_old);
  int* const p_packed_new = reinterpret_cast<int*>(packed_new);
  int seen_old_value = 0;
  do {
    packed_old[0] = *address;
    packed_old[1] = *(address + 1);
    packed_new[0] = half(float(packed_old[0]) + float(value));
    packed_new[1] = packed_old[1];
    seen_old_value = atomicCAS(reinterpret_cast<int*>(address), *p_packed_old, *p_packed_new);
  } while (seen_old_value != *p_packed_old);
#else
  atomicAdd(address, value);
#endif
}

}  // namespace hip
}  // namespace onnxruntime