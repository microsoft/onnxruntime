// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <cstdint>

namespace onnxruntime::contrib::paged {

using mutex_t = uint32_t;

namespace detail {

__forceinline__ __device__ void
nanosleep(uint32_t duration_ns) {
#if !defined(__HIPCC__)
  __nanosleep(duration_ns);
#else
  asm volatile(R"asm(
      S_NOP 0
      S_NOP 0
      S_NOP 0
      S_NOP 0
      S_NOP 0
      S_NOP 0
      S_NOP 0
      S_NOP 0
    )asm" ::);
#endif
}
}  // namespace detail

// TODO: move to task.cuh
__forceinline__ __device__ void
backoff(uint32_t duration_ns = 32) {
  detail::nanosleep(duration_ns);
}

__forceinline__ __device__ void
mutex_lock(mutex_t& mutex) {
  uint32_t ns = 8;
  while (atomicCAS(&mutex, 0, 1) == 1) {
    backoff(ns);
    if (ns < 256) {
      ns *= 2;
    }
  }
}

__forceinline__ __device__ bool
try_mutex_lock(mutex_t& mutex) {
  return atomicCAS(&mutex, 0, 1) == 0;
}

__forceinline__ __device__ void
mutex_unlock(mutex_t& mutex) {
  atomicExch(&mutex, 0);
}

struct lock_guard {
  mutex_t& mutex_;
  __forceinline__ __device__
  lock_guard(mutex_t& mutex)
      : mutex_{mutex} {
    mutex_lock(mutex_);
  }

  __forceinline__ __device__
  ~lock_guard() {
    mutex_unlock(mutex_);
  }
};

}  // namespace onnxruntime::contrib::paged
