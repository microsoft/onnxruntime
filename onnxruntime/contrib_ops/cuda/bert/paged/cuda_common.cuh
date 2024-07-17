// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "contrib_ops/cuda/bert/paged/cuda_common.h"
#if !defined(__HIPCC__)
#include "contrib_ops/cuda/bert/paged/platform_ext.cuh"
#else
#include "contrib_ops/rocm/bert/paged/platform_ext.cuh"
#endif

#if !defined(__HIPCC__)
#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#if !defined(__HIPCC__)
#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
#define SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

namespace onnxruntime::contrib::paged {

__forceinline__ __device__ auto
lane_id() {
  return threadIdx.x % constant::WarpSize;
}

__forceinline__ __device__ auto
warp_id() {
  return threadIdx.x / constant::WarpSize;
}

#if !defined(__CUDACC__)
__forceinline__
#endif
__host__ __device__ constexpr uint32_t
ceil_log2(uint32_t n) {
  return n <= 1 ? 0 : 1 + ceil_log2((n + 1) / 2);
}

__forceinline__ __host__ __device__ constexpr uint32_t
next_power_of_two(uint32_t n) {
  return uint32_t(1) << ceil_log2(n);
}

}  // namespace onnxruntime::contrib::paged
