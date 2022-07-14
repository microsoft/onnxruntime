// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is adapted from microsoft/DeepSpeed
// type_shim.h

/* Taken from NVIDIA/apex commit 855808f3fc268e9715d613f3c2e56469d8c986d8 */
#include <ATen/ATen.h>

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)                   \
    switch (TYPE) {                                                              \
        case at::ScalarType::Double: {                                           \
            using scalar_t_##LEVEL = double;                                     \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Float: {                                            \
            using scalar_t_##LEVEL = float;                                      \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Half: {                                             \
            using scalar_t_##LEVEL = at::Half;                                   \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        default: AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }

template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes(T* x, T val, int lanes = 1,
                                                     bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
#if defined(CUDA_VERSION) && CUDA_VERSION >= 9000
    for (int i = 16; i >= lanes; i >>= 1) final = final + __shfl_down_sync(0xffffffff, final, i);
#else
    for (int i = 16; i >= lanes; i >>= 1) final = final + __shfl_down(final, i);
#endif
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}
