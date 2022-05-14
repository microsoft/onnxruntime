// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

/**
 * These functions MUST be called with an unroll factor that evenly divides the number of threads in a warp (32).
 * In addition, this kernel MUST be launched with a number of threads in a thread block which is evenly
 * divisible by the number of threads in a warp (32).
 *
 * Take unroll factor of 4 as example, we take the following approach (for threads in the first warp, that is):
 *
 * Thread 0 generates output booleans 0-3
 * Thread 1 generates output booleans 4-7
 * ...
 * Thread 7 generates output booleans 28-31
 *
 * These threads all agree on the same thread mask by determining which output bitmask index they want to write to.
 * Threads 0-7 will generate the same thread mask (for ouput index 0), threads 8-15 will generate the same thread mask
 * (for output index 1), and so on.
 *
 * After (partially before) agreeing upon which threads will collaborate to write out a single index,
 * each thread generates 4 random values, and shifts them into the right location in the output uint32_t.
 * For instance:
 *
 * Thread 0 will perform a shift of 0
 * Thread 1 will perform a shift of 4
 * Thread 2 will perform a shift of 8
 * ...
 *
 * For index 0, this gives us the following composition of random bits (number represents which thread generated it):
 *
 * 77776666555544443333222211110000
 *
 * After each thread shifts its bits into the right location, we broadcast the reduced value to all threads. Finally,
 * we just choose a single thread (in our case, we choose the thread with 0 shift, but any thread from 0-7 would work
 * for the 0-7 group).
 *
 * Keep in mind that this must not be conditionally called, as all threads in the warp (that haven't already exited)
 * must reach these function calls an equal number of times, or else UB is invoked.
 *
 * We conditionally update the local thread's mask (with the "li < N" check), but all active threads always collaborate
 * on the reduced value.
 */

namespace onnxruntime {
namespace cuda {

template <int NumUnroll>
__device__ __forceinline__ void SetBitmask(CUDA_LONG id, const fast_divmod gpu_warp_size_fdm, uint32_t thread_bitmask,
                                           uint32_t* mask_data) {
  int bitmask_idx, bitmask_shift;
  gpu_warp_size_fdm.divmod(id, bitmask_idx, bitmask_shift);
  uint32_t bitmask = (thread_bitmask << bitmask_shift);
#if defined(USE_CUDA) && __CUDA_ARCH__ >= 800
  // All thread which intend to write to the same output index will have the same thread mask.
  uint32_t thread_mask = __match_any_sync(0xFFFFFFFF, bitmask_idx);
  // All threads with the same thread mask (threads which intend to write to the same output index) collaborate
  // on a bitwise-or reduction.
  bitmask = __reduce_or_sync(thread_mask, bitmask);
#else
#pragma unroll
  for (int stride = GPU_WARP_SIZE / (NumUnroll * 2); stride > 0; stride /= 2) {
    bitmask |= WARP_SHFL_DOWN(bitmask, stride);
  }
#endif

  // Choose a single from the "thread mask" group to perform the output write.
  if (bitmask_shift == 0) {
    mask_data[bitmask_idx] = bitmask;
  }
}

template <int NumUnroll>
__device__ __forceinline__ void GetMasks(CUDA_LONG id, const fast_divmod gpu_warp_size_fdm, const uint32_t* mask_data,
                                         bool* mask_result) {
  int bitmask_idx, bitmask_shift;
  gpu_warp_size_fdm.divmod(id, bitmask_idx, bitmask_shift);
  uint32_t shifted_mask = mask_data[bitmask_idx] >> bitmask_shift;
#pragma unroll
  for (int i = 0; i < NumUnroll; i++) {
    mask_result[i] = (shifted_mask & (1 << i)) != 0;
  }
}

}  // namespace cuda
}  // namespace onnxruntime
