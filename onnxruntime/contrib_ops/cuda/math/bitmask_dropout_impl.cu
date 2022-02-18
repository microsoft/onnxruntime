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

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/bitmask_dropout.h"

#include <curand_kernel.h>
#include <algorithm>

constexpr unsigned FULL_MASK = 0XFFFFFFFF;
constexpr int UNROLL = 4;
constexpr int WARP_SIZE = 32;

namespace onnxruntime {
namespace contrib {
namespace cuda {

/**
 * This kernel MUST be built with an unroll factor that evenly divides the number of threads in a warp.
 *
 * In addition, this kernel MUST be launched with a number of threads in a thread block which is evenly
 * divisible by the number of threads in a warp (32).
 *
 * TODO: Investigate vectorization of data output. Vectorization won't help us too much with writing out the
 * bitmask (and in fact, it may slow us down a bit). We want to match the same output values as the standard
 * Dropout kernel, and to do so we have to iterate over generated random values in the same order.
 *
 * For an unroll factor of 4, we take the following approach (for threads in the first warp, that is):
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
 * After each thread shifts its bits into the right location, we take advantage of the previously generated thread mask
 * to perform a bitwise-or reduction with "__reduce_or_sync". This broadcasts the reduced value to all threads. Finally,
 * we just choose a single thread (in our case, we choose the thread with 0 shift, but any thread from 0-7 would work for
 * the 0-7 group).
 *
 * Finally, this value is written out to the correct value in the output mask, and threads go on to generate the next
 * value.
 *
 * Keep in mind that the "__match_any_sync" and "__reduce_or_sync" must not be conditionally called, as all threads in
 * the warp (that haven't already exited) must reach these function calls an equal number of times, or else UB is invoked.
 *
 * We conditionally update the local thread's mask (with the "li < N" check), but all active threads always collaborate
 * on the reduced value.
 */
template <typename T>
__global__ void BitmaskDropoutKernel(
    const int64_t N,
    const float ratio,
    const std::pair<uint64_t, uint64_t> seeds,
    const T* X_data,
    T* Y_data,
    uint32_t* mask_data) {
  static_assert(WARP_SIZE % UNROLL == 0, "number of threads in warp must be evenly divisble by unroll factor");

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = ((blockDim.x * blockIdx.x) + threadIdx.x) * UNROLL;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx / UNROLL, seeds.second, &state);

  float4 rand;

  // We ensure every thread generates the same number of random numbers (by rounding
  // up the size) and at the same timestep (by syncing threads).
  // From CUDA curand documentation:
  //   The Philox_4x32_10 algorithm is closely tied to the thread and block count.
  //   Each thread computes 4 random numbers in the same time thus the most efficient
  //   use of Philox_4x32_10 is to generate a multiple of 4 times number of threads.
  for (CUDA_LONG id = idx; id < N; id += step_size) {
    rand = curand_uniform4(&state);

    uint32_t thread_bitmask = 0;

// actual computation
#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      CUDA_LONG li = id + i;
      if (li < N) {
        bool mask = (&rand.x)[i] < p;
        Y_data[li] = T(float(X_data[li]) * mask * scale);
        thread_bitmask |= (mask << i);
      }
    }

    CUDA_LONG bitmask_idx = id / WARP_SIZE;
    // All thread which intend to write to the same output index will have the same thread mask.
    unsigned int thread_mask = __match_any_sync(FULL_MASK, bitmask_idx);

    CUDA_LONG bitmask_shift = id % WARP_SIZE;
    // All threads with the same thread mask (threads which intend to write to the same output index) collaborate
    // on a bitwise-or reduction.
    // TODO: Investigate backwards compatibility with older CUDA architectures.
    uint32_t full_bitmask = __reduce_or_sync(thread_mask, thread_bitmask << bitmask_shift);

    // Choose a single from the "thread mask" group to perform the output write.
    if (bitmask_shift == 0) {
      mask_data[bitmask_idx] = full_bitmask;
    }

    __syncthreads();
  }
}

template <typename T>
void BitmaskDropoutKernelImpl(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const int64_t N,
    const float ratio,
    PhiloxGenerator& generator,
    const T* X_data,
    T* Y_data,
    uint32_t* mask_data) {
  constexpr int block_size = 256;
  static_assert(block_size % WARP_SIZE == 0, "number of threads in block must be evenly divisible by warp size");

  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  const int grid_size = std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, block_size * UNROLL)));

  // Compute the number of random numbers generated by each thread, and increment philox generator offset by that amount.
  const uint64_t counter_offset = static_cast<uint64_t>(((N - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL);
  auto seeds = generator.NextPhiloxSeeds(counter_offset);

  BitmaskDropoutKernel<T><<<grid_size, block_size, 0, stream>>>(N, ratio, seeds, X_data, Y_data, mask_data);
}

#define SPECIALIZED_BITMASK_DROPOUT_IMPL(T) \
  template void BitmaskDropoutKernelImpl(   \
      const cudaDeviceProp& prop,           \
      cudaStream_t stream,                  \
      const int64_t N,                      \
      const float ratio,                    \
      PhiloxGenerator& generator,           \
      const T* X_data,                      \
      T* Y_data,                            \
      uint32_t* mask_data);

SPECIALIZED_BITMASK_DROPOUT_IMPL(float)
SPECIALIZED_BITMASK_DROPOUT_IMPL(double)
SPECIALIZED_BITMASK_DROPOUT_IMPL(half)
SPECIALIZED_BITMASK_DROPOUT_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime