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

#include "core/providers/cuda/nn/dropout_impl.h"

#include <curand_kernel.h>
#include <algorithm>
#include "core/providers/cuda/cu_inc/bitmask.cuh"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T, bool UseBitmask>
__global__ void DropoutKernel(const CUDA_LONG N, const CUDA_LONG mask_element_count, const int step_size,
                              const int steps_per_thread, const fast_divmod fdm_bits_per_element, const float ratio,
                              const std::pair<uint64_t, uint64_t> seeds, const T* X_data, T* Y_data, void* mask_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  float4 rand;

  // We ensure every thread generates the same number of random numbers (by rounding
  // up the size) and at the same timestep (by syncing threads).
  // From CUDA curand documentation:
  //   The Philox_4x32_10 algorithm is closely tied to the thread and block count.
  //   Each thread computes 4 random numbers in the same time thus the most efficient
  //   use of Philox_4x32_10 is to generate a multiple of 4 times number of threads.
  for (int i = 0; i < steps_per_thread; ++i) {
    CUDA_LONG id = idx * kNumUnroll + i * step_size;
    rand = curand_uniform4(&state);
    BitmaskElementType thread_bitmask = 0;

// actual computation
#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        bool mask = (&rand.x)[i] < p;
        Y_data[li] = static_cast<T>(static_cast<float>(X_data[li]) * mask * scale);
        if (UseBitmask) {
          thread_bitmask |= (mask << i);
        } else {
          reinterpret_cast<bool*>(mask_data)[li] = mask;
        }
      }
    }

    if (UseBitmask) {
      SetBitmask<kNumUnroll>(id, mask_element_count, fdm_bits_per_element, thread_bitmask,
                             reinterpret_cast<BitmaskElementType*>(mask_data));
    }

    __syncthreads();
  }
}

template <typename T, bool UseBitmask>
__global__ void DropoutVectorizedKernel(const CUDA_LONG N, const CUDA_LONG mask_element_count, const int step_size,
                                        const int steps_per_thread, const fast_divmod fdm_bits_per_element,
                                        const float ratio, const std::pair<uint64_t, uint64_t> seeds, const T* X_data,
                                        T* Y_data, void* mask_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  float4 rand;

  // using vectorized data load/store approach when N % 4 == 0 since this is
  // typical case for input shape size
  using LoadT = aligned_vector<T, kNumUnroll>;
  using MaskLoadT = aligned_vector<bool, kNumUnroll>;

  for (int i = 0; i < steps_per_thread; ++i) {
    CUDA_LONG id = idx * kNumUnroll + i * step_size;
    rand = curand_uniform4(&state);
    BitmaskElementType thread_bitmask = 0;

    if (id < N) {
      // vectorized load into storage
      T src[kNumUnroll];
      LoadT* value = reinterpret_cast<LoadT*>(&src);
      *value = *reinterpret_cast<const LoadT*>(&X_data[id]);

      T r[kNumUnroll];
      bool masks[kNumUnroll];

// actual computation
#pragma unroll
      for (int ii = 0; ii < kNumUnroll; ++ii) {
        bool mask = (&rand.x)[ii] < p;
        r[ii] = static_cast<T>(static_cast<float>(src[ii]) * mask * scale);
        if (UseBitmask) {
          thread_bitmask |= (mask << ii);
        } else {
          masks[ii] = mask;
        }
      }
      // Vectorized writes for mask_data & Y_data
      *(reinterpret_cast<LoadT*>(&Y_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);
      if (!UseBitmask) {
        *(reinterpret_cast<MaskLoadT*>(&reinterpret_cast<bool*>(mask_data)[id])) =
            *reinterpret_cast<MaskLoadT*>(&masks[0]);
      }
    }

    if (UseBitmask) {
      SetBitmask<kNumUnroll>(id, mask_element_count, fdm_bits_per_element, thread_bitmask,
                             reinterpret_cast<BitmaskElementType*>(mask_data));
    }

    __syncthreads();
  }
}

#define LAUNCH_DROPOUT_KERNEL(FuncName, UseBitmask)                                                       \
  FuncName<T, UseBitmask><<<grid_size, kBlockSize, 0, stream>>>(                                          \
      static_cast<CUDA_LONG>(N), static_cast<CUDA_LONG>(mask_element_count), step_size, steps_per_thread, \
      fdm_bits_per_element, ratio, seeds, X_data, Y_data, mask_data)

#define HANDLE_DROPOUT_USE_BITMASK(FuncName) \
  if (use_bitmask) {                         \
    LAUNCH_DROPOUT_KERNEL(FuncName, true);   \
  } else {                                   \
    LAUNCH_DROPOUT_KERNEL(FuncName, false);  \
  }

template <typename T>
void DropoutKernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N,
                       const int64_t mask_element_count, const float ratio, PhiloxGenerator& generator, const T* X_data,
                       T* Y_data, void* mask_data, bool use_bitmask) {
  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / kBlockSize;
  const int grid_size =
      std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, kBlockSize * kNumUnroll)));

  // Compute the number of random numbers generated by each thread, and increment philox generator offset by that
  // amount.
  const int step_size = kBlockSize * grid_size * kNumUnroll;
  const int steps_per_thread = static_cast<int>(CeilDiv(N, step_size));
  auto seeds = generator.NextPhiloxSeeds(static_cast<uint64_t>(steps_per_thread * kNumUnroll));

  fast_divmod fdm_bits_per_element(kNumBitsPerBitmaskElement);
  if (N % kNumUnroll != 0) {
    HANDLE_DROPOUT_USE_BITMASK(DropoutKernel);
  } else {
    HANDLE_DROPOUT_USE_BITMASK(DropoutVectorizedKernel);
  }
}

#undef HANDLE_DROPOUT_USE_BITMASK
#undef LAUNCH_DROPOUT_KERNEL

#define SPECIALIZED_DROPOUT_IMPL(T)                                                                                   \
  template void DropoutKernelImpl<T>(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N,                \
                                     const int64_t mask_element_count, const float ratio, PhiloxGenerator& generator, \
                                     const T* X_data, T* Y_data, void* mask_data, bool use_bitmask);

SPECIALIZED_DROPOUT_IMPL(float)
SPECIALIZED_DROPOUT_IMPL(double)
SPECIALIZED_DROPOUT_IMPL(half)
SPECIALIZED_DROPOUT_IMPL(BFloat16)

#undef SPECIALIZED_DROPOUT_IMPL

}  // namespace cuda
}  // namespace onnxruntime
