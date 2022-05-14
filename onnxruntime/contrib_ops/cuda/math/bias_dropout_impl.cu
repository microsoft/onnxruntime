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

#include "contrib_ops/cuda/math/bias_dropout.h"

#include <algorithm>
#include <curand_kernel.h>
#include "core/providers/cuda/cu_inc/bitmask.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T, bool HasSameShapeBias, bool HasResidual, bool UseBitmask>
__global__ void BiasDropoutKernel(const int64_t N, const fast_divmod fdm_gpu_warp_size, const fast_divmod fdm_dim,
                                  const float ratio, const std::pair<uint64_t, uint64_t> seeds, const T* X_data,
                                  const T* bias_data, const T* residual_data, T* Y_data, void* mask_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = (blockDim.x * blockIdx.x + threadIdx.x) * kNumUnroll;
  CUDA_LONG step_size = gridDim.x * blockDim.x * kNumUnroll;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

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
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        float bias;
        if (HasSameShapeBias) {
          bias = float(bias_data[li]);
        } else {
          int offset = fdm_dim.mod(li);
          bias = float(bias_data[offset]);
        }

        bool mask = (&rand.x)[i] < p;
        float output_data = (float(X_data[li]) + bias) * mask * scale;
        if (HasResidual) {
          output_data += float(residual_data[li]);
        }

        Y_data[li] = T(output_data);
        if (UseBitmask) {
          thread_bitmask |= (mask << i);
        } else {
          reinterpret_cast<bool*>(mask_data)[li] = mask;
        }
      }
    }

    if (UseBitmask) {
      SetBitmask<kNumUnroll>(id, fdm_gpu_warp_size, thread_bitmask, reinterpret_cast<uint32_t*>(mask_data));
    }

    __syncthreads();
  }
}

template <typename T, bool HasSameShapeBias, bool HasResidual, bool UseBitmask>
__global__ void BiasDropoutVectorizedKernel(const int64_t N, const fast_divmod fdm_gpu_warp_size,
                                            const fast_divmod fdm_dim, const float ratio,
                                            const std::pair<uint64_t, uint64_t> seeds, const T* X_data,
                                            const T* bias_data, const T* residual_data, T* Y_data, void* mask_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = (blockDim.x * blockIdx.x + threadIdx.x) * kNumUnroll;
  CUDA_LONG step_size = gridDim.x * blockDim.x * kNumUnroll;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  float4 rand;

  // using vectorized data load/store approach when N % 4 == 0
  // since this is typical case for input shape size
  using LoadT = aligned_vector<T, kNumUnroll>;
  using MaskLoadT = aligned_vector<bool, kNumUnroll>;
  using ResidualLoadT = aligned_vector<T, kNumUnroll>;

  for (CUDA_LONG id = idx; id < N; id += step_size) {
    rand = curand_uniform4(&state);
    uint32_t thread_bitmask = 0;

    // vectorized load into storage
    T bias_vec[kNumUnroll];
    if (HasSameShapeBias) {
      LoadT* value0 = reinterpret_cast<LoadT*>(&bias_vec);
      *value0 = *reinterpret_cast<const LoadT*>(&bias_data[id]);
    }

    T src[kNumUnroll];
    LoadT* value1 = reinterpret_cast<LoadT*>(&src);
    *value1 = *reinterpret_cast<const LoadT*>(&X_data[id]);

    T residual[kNumUnroll];
    if (HasResidual) {
      ResidualLoadT* value2 = reinterpret_cast<ResidualLoadT*>(&residual);
      *value2 = *reinterpret_cast<const ResidualLoadT*>(&residual_data[id]);
    }

    T r[kNumUnroll];
    bool masks[kNumUnroll];

// actual computation
#pragma unroll
    for (int ii = 0; ii < kNumUnroll; ii++) {
      float bias;
      if (HasSameShapeBias) {
        bias = float(bias_vec[ii]);
      } else {
        int offset = fdm_dim.mod(id + ii);
        bias = float(bias_data[offset]);
      }

      bool mask = (&rand.x)[ii] < p;
      float output_data = (float(src[ii]) + bias) * mask * scale;
      if (HasResidual) {
        output_data += float(residual[ii]);
      }
      r[ii] = T(output_data);
      if (UseBitmask) {
        thread_bitmask |= (mask << ii);
      } else {
        masks[ii] = mask;
      }
    }

    // Vectorized writes for mask_data & Y_data
    *(reinterpret_cast<LoadT*>(&Y_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);
    if (UseBitmask) {
      SetBitmask<kNumUnroll>(id, fdm_gpu_warp_size, thread_bitmask, reinterpret_cast<uint32_t*>(mask_data));
    } else {
      *(reinterpret_cast<MaskLoadT*>(&reinterpret_cast<bool*>(mask_data)[id])) =
          *reinterpret_cast<MaskLoadT*>(&masks[0]);
    }

    __syncthreads();
  }
}

#define LAUNCH_BIAS_DROPOUT_KERNEL(FuncName, HasSameShapeBias, HasResidual)                     \
  FuncName<T, HasSameShapeBias, HasResidual, UseBitmask><<<grid_size, kBlockSize, 0, stream>>>( \
      N, fdm_gpu_warp_size, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data)

template <typename T, bool UseBitmask>
void BiasDropoutKernelImpl(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const fast_divmod fdm_dim,
                           const float ratio, PhiloxGenerator& generator, const T* X_data, const T* bias_data,
                           const T* residual_data, T* Y_data, void* mask_data, bool has_same_shape_bias) {
  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / kBlockSize;
  const int grid_size =
      std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, kBlockSize * kNumUnroll)));

  // Compute the number of random numbers generated by each thread, and increment philox generator offset by that
  // amount.
  const uint64_t counter_offset =
      static_cast<uint64_t>(((N - 1) / (kBlockSize * grid_size * kNumUnroll) + 1) * kNumUnroll);
  auto seeds = generator.NextPhiloxSeeds(counter_offset);
  fast_divmod fdm_gpu_warp_size(GPU_WARP_SIZE);
  if (N % kNumUnroll != 0) {
    if (has_same_shape_bias) {
      if (!residual_data) {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutKernel, true, false);
      } else {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutKernel, true, true);
      }
    } else {
      if (!residual_data) {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutKernel, false, false);
      } else {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutKernel, false, true);
      }
    }
  } else {
    if (has_same_shape_bias) {
      if (!residual_data) {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutVectorizedKernel, true, false);
      } else {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutVectorizedKernel, true, true);
      }
    } else {
      if (!residual_data) {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutVectorizedKernel, false, false);
      } else {
        LAUNCH_BIAS_DROPOUT_KERNEL(BiasDropoutVectorizedKernel, false, true);
      }
    }
  }
}

#undef LAUNCH_BIAS_DROPOUT_KERNEL

#define SPECIALIZED_BIAS_DROPOUT_IMPL(T, UseBitmask)                                                                  \
  template void BiasDropoutKernelImpl<T, UseBitmask>(                                                                 \
      const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const fast_divmod fdm_dim, const float ratio, \
      PhiloxGenerator& generator, const T* X_data, const T* bias_data, const T* residual_data, T* Y_data,             \
      void* mask_data, bool has_same_shape_bias);

#define SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED(T) \
  SPECIALIZED_BIAS_DROPOUT_IMPL(T, true)       \
  SPECIALIZED_BIAS_DROPOUT_IMPL(T, false)

SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED(float)
SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED(double)
SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED(half)
SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED(BFloat16)

#undef SPECIALIZED_BIAS_DROPOUT_IMPL_TYPED
#undef SPECIALIZED_BIAS_DROPOUT_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
