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
#include "contrib_ops/cuda/math/bias_dropout.h"

#include <curand_kernel.h>
#include <algorithm>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int UNROLL = 4;

template <typename T, bool has_same_shape_bias, bool has_residual>
__global__ void BiasDropoutKernel(
    const int64_t N,
    const fast_divmod fdm_dim,
    const float ratio,
    const std::pair<uint64_t, uint64_t> seeds,
    const T* X_data,
    const T* bias_data,
    const T* residual_data,
    T* Y_data,
    bool* mask_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  float4 rand;

  // We ensure every thread generates the same number of random numbers (by rounding
  // up the size) and at the same timestep (by syncing threads).
  // From CUDA curand documentation:
  //   The Philox_4x32_10 algorithm is closely tied to the thread and block count.
  //   Each thread computes 4 random numbers in the same time thus the most efficient
  //   use of Philox_4x32_10 is to generate a multiple of 4 times number of threads.
  for (CUDA_LONG id = idx * UNROLL; id < N; id += step_size) {
    rand = curand_uniform4(&state);

    // actual computation
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      CUDA_LONG li = id + i;
      if (li < N) {
        float bias;
        if (has_same_shape_bias) {
          bias = float(bias_data[li]);
        } else {
          int offset = fdm_dim.mod(li);
          bias = float(bias_data[offset]);
        }

        mask_data[li] = (&rand.x)[i] < p;
        float output_data = (float(X_data[li]) + bias) * mask_data[li] * scale;
        if (has_residual) {
          output_data += float(residual_data[li]);
        }

        Y_data[li] = T(output_data);
      }
    }

    __syncthreads();
  }

}


template <typename T, bool has_same_shape_bias, bool has_residual>
__global__ void BiasDropoutVectorizedKernel(
    const int64_t N,
    const fast_divmod fdm_dim,
    const float ratio,
    const std::pair<uint64_t, uint64_t> seeds,
    const T* X_data,
    const T* bias_data,
    const T* residual_data,
    T* Y_data,
    bool* mask_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG step_size = gridDim.x * blockDim.x * UNROLL;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  float4 rand;

  // using vectorized data load/store approach when N % 4 == 0
  // since this is typical case for input shape size
  using LoadT = aligned_vector<T, UNROLL>;
  using MaskLoadT = aligned_vector<bool, UNROLL>;
  using ResidualLoadT = aligned_vector<T, UNROLL>;

  for (CUDA_LONG id = idx * UNROLL; id < N; id += step_size) {
    rand = curand_uniform4(&state);

    // vectorized load into storage
    T bias_vec[UNROLL];
    if (has_same_shape_bias) {
      LoadT *value0 = reinterpret_cast<LoadT*>(&bias_vec);
      *value0 = *reinterpret_cast<const LoadT*>(&bias_data[id]);
    }

    T src[UNROLL];
    LoadT *value1 = reinterpret_cast<LoadT*>(&src);
    *value1 = *reinterpret_cast<const LoadT*>(&X_data[id]);

    T residual[UNROLL];
    if (has_residual) {
      ResidualLoadT *value2 = reinterpret_cast<ResidualLoadT*>(&residual);
      *value2 = *reinterpret_cast<const ResidualLoadT*>(&residual_data[id]);
    }

    T r[UNROLL];
    bool mask[UNROLL];

    // actual computation
    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      float bias;
      if (has_same_shape_bias) {
        bias = float(bias_vec[ii]);
      } else {
        int offset = fdm_dim.mod(id + ii);
        bias = float(bias_data[offset]);
      }

      mask[ii] = (&rand.x)[ii] < p;
      float output_data = (float(src[ii]) + bias) * mask[ii] * scale;
      if (has_residual) {
        output_data += float(residual[ii]);
      }
      r[ii] = T(output_data);
    }
    // Vectorized writes for mask_data & Y_data
    *(reinterpret_cast<LoadT*>(&Y_data[id])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&mask_data[id])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    __syncthreads();
  }

}

template <typename T>
void BiasDropoutKernelImpl(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const int64_t N,
    const fast_divmod fdm_dim,
    const float ratio,
    PhiloxGenerator& generator,
    const T* X_data,
    const T* bias_data,
    const T* residual_data,
    T* Y_data,
    bool* mask_data,
    bool has_same_shape_bias) {
  const int block_size = 256;
  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  const int grid_size = std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, block_size * UNROLL)));

  // Compute the number of random numbers generated by each thread, and increment philox generator offset by that amount.
  const uint64_t counter_offset = static_cast<uint64_t>(((N - 1) / (block_size * grid_size * UNROLL) + 1) * UNROLL);
  auto seeds = generator.NextPhiloxSeeds(counter_offset);

  if (N % UNROLL != 0) {
    if (has_same_shape_bias) {
      if (residual_data == nullptr) {
        BiasDropoutKernel<T, true, false><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      } else {
        BiasDropoutKernel<T, true, true><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      }
    } else {
      if (residual_data == nullptr) {
        BiasDropoutKernel<T, false, false><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      } else {
        BiasDropoutKernel<T, false, true><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      }
    }
  } else {
    if (has_same_shape_bias) {
      if (residual_data == nullptr) {
        BiasDropoutVectorizedKernel<T, true, false><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      } else {
        BiasDropoutVectorizedKernel<T, true, true><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      }
    } else {
      if (residual_data == nullptr) {
        BiasDropoutVectorizedKernel<T, false, false><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      } else {
        BiasDropoutVectorizedKernel<T, false, true><<<grid_size, block_size, 0, stream>>>(N, fdm_dim, ratio, seeds, X_data, bias_data, residual_data, Y_data, mask_data);
      }
    }
  }
}

#define SPECIALIZED_BIAS_DROPOUT_IMPL(T) \
  template void BiasDropoutKernelImpl(  \
      const cudaDeviceProp& prop,   \
      cudaStream_t stream,          \
      const int64_t N,              \
      const fast_divmod fdm_dim,    \
      const float ratio,            \
      PhiloxGenerator& generator,   \
      const T* X_data,              \
      const T* bias_data,           \
      const T* residual_data,       \
      T* Y_data,                    \
      bool* mask_data,              \
      bool has_same_shape_bias);


SPECIALIZED_BIAS_DROPOUT_IMPL(float)
SPECIALIZED_BIAS_DROPOUT_IMPL(double)
SPECIALIZED_BIAS_DROPOUT_IMPL(half)
SPECIALIZED_BIAS_DROPOUT_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib {
}  // namespace onnxruntime
