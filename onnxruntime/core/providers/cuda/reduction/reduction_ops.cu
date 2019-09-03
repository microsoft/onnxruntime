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

//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// NVIDIA/apex is licensed under the
// BSD 3 - Clause "New" or "Revised" License
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "reduction_ops.h"

namespace onnxruntime {
namespace cuda {
namespace apex {

#define ELEMENTS_PER_ITER 4  // enables concurrency within each thread to hide latency
#define ELEMENTS_PER_THREAD 16
#define OPTIMAL_TILE_W 32
#define MAX_H_BLOCK 128
#define MAX_BLOCK_SIZE 512

bool is_apex_reduction_sum(
    const cudnnDataType_t cudnn_type,
    const cudnnReduceTensorOp_t cudnnReduceOp,
    const int m,
    const int n,
    const size_t rank,
    std::vector<int64_t> axes) {
  if (m % 2 != 0)
    return false;

  if (n % 2 != 0)
    return false;

  if (rank < 2)
    return false;

  if (cudnn_type != CUDNN_DATA_FLOAT)
    return false;

  if (cudnnReduceOp != CUDNN_REDUCE_TENSOR_ADD)
    return false;

  // Check if all but the last axis are reduced. For example, reducing
  // [N, C, H, W]-tensor to [W]-tensor can pass these two checks but reducing
  // [N, C]-tensor to [N, 1]-tensor cannot.
  if (axes.size() != rank - 1)
    return false;

  // The last reduced axis should be the second last axis. For
  // [N, C, H, W]-input, the sorted axes should be [0, 1, 2].
  std::sort(axes.begin(), axes.end());
  if (axes.back() != rank - 2)
    return false;

  return true;
}

inline int h_last_pow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return n - (n >> 1);
}

inline int div_ru(int x, int y) {
  return h_last_pow2(1 + (x - 1) / y);
}

void compute_reduction_grid_and_block(
    const int reduction_size,
    const int stride,
    const bool coop_flag,
    dim3& block,
    dim3& grid) {
  int block_x = std::min(h_last_pow2(stride), OPTIMAL_TILE_W);
  int block_y = std::min(h_last_pow2(
                             div_ru(reduction_size, ELEMENTS_PER_THREAD)),
                         MAX_BLOCK_SIZE / block_x);

  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(h_last_pow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = div_ru(stride, block_x);
  int grid_y = std::min(
      div_ru(reduction_size, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK);

  if (coop_flag) {
    // it's not worth having a grid reduction_size if the reduction_size
    // dimension is not big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template <typename T>
__device__ __forceinline__ void merge_block_vertical(
    T& mean, T* shmem_mean) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;
  shmem_mean[address_base] = mean;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto mean_new = shmem_mean[address];
      mean += mean_new;
      // last write is not necessary
      shmem_mean[address_base] = mean;
    }
  }
}

// welford kernel for c last tensor calculating mean/biased_variance/unbiased_variance
template <typename scalar_t,
          typename accscalar_t,
          typename outscalar_t,
          int PARALLEL_LOADS>
__global__ void
reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    outscalar_t* __restrict__ out_mean,
    volatile accscalar_t* staging_data,
    int* semaphores,
    const int reduction_size,
    const int stride) {
  // hide latency with concurrency
  accscalar_t x_mean[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_math[PARALLEL_LOADS];

// load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input[address_base];
      } else {
        x_math[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

// calculate mean/m2n with welford
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++)
      x_mean[j] += x_math[j];
  }

// thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    x_mean[0] += x_mean[j];
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  // block-wise reduction with shared memory (since reduction cannot be done within a warp)

  static __shared__ accscalar_t shmem_mean[MAX_BLOCK_SIZE];

  merge_block_vertical(mean_th, shmem_mean);

  // grid reduction if needed (coop launch used at the first place)
  if (gridDim.y > 1) {
    volatile accscalar_t* staging_mean = staging_data;

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
    }

    __threadfence();
    __syncthreads();  // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y - 1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      mean_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        accscalar_t mean_new = c_offset < stride ? staging_mean[address_base] : accscalar_t(0.0);
        mean_th += mean_new;
      }

      merge_block_vertical(mean_th, shmem_mean);
      if (threadIdx.y == 0 && c_offset < stride) {
        out_mean[c_offset] = static_cast<outscalar_t>(mean_th);
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      out_mean[c_offset] = static_cast<outscalar_t>(mean_th);
    }
  }
}

void reduce_sum_along_all_but_the_last_axis(
    const float* input, float* output,
    const dim3 grid, const dim3 block,
    const int reduction_size, const int stride,
    float* staging_data, int* semaphores) {
  if (grid.y > 1) {
    reduce_sum_kernel<float, float, float, ELEMENTS_PER_ITER><<<grid, block, 0, 0>>>(
        input,
        output,
        staging_data,
        semaphores,
        reduction_size,
        stride);
  } else {
    reduce_sum_kernel<float, float, float, ELEMENTS_PER_ITER><<<grid, block, 0, 0>>>(
        input,
        output,
        nullptr,
        nullptr,
        reduction_size,
        stride);
  }
}

}  // namespace apex
}  // namespace cuda
}  // namespace onnxruntime