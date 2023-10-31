/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The CUDA kernel is modified from GroupNorm plugin of TensorRT 8.5
// Modifications: heuristic channels per block; support epsilon; support skip and bias; update coding style.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/diffusion/group_norm_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// TODO: Similar to SkipLayerNorm kernel, read/write up to 8 channels at same time.
constexpr static int32_t CHANNELS_PER_THREAD = 2;

constexpr static int kSizes[] = {128, 256, 320, 384, 512};
constexpr static size_t kNumOfSizes = sizeof(kSizes) / sizeof(kSizes[0]);
constexpr static int kMaxSize = kSizes[kNumOfSizes - 1];

int NextSize(int x) {
  for (size_t i = 0; i < kNumOfSizes; ++i) {
    if (x <= kSizes[i]) {
      return kSizes[i];
    }
  }

  return x;
}
}  // namespace

static inline int32_t DivUp(int32_t m, int32_t n) {
  return (m + n - 1) / n;
}

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sum_sq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sum_sq = b.flag ? b.sum_sq : (a.sum_sq + b.sum_sq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

template <typename T>
struct GroupNormNHWCParams {
  // The output buffer. Shape is (n, h, w, c).
  T* dst;

  // Optional output of element-wise add result of src, skip and bias. Shape is (n, h, w, c).
  T* add_out;

  // The input buffer. Shape is (n, h, w, c).
  T const* src;

  // Optional input buffer for skip tensor. Shape is (n, h, w, c) or (n, 1, 1, c) or (n, c).
  T const* skip;

  // Optional input buffer for bias tensor. Shape is (c).
  T const* bias;

  // The gamma scaling factor.
  float const* gamma;

  // The beta term to add in GN.
  float const* beta;

  // The temporary buffer to do the global parallel reduction. Shape is (n, 2, g), where g is number of groups.
  float* group_sum_buffer;

  // The number of instances in the batch.
  int32_t n;

  // The height and width of each activation map.
  int32_t h;
  int32_t w;

  // Number of channels.
  int32_t c;

  // Number of groups.
  int32_t groups;

  // Do we apply the SiLU activation function?
  bool use_silu;

  // Precomputed values and parameters to control the execution of the kernels.

  // Number of activations per instance (h * w)
  int32_t hw;

  // Number of activations per block
  int32_t hw_per_block;

  // Number of channels per block in the C dimension.
  int32_t channels_per_block;

  // Number of channels per group in the C dimension.
  int32_t channels_per_group;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hw*channels_per_group to compute mean of a group.
  float inv_hw_channels_per_group;
  // The precomputed number of groups per block.
  int32_t groups_per_block;

  // Number of threads per block
  int32_t threads_per_block;

  // Epsilon to get stable variance in normalization.
  float epsilon;

  // Whether skip need broadcast. True if shape of skip is (N, C) or (N, 1, 1, C); False otherwise.
  bool broadcast_skip;

  // For SkipGroupNorm, it points to the intermediate result of adding skip and bias.
  T* skip_workspace;
};

template <typename T>
inline __device__ void UpdateSum(const T* src, int64_t offset, float& sum, float& sum_sq);

template <>
inline __device__ void UpdateSum(const half* src, int64_t offset, float& sum, float& sum_sq) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);

  float2 f2 = __half22float2(h2);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum(const float* src, int64_t offset, float& sum, float& sum_sq) {
  // Fetch two channels per thread.
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

// Sum for SkipGroupNorm: add_out[offset] = src[offset] + skip[skip_offset] + bias[bias_offset]
template <typename T>
inline __device__ void AddSkipBias(T* add_out, const T* src, const T* skip, const T* bias,
                                   int64_t offset, int64_t skip_offset, int64_t bias_offset, float& sum, float& sum_sq);

template <>
inline __device__ void AddSkipBias(half* add_out, const half* src, const half* skip, const half* bias,
                                   int64_t offset, int64_t skip_offset, int64_t bias_offset, float& sum, float& sum_sq) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);
  __half2 s = *reinterpret_cast<__half2 const*>(&skip[skip_offset]);
  __half2 b = *reinterpret_cast<__half2 const*>(&bias[bias_offset]);
  h2 = h2 + b;
  h2 = h2 + s;

  *reinterpret_cast<__half2*>(&add_out[offset]) = h2;

  float2 f2 = __half22float2(h2);
  sum += f2.x + f2.y;
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void AddSkipBias(float* add_out, const float* src, const float* skip, const float* bias,
                                   int64_t offset, int64_t skip_offset, int64_t bias_offset, float& sum, float& sum_sq) {
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);
  float2 s = *reinterpret_cast<float2 const*>(&skip[skip_offset]);
  float2 b = *reinterpret_cast<float2 const*>(&bias[bias_offset]);
  f2.x += s.x + b.x;
  f2.y += s.y + b.y;

  *reinterpret_cast<float2*>(&add_out[offset]) = f2;

  sum += f2.x + f2.y;
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

// Sum for SkipGroupNorm without bias: add_out[offset] = src[offset] + skip[skip_offset]
template <typename T>
inline __device__ void AddSkip(T* add_out, const T* src, const T* skip,
                               int64_t offset, int64_t skip_offset, float& sum, float& sum_sq);

template <>
inline __device__ void AddSkip(half* add_out, const half* src, const half* skip,
                               int64_t offset, int64_t skip_offset, float& sum, float& sum_sq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);
  __half2 s = *reinterpret_cast<__half2 const*>(&skip[skip_offset]);
  h2 = h2 + s;

  *reinterpret_cast<__half2*>(&add_out[offset]) = h2;

  float2 f2 = __half22float2(h2);
  sum += f2.x + f2.y;
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void AddSkip(float* add_out, const float* src, const float* skip,
                               int64_t offset, int64_t skip_offset, float& sum, float& sum_sq) {
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);
  float2 s = *reinterpret_cast<float2 const*>(&skip[skip_offset]);
  f2.x += s.x;
  f2.y += s.y;
  *reinterpret_cast<float2*>(&add_out[offset]) = f2;
  sum += f2.x + f2.y;
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <typename T, int32_t THREADS_PER_BLOCK>
__global__ void GroupNormNHWCSumKernel(GroupNormNHWCParams<T> params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, THREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Allocate shared memory for the groups. We could reduce the amount of shared memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * params.channels_per_block + threadIdx.x * CHANNELS_PER_THREAD;

  if (ci >= params.c || threadIdx.x * CHANNELS_PER_THREAD >= params.channels_per_block) {
    return;
  }

  // The first activation loaded by that block.
  int32_t hw_begin = blockIdx.y * params.hw_per_block;
  // The last activation loaded by that block.
  int32_t hw_end = min(hw_begin + params.hw_per_block, params.hw);

  // The sums.
  float sum = 0.F;
  float sum_sq = 0.F;

  // Iterate over the activations to compute the sums.
  int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hw_begin) * params.c + ci;
  if (params.skip != nullptr) {
    // SkipGroupNorm: skip is (n, h, w, c) or (n, 1, 1, c) or (n, c),  bias is (c), and add_out is (n, h, w, c)
    const int64_t bias_offset = static_cast<int64_t>(ci);
    T* add_out = params.skip_workspace;
    if (params.broadcast_skip) {
      const int64_t skip_offset = static_cast<int64_t>(ni) * params.c + ci;

      if (params.bias != nullptr) {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
          AddSkipBias(add_out, params.src, params.skip, params.bias, offset, skip_offset, bias_offset, sum, sum_sq);
        }
      } else {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
          AddSkip(add_out, params.src, params.skip, offset, skip_offset, sum, sum_sq);
        }
      }
    } else {
      if (params.bias != nullptr) {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
          AddSkipBias(add_out, params.src, params.skip, params.bias, offset, offset, bias_offset, sum, sum_sq);
        }
      } else {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
          AddSkip(add_out, params.src, params.skip, offset, offset, sum, sum_sq);
        }
      }
    }
  } else {  // GroupNorm
    for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
      UpdateSum(params.src, offset, sum, sum_sq);
    }
  }

  // The group index relative to the first group within the same block.
  int32_t gi = threadIdx.x * CHANNELS_PER_THREAD / params.channels_per_group;
  // The channel in the group.
  int32_t cj = ci % params.channels_per_group;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sum_sq};

  // Do the segmented scan. InclusiveScan is not deterministic.
  GroupSums out;
  BlockScan(temp_storage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced stores later).
  // For each group, only the last thread of that group is picked to save sum to shared memory.
  if (cj == params.channels_per_group - CHANNELS_PER_THREAD) {
    smem[gi] = make_float2(out.sum, out.sum_sq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= params.groups_per_block) {
    return;
  }

  // The global group index.
  // Use neighboring threads for coalesced write.
  int32_t gj = blockIdx.x * params.groups_per_block + threadIdx.x;

  if (gj < params.groups) {
    float2 sums = smem[threadIdx.x];
    const int index = (2 * ni) * params.groups + gj;
    atomicAdd(&params.group_sum_buffer[index], sums.x);
    atomicAdd(&params.group_sum_buffer[index + params.groups], sums.y);
  }
}

template <typename T>
void GroupNormNHWCSum(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params.c, params.channels_per_block);

  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params.hw, params.hw_per_block);

  // The number of instances.
  grid.z = params.n;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params.threads_per_block) {
    case 256:
      GroupNormNHWCSumKernel<T, 256><<<grid, 256, 0, stream>>>(params);
      break;
    case 192:
      GroupNormNHWCSumKernel<T, 192><<<grid, 192, 0, stream>>>(params);
      break;
    case 160:
      GroupNormNHWCSumKernel<T, 160><<<grid, 160, 0, stream>>>(params);
      break;
    case 128:
      GroupNormNHWCSumKernel<T, 128><<<grid, 128, 0, stream>>>(params);
      break;
    case 64:
      GroupNormNHWCSumKernel<T, 64><<<grid, 64, 0, stream>>>(params);
      break;
  }
}

template <typename T>
__device__ void ComputeGroupNorm(const T* src, T* dst, int64_t offset, float mean, float inv_std_dev,
                                 float2& gamma_f2, float2& beta_f2, bool silu);

template <>
__device__ void ComputeGroupNorm(const half* src, half* dst, int64_t offset, float mean, float inv_std_dev,
                                 float2& gamma_f2, float2& beta_f2, bool silu) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);

  // Extract the two half values.
  float2 f2 = __half22float2(h2);

  // Normalize the channels.
  f2.x = (f2.x - mean) * inv_std_dev;
  f2.y = (f2.y - mean) * inv_std_dev;

  // Scale by gamma and add beta.
  f2.x = gamma_f2.x * f2.x + beta_f2.x;
  f2.y = gamma_f2.y * f2.y + beta_f2.y;

  // Apply SiLU activation if needed.
  if (silu) {
    f2.x = f2.x * sigmoid(f2.x);
    f2.y = f2.y * sigmoid(f2.y);
  }

  *reinterpret_cast<__half2*>(&dst[offset]) = __float22half2_rn(f2);
}

template <>
__device__ void ComputeGroupNorm(const float* src, float* dst, int64_t offset, float mean, float inv_std_dev,
                                 float2& gamma_f2, float2& beta_f2, bool silu) {
  // Fetch two channels per thread.
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);

  // Normalize the channels.
  f2.x = (f2.x - mean) * inv_std_dev;
  f2.y = (f2.y - mean) * inv_std_dev;

  // Scale by gamma and add beta.
  f2.x = gamma_f2.x * f2.x + beta_f2.x;
  f2.y = gamma_f2.y * f2.y + beta_f2.y;

  // Apply SiLU activation if needed.
  if (silu) {
    f2.x = f2.x * sigmoid(f2.x);
    f2.y = f2.y * sigmoid(f2.y);
  }

  *reinterpret_cast<float2*>(&dst[offset]) = f2;
}

template <typename T>
__global__ void GroupNormNHWCScaleKernel(GroupNormNHWCParams<T> params) {
  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * params.channels_per_block + threadIdx.x * CHANNELS_PER_THREAD;
  if (ci >= params.c || threadIdx.x * CHANNELS_PER_THREAD >= params.channels_per_block) {
    return;
  }

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The group that thread works on.
  int32_t gi = ci / params.channels_per_group;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sum_sq = 0.F;
  if (gi < params.groups) {
    const int index = (2 * ni) * params.groups + gi;
    sum = params.group_sum_buffer[index];
    sum_sq = params.group_sum_buffer[index + params.groups];
  }

  // Load gamma/beta. Fetch two per thread.
  float2 gamma_f2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
  float2 beta_f2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);

  // Compute the mean.
  float mean = sum * params.inv_hw_channels_per_group;
  // Compute the variance.
  float var = sum_sq * params.inv_hw_channels_per_group - (mean * mean);
  // Compute the inverse of the stddev.
  float inv_std_dev = rsqrtf(var + params.epsilon);

  int32_t hw_begin = blockIdx.y * params.hw_per_block;
  int32_t hw_end = min(hw_begin + params.hw_per_block, params.hw);

  const T* input = (params.skip != nullptr) ? params.skip_workspace : params.src;
  int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hw_begin) * params.c + ci;
  for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += params.c) {
    ComputeGroupNorm<T>(input, params.dst, offset, mean, inv_std_dev, gamma_f2, beta_f2, params.use_silu);
  }
}

template <typename T>
void GroupNormNHWCScale(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = DivUp(params.c, params.channels_per_block);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = DivUp(params.hw, params.hw_per_block);
  // The number of instances.
  grid.z = params.n;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params.threads_per_block) {
    case 256:
      GroupNormNHWCScaleKernel<T><<<grid, 256, 0, stream>>>(params);
      break;
    case 192:
      GroupNormNHWCScaleKernel<T><<<grid, 192, 0, stream>>>(params);
      break;
    case 160:
      GroupNormNHWCScaleKernel<T><<<grid, 160, 0, stream>>>(params);
      break;
    case 128:
      GroupNormNHWCScaleKernel<T><<<grid, 128, 0, stream>>>(params);
      break;
    case 64:
      GroupNormNHWCScaleKernel<T><<<grid, 64, 0, stream>>>(params);
      break;
  }
}

int32_t FindMaxDivisor(int32_t n, int32_t max_allowed_divisor) {
  int32_t max_divisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > max_divisor && divisor1 < max_allowed_divisor) {
        max_divisor = divisor1;
      }
      if (divisor2 > max_divisor && divisor2 < max_allowed_divisor) {
        max_divisor = divisor2;
      }
    }
  }
  return max_divisor;
}

// Find proper channels per block based on a cost function: The cost is number of channels corresponding to
// extra threads allocated but no channels assigned to them to work on. If cost is zero, every thread has
// work to do so it is ideal case.
int FindChannelsPerBlock(int num_channels, int channels_per_group) {
  int min_cost = -1;
  int best_candidate = -1;
  for (size_t i = kNumOfSizes; i > 0; --i) {
    if (kSizes[i - 1] < channels_per_group) {
      break;
    }

    int channels_per_block = kSizes[i - 1] / channels_per_group * channels_per_group;
    int blocks = (num_channels + channels_per_block - 1) / channels_per_block;
    int cost = blocks * kSizes[i - 1] - num_channels;
    if (cost == 0) {
      return channels_per_block;
    }

    if (min_cost == -1 || cost < min_cost) {
      min_cost = cost;
      best_candidate = channels_per_block;
    }
  }

  return best_candidate;
}

int GetChannelsPerBlock(int num_channels, int num_groups) {
  int32_t channels_per_group = num_channels / num_groups;
  int32_t channels_per_block = channels_per_group;
  if (channels_per_group < kMaxSize / 2) {
    channels_per_block = FindChannelsPerBlock(num_channels, channels_per_group);
  }
  return channels_per_block;
}

template <typename T>
Status LaunchGroupNormKernel(
    cudaStream_t stream,
    T* output,
    T* add_out,
    const T* input,
    const T* skip,
    const T* bias,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_silu,
    bool broadcast_skip,
    int channels_per_block) {
  GroupNormNHWCParams<T> params;

  int32_t channels_per_group = num_channels / num_groups;
  // channels_per_block is computed in PrePack.
  // If the gamma is not initializer, channels_per_block might be zero after PrePack. In that happens, compute it here.
  if (channels_per_block < channels_per_group) {
    channels_per_block = GetChannelsPerBlock(num_channels, num_groups);
  }

  // TODO: Update the kernel to support CHANNELS_PER_THREAD==1 and other corner cases
  if (channels_per_block % channels_per_group != 0 ||
      channels_per_block > kMaxSize ||
      (channels_per_group % CHANNELS_PER_THREAD != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GroupNorm in CUDA does not support the input: n=", batch_size,
                           " h=", height,
                           " w=", width,
                           " c=", num_channels,
                           " groups=", num_groups);
  }

  params.use_silu = use_silu;
  params.dst = output;
  params.add_out = add_out;
  params.src = input;
  params.skip = skip;
  params.bias = bias;
  params.gamma = gamma;
  params.beta = beta;
  params.group_sum_buffer = reinterpret_cast<float*>(workspace);
  params.n = batch_size;
  params.h = height;
  params.w = width;
  params.c = num_channels;
  params.groups = num_groups;
  params.hw = params.h * params.w;

  // This will allocate as many blocks as possible to partition HW.
  // For Stable Diffusion, latent hw is 4K ~ 16K. This will allocate 1024 blocks, and each handles 4~16 hw.
  // TODO: tune this logic to find proper blocks when hw is small.
  constexpr int32_t max_blocks_per_hw = 1024;
  const int32_t blocks_per_hw = FindMaxDivisor(params.hw, max_blocks_per_hw);
  params.hw_per_block = DivUp(params.hw, blocks_per_hw);

  params.channels_per_block = channels_per_block;
  params.channels_per_group = channels_per_group;
  params.hwc = params.hw * params.c;
  params.inv_hw_channels_per_group = 1.F / (float)(params.hw * params.channels_per_group);
  params.groups_per_block = channels_per_block / params.channels_per_group;
  params.epsilon = epsilon;
  params.broadcast_skip = broadcast_skip;

  // Workspace for SkipGroupNorm to store intermediate results of src+skip+bias.
  params.skip_workspace = (params.add_out != nullptr) ? params.add_out : params.dst;

  params.threads_per_block = NextSize(channels_per_block) / CHANNELS_PER_THREAD;

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
      params.group_sum_buffer, 0, GetGroupNormWorkspaceSizeInBytes(batch_size, num_groups), stream));

  GroupNormNHWCSum<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("workspace", params.group_sum_buffer, batch_size, 2, num_groups);

  GroupNormNHWCScale<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return Status::OK();
}

template Status LaunchGroupNormKernel<half>(cudaStream_t stream, half* output, half* add_out,
                                            const half* input, const half* skip, const half* bias,
                                            const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool silu,
                                            bool broadcast_skip, int channels_per_block);

template Status LaunchGroupNormKernel<float>(cudaStream_t stream, float* output, float* add_out,
                                             const float* input, const float* skip, const float* bias,
                                             const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool silu,
                                             bool broadcast_skip, int channels_per_block);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
