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
#pragma once
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

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

template <typename T, int ILP>
inline __device__ void UpdateSum(const T* src, int64_t offset, float& sum, float& sum_sq) {
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    const float val = static_cast<float>(input_v.val[i]);
    sum += val;
    sum_sq += val * val;
  }
}

template <>
inline __device__ void UpdateSum<half, 2>(const half* src, int64_t offset, float& sum, float& sum_sq) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);

  float2 f2 = __half22float2(h2);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum<float, 2>(const float* src, int64_t offset, float& sum, float& sum_sq) {
  // Fetch two channels per thread.
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

// Sum for SkipGroupNorm: add_out[offset] = src[offset] + skip[skip_offset] + bias[bias_offset]
template <typename T, int32_t ILP>
inline __device__ void AddSkipBias(T* add_out, const T* src, const T* skip, const T* bias,
                                   int64_t offset, int64_t skip_offset, int64_t bias_offset, float& sum, float& sum_sq) {
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);
  const VecT skip_v = *reinterpret_cast<const VecT*>(skip + skip_offset);
  const VecT bias_v = *reinterpret_cast<const VecT*>(bias + bias_offset);
  VecT output_v = *reinterpret_cast<VecT*>(add_out + offset);

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    output_v.val[i] = input_v.val[i] + skip_v.val[i] + bias_v.val[i];
    const float val = static_cast<float>(output_v.val[i]);
    sum += val;
    sum_sq += val * val;
  }
  *(reinterpret_cast<VecT*>(add_out + offset)) = output_v;
}

template <>
inline __device__ void AddSkipBias<half, 2>(half* add_out, const half* src, const half* skip, const half* bias,
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
inline __device__ void AddSkipBias<float, 2>(float* add_out, const float* src, const float* skip, const float* bias,
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
template <typename T, int32_t ILP>
inline __device__ void AddSkip(T* add_out, const T* src, const T* skip,
                               int64_t offset, int64_t skip_offset, float& sum, float& sum_sq) {
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);
  const VecT skip_v = *reinterpret_cast<const VecT*>(skip + skip_offset);
  VecT output_v = *reinterpret_cast<VecT*>(add_out + offset);

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    output_v.val[i] = input_v.val[i] + skip_v.val[i];
    const float val = static_cast<float>(output_v.val[i]);
    sum += val;
    sum_sq += val * val;
  }
  *(reinterpret_cast<VecT*>(add_out + offset)) = output_v;
}

template <>
inline __device__ void AddSkip<half, 2>(half* add_out, const half* src, const half* skip,
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
inline __device__ void AddSkip<float, 2>(float* add_out, const float* src, const float* skip,
                                         int64_t offset, int64_t skip_offset, float& sum, float& sum_sq) {
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);
  float2 s = *reinterpret_cast<float2 const*>(&skip[skip_offset]);
  f2.x += s.x;
  f2.y += s.y;
  *reinterpret_cast<float2*>(&add_out[offset]) = f2;
  sum += f2.x + f2.y;
  sum_sq += f2.x * f2.x + f2.y * f2.y;
}

template <typename T, int32_t THREADS_PER_BLOCK, int32_t ILP>
__global__ void GroupNormNHWCSumKernel(T* skip_workspace, float* group_sum_buffer, const T* src, const T* skip, const T* bias,
                                       int32_t channels_per_block, int32_t hw_per_block, int32_t hw, int32_t hwc, int32_t c,
                                       int32_t channels_per_group, int32_t groups, int32_t groups_per_block, bool broadcast_skip) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, THREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Allocate shared memory for the groups. We could reduce the amount of shared memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * channels_per_block + threadIdx.x * ILP;

  if (ci >= c || threadIdx.x * ILP >= channels_per_block) {
    return;
  }

  // The first activation loaded by that block.
  int32_t hw_begin = blockIdx.y * hw_per_block;
  // The last activation loaded by that block.
  int32_t hw_end = min(hw_begin + hw_per_block, hw);

  // The sums.
  float sum = 0.F;
  float sum_sq = 0.F;

  // Iterate over the activations to compute the sums.
  int64_t offset = static_cast<int64_t>(ni) * hwc + static_cast<int64_t>(hw_begin) * c + ci;
  if (skip != nullptr) {
    // SkipGroupNorm: skip is (n, h, w, c) or (n, 1, 1, c) or (n, c),  bias is (c), and add_out is (n, h, w, c)
    const int64_t bias_offset = static_cast<int64_t>(ci);
    T* add_out = skip_workspace;
    if (broadcast_skip) {
      const int64_t skip_offset = static_cast<int64_t>(ni) * c + ci;

      if (bias != nullptr) {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
          AddSkipBias<T, ILP>(add_out, src, skip, bias, offset, skip_offset, bias_offset, sum, sum_sq);
        }
      } else {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
          AddSkip<T, ILP>(add_out, src, skip, offset, skip_offset, sum, sum_sq);
        }
      }
    } else {
      if (bias != nullptr) {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
          AddSkipBias<T, ILP>(add_out, src, skip, bias, offset, offset, bias_offset, sum, sum_sq);
        }
      } else {
        for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
          AddSkip<T, ILP>(add_out, src, skip, offset, offset, sum, sum_sq);
        }
      }
    }
  } else {  // GroupNorm
    for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
      UpdateSum<T, ILP>(src, offset, sum, sum_sq);
    }
  }

  // The group index relative to the first group within the same block.
  int32_t gi = threadIdx.x * ILP / channels_per_group;
  // The channel in the group.
  int32_t cj = ci % channels_per_group;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sum_sq};

  // Do the segmented scan. InclusiveScan is not deterministic.
  GroupSums out;
  BlockScan(temp_storage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced stores later).
  // For each group, only the last thread of that group is picked to save sum to shared memory.
  if (cj == channels_per_group - ILP) {
    smem[gi] = make_float2(out.sum, out.sum_sq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= groups_per_block) {
    return;
  }

  // The global group index.
  // Use neighboring threads for coalesced write.
  int32_t gj = blockIdx.x * groups_per_block + threadIdx.x;

  if (gj < groups) {
    float2 sums = smem[threadIdx.x];
    const int index = (2 * ni) * groups + gj;
    atomicAdd(&group_sum_buffer[index], sums.x);
    atomicAdd(&group_sum_buffer[index + groups], sums.y);
  }
}

template <typename T, int32_t ILP>
__device__ void computeGroupNormVec(const T* src, T* dst, int64_t offset, float mean, float inv_std_dev,
                                    const float* gamma_v, const float* beta_v, bool silu) {
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);
  VecT output_v;

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    float val = static_cast<float>(input_v.val[i]);
    val = (val - mean) * inv_std_dev;
    val = gamma_v[i] * val + beta_v[i];

    if (silu) {
      val = val * sigmoid(val);
    }
    output_v.val[i] = static_cast<T>(val);
  }
  *(reinterpret_cast<VecT*>(dst + offset)) = output_v;
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

template <typename T, int32_t ILP>
__device__ void ComputeGroupNormKernel(const T* input, T* dst, int64_t offset, float mean, float inv_std_dev,
                                       const float* gamma, const float* beta, bool use_silu, int32_t c, int32_t ci, int32_t hw_begin, int32_t hw_end) {
  using VecF = onnxruntime::cuda::aligned_vector<float, ILP>;

  const VecF gamma_v = *reinterpret_cast<const VecF*>(gamma + ci);
  const VecF beta_v = *reinterpret_cast<const VecF*>(beta + ci);
  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
    // Fetch ILP channels per thread.
    computeGroupNormVec<T, ILP>(input, dst, offset, mean, inv_std_dev, gamma_v.val, beta_v.val, use_silu);
  }
}

template <>
__device__ void ComputeGroupNormKernel<float, 2>(const float* input, float* dst, int64_t offset, float mean, float inv_std_dev,
                                                 const float* gamma, const float* beta, bool use_silu, int32_t c, int32_t ci, int32_t hw_begin, int32_t hw_end) {
  // Load gamma/beta. Fetch two per thread.
  float2 gamma_f2 = *reinterpret_cast<float2 const*>(&gamma[ci]);
  float2 beta_f2 = *reinterpret_cast<float2 const*>(&beta[ci]);
  for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
    ComputeGroupNorm<float>(input, dst, offset, mean, inv_std_dev, gamma_f2, beta_f2, use_silu);
  }
}

template <>
__device__ void ComputeGroupNormKernel<half, 2>(const half* input, half* dst, int64_t offset, float mean, float inv_std_dev,
                                                const float* gamma, const float* beta, bool use_silu, int32_t c, int32_t ci, int32_t hw_begin, int32_t hw_end) {
  // Load gamma/beta. Fetch two per thread.
  float2 gamma_f2 = *reinterpret_cast<float2 const*>(&gamma[ci]);
  float2 beta_f2 = *reinterpret_cast<float2 const*>(&beta[ci]);
  for (int32_t hwi = hw_begin; hwi < hw_end; ++hwi, offset += c) {
    ComputeGroupNorm<half>(input, dst, offset, mean, inv_std_dev, gamma_f2, beta_f2, use_silu);
  }
}

template <typename T, int32_t ILP>
__global__ void GroupNormNHWCScaleKernel(T* dst, const T* src, const T* skip, const float* gamma, const float* beta,
                                         const T* skip_workspace, const float* group_sum_buffer, float epsilon,
                                         int32_t c, int32_t channels_per_block, int32_t channels_per_group,
                                         int32_t groups, int32_t hwc, float inv_hw_channels_per_group,
                                         int32_t hw, int32_t hw_per_block, bool use_silu) {
  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * channels_per_block + threadIdx.x * ILP;
  if (ci >= c || threadIdx.x * ILP >= channels_per_block) {
    return;
  }

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The group that thread works on.
  int32_t gi = ci / channels_per_group;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sum_sq = 0.F;
  if (gi < groups) {
    const int index = (2 * ni) * groups + gi;
    sum = group_sum_buffer[index];
    sum_sq = group_sum_buffer[index + groups];
  }

  // Compute the mean.
  float mean = sum * inv_hw_channels_per_group;
  // Compute the variance.
  float var = sum_sq * inv_hw_channels_per_group - (mean * mean);
  // Compute the inverse of the stddev.
  float inv_std_dev = rsqrtf(var + epsilon);

  int32_t hw_begin = blockIdx.y * hw_per_block;
  int32_t hw_end = min(hw_begin + hw_per_block, hw);

  const T* input = (skip != nullptr) ? skip_workspace : src;
  int64_t offset = static_cast<int64_t>(ni) * hwc + static_cast<int64_t>(hw_begin) * c + ci;
  ComputeGroupNormKernel<T, ILP>(input, dst, offset, mean, inv_std_dev, gamma, beta, use_silu, c, ci, hw_begin, hw_end);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
