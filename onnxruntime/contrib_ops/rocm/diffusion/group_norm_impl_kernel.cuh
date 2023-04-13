// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The ROCm kernel is modified from TensorRT 8.5.
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

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sumSq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

template <typename T, typename U, int ILP>
inline __device__ void UpdateSum(const T* src, int64_t offset, U& sum, U& sumSq) {
  using VecT = onnxruntime::rocm::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    const U val = static_cast<U>(input_v.val[i]);
    sum += val;
    sumSq += val * val;
  }
}

template <typename T, int ThreadsPerBlock, int ILP>
__global__ void groupNormNHWCSumKernel(const T* src, float* redBuffer, int32_t cPerBlock, int32_t hwPerBlock, int32_t hw,
                                       int32_t hwc, int32_t c, int32_t cPerGroup, int32_t groups, int32_t groupsPerBlock) {
  // The object in charge of doing the sums for the different blocks.
  typedef hipcub::BlockScan<GroupSums, ThreadsPerBlock> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[ThreadsPerBlock];

  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (ILP channels per thread).
  int32_t ci = blockIdx.x * cPerBlock + threadIdx.x * ILP;

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + hwPerBlock, hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  // Iterate over the activations to compute the sums.
  if (ci < c) {
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
      // The offset.
      int64_t offset = static_cast<int64_t>(ni) * hwc + static_cast<int64_t>(hwi) * c + ci;
      UpdateSum<T, float, ILP>(src, offset, sum, sumSq);
    }
  }

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = threadIdx.x * ILP / cPerGroup;
  int32_t cj = threadIdx.x * ILP - cPerGroup * gi;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced
  // stores later).
  if (cj == cPerGroup - ILP) {  // ILP channels per thread
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int32_t gj = blockIdx.x * groupsPerBlock + threadIdx.x;

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= groupsPerBlock || gj >= groups) {
    return;
  }

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];

  // Store to global memory.
  atomicAdd(&redBuffer[(2 * ni + 0) * groups + gj], sums.x);
  atomicAdd(&redBuffer[(2 * ni + 1) * groups + gj], sums.y);
}

template <typename T, typename U, int32_t ILP>
__device__ void computeGroupNorm(const T* src, T* dst, int64_t offset, U mean, U invStdDev,
                                 const U* gamma_v, const U* beta_v, bool swish) {
  using VecT = onnxruntime::rocm::aligned_vector<T, ILP>;
  const VecT input_v = *reinterpret_cast<const VecT*>(src + offset);
  VecT output_v;

#pragma unroll
  for (int i = 0; i < ILP; i++) {
    U val = static_cast<U>(input_v.val[i]);
    val = (val - mean) * invStdDev;
    val = gamma_v[i] * val + beta_v[i];

    if (swish) {
      val = val * sigmoid(val);
    }
    output_v.val[i] = static_cast<T>(val);
  }
  *(reinterpret_cast<VecT*>(dst + offset)) = output_v;
}

template <typename T, int ThreadsPerBlock, int ILP>
__global__ void groupNormNHWCScaleKernel(T* dst, const T* src, const float* gamma, const float* beta, const float* redBuffer, float epsilon, int32_t c, int32_t cPerBlock,
                                         int32_t cPerGroup, int32_t groups, int32_t hwc, float invHWC, int32_t hw, int32_t hwPerBlock, bool withSwish) {
  // The channel loaded by that thread (ILP channels per thread for F16x2).
  int32_t ci = blockIdx.x * cPerBlock + threadIdx.x * ILP;
  if (ci >= c) {
    return;
  }

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < groups) {
    sum = redBuffer[(2 * ni + 0) * groups + gi];
    sumSq = redBuffer[(2 * ni + 1) * groups + gi];
  }

  using VecF = onnxruntime::rocm::aligned_vector<float, ILP>;

  const VecF gamma_v = *reinterpret_cast<const VecF*>(gamma + ci);
  const VecF beta_v = *reinterpret_cast<const VecF*>(beta + ci);

  // Compute the mean.
  float mean = sum * invHWC;
  // Compute the variance.
  float var = sumSq * invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = var <= 0.F ? 1.F : rsqrtf(var + epsilon);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + hwPerBlock, hw);

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)ni * hwc + hwi * c + ci;

    // Fetch ILP channels per thread.
    computeGroupNorm<T, float, ILP>(src, dst, offset, mean, invStdDev, gamma_v.val, beta_v.val, withSwish);
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
