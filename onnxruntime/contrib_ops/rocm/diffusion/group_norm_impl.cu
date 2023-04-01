#include "hip/hip_runtime.h"
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

// The ROCM kernel is hipified from CUDA kernel.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_impl.h"
#include "contrib_ops/rocm/transformers/dump_rocm_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

static inline int32_t divUp(int32_t m, int32_t n) {
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

template <typename T>
struct GroupNormNHWCParams {
  // The output buffer. Layout NHWC.
  T* dst;
  // The input buffer. Layout NHWC.
  T const* src;
  // The gamma scaling factor.
  float const* gamma;
  // The beta term to add in GN.
  float const* beta;
  // The temporary buffer to do the global parallel reduction. Size:
  // BLOCKS_PER_BATCH x C x 2.
  float* redBuffer;

  // The number of instances in the batch.
  int32_t n;
  // The height and width of each activation map.
  int32_t h;
  int32_t w;
  // The number of channels.
  int32_t c;
  // The number of groups.
  int32_t groups;
  // Do we apply the Swish activation function?
  bool withSwish;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (h * w) and the number of
  // activations per block.
  int32_t hw;
  int32_t hwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock;
  int32_t cPerGroup;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hwc in floats (to compute mean/var).
  float invHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;
};

template <typename T, typename U, int32_t ILP>
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

template <typename T, int32_t tTHREADS_PER_BLOCK, int32_t ILP>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams<T> params) {
  // The object in charge of doing the sums for the different blocks.
  typedef hipcub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[tTHREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (ILP channels per thread).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * ILP;

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  // Iterate over the activations to compute the sums.
  if (ci < params.c) {
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
      // The offset.
      int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;
      UpdateSum<T, float, ILP>(params.src, offset, sum, sumSq);
    }
  }

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = threadIdx.x * ILP / params.cPerGroup;
  int32_t cj = threadIdx.x * ILP - params.cPerGroup * gi;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced
  // stores later).
  if (cj == params.cPerGroup - ILP) {  // ILP channels per thread
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups) {
    return;
  }

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];

  // Store to global memory.
  atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
  atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

template <typename T>
void groupNormNHWCSum(GroupNormNHWCParams<T> const& params, hipStream_t stream) {
  // Make sure the values are as we expect.
  ORT_ENFORCE(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(params.cPerBlock % params.cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCSumKernel<T, 256, 2><<<grid, 256, 0, stream>>>(params);
      break;
    case 480:
      groupNormNHWCSumKernel<T, 256, 2><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      groupNormNHWCSumKernel<T, 128, 2><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCSumKernel<T, 64, 2><<<grid, 64, 0, stream>>>(params);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
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

template <typename T, int32_t tTHREADS_PER_BLOCK, int32_t ILP>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams<T> params) {
  // The channel loaded by that thread (ILP channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * ILP;
  if (ci >= params.c) {
    return;
  }

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < params.groups) {
    sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
  }

  using VecF = onnxruntime::rocm::aligned_vector<float, ILP>;

  const VecF gamma_v = *reinterpret_cast<const VecF*>(params.gamma + ci);
  const VecF beta_v = *reinterpret_cast<const VecF*>(params.beta + ci);

  // Compute the mean.
  float mean = sum * params.invHWC;
  // Compute the variance.
  float var = sumSq * params.invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)ni * params.hwc + hwi * params.c + ci;

    // Fetch ILP channels per thread.
    computeGroupNorm<T, float, ILP>(params.src, params.dst, offset, mean, invStdDev, gamma_v.val, beta_v.val, params.withSwish);
  }
}

template <typename T>
void groupNormNHWCScale(GroupNormNHWCParams<T> const& params, hipStream_t stream) {
  // Make sure the dimensions are aligned with what we expect.
  ORT_ENFORCE(params.c % params.cPerBlock == 0);
  // Make sure a group does not span multiple blocks.
  ORT_ENFORCE(params.cPerBlock % params.cPerGroup == 0);

  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCScaleKernel<T, 256, 2><<<grid, 256, 0, stream>>>(params);
      break;
    case 480:
      groupNormNHWCScaleKernel<T, 256, 2><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      groupNormNHWCScaleKernel<T, 128, 2><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCScaleKernel<T, 64, 2><<<grid, 64, 0, stream>>>(params);
      break;
    default:
      ORT_NOT_IMPLEMENTED("Not implemented");
  }
}

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <typename T>
Status LaunchGroupNormKernel(
    hipStream_t stream,
    T* output,
    const T* input,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_swish_activation) {
  if (batch_size > static_cast<int>(kMaxGroupNormBatchSize)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only support batch_size <= 32. Got", batch_size);
  }

  if (num_groups != static_cast<int>(kGroupNormNumberOfGroups)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only num_groups=32 is supported. Got", num_groups);
  }

  GroupNormNHWCParams<T> params;
  int32_t cPerBlock = 320;
  int32_t maxBlocksPerHW = 1024;
  switch (num_channels) {
    case 960:
    case 1920:
      cPerBlock = 480;
      break;
    case 512:
    case 256:
      cPerBlock = 256;
      break;
    case 128:
      cPerBlock = 128;
      break;
    default:
      cPerBlock = 320;
  }

  params.withSwish = use_swish_activation;
  params.dst = output;
  params.src = input;
  params.gamma = gamma;
  params.beta = beta;
  params.redBuffer = reinterpret_cast<float*>(workspace);
  params.n = batch_size;
  params.h = height;
  params.w = width;
  params.c = num_channels;
  params.groups = num_groups;
  params.hw = params.h * params.w;
  const int32_t blocksPerHW = findMaxDivisor(params.hw, maxBlocksPerHW);
  params.hwPerBlock = divUp(params.hw, blocksPerHW);
  params.cPerBlock = cPerBlock;
  params.cPerGroup = params.c / params.groups;
  params.hwc = params.hw * params.c;
  params.invHWC = 1.F / (float)(params.hw * params.cPerGroup);
  params.groupsPerBlock = cPerBlock / params.cPerGroup;

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("input", input, batch_size, num_channels, height * width);
  DUMP_TENSOR("gamma", gamma, 1, num_channels);
  DUMP_TENSOR("beta", beta, 1, num_channels);
  HIP_RETURN_IF_ERROR(hipMemsetAsync(params.redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(), stream));
  groupNormNHWCSum<T>(params, stream);
  DUMP_TENSOR("workspace", params.redBuffer, batch_size, num_groups, 2);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  groupNormNHWCScale<T>(params, stream);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  DUMP_TENSOR("output", output, batch_size, num_channels, height * width);
  return Status::OK();
}

template Status LaunchGroupNormKernel<half>(hipStream_t stream, half* output,
                                            const half* input, const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool swish);

template Status LaunchGroupNormKernel<float>(hipStream_t stream, float* output,
                                             const float* input, const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool swish);
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
