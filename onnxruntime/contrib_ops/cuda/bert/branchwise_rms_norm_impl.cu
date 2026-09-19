// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kFloat = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
constexpr int kFloat16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;

__device__ __forceinline__ float LoadScale(const void* data, int64_t index, int element_type) {
  if (element_type == kFloat) {
    return static_cast<const float*>(data)[index];
  }
  if (element_type == kFloat16) {
    return __half2float(static_cast<const half*>(data)[index]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

template <typename T, int ILP>
__global__ void MixedScaleBranchwiseRMSNormKernel(const T* x, const void* scale, int scale_type,
                                                  T* y, int64_t groups, int branches, int hidden,
                                                  bool shared_scale, float epsilon) {
  const int64_t group = blockIdx.x;
  if (group >= groups) {
    return;
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  using VecT = onnxruntime::cuda::aligned_vector<T, ILP>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float inv_rms;

  const int64_t offset = group * hidden;
  float sum_sq = 0.0f;
  for (int h = threadIdx.x * ILP; h < hidden; h += blockDim.x * ILP) {
    const VecT values = *reinterpret_cast<const VecT*>(x + offset + h);
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      const float value = to_float<T>(values.val[i]);
      sum_sq += value * value;
    }
  }

  const float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
  if (threadIdx.x == 0) {
    inv_rms = rsqrtf(total_sum_sq / hidden + epsilon);
  }
  __syncthreads();

  const int64_t branch_offset = (group % branches) * hidden;
  for (int h = threadIdx.x * ILP; h < hidden; h += blockDim.x * ILP) {
    const VecT values = *reinterpret_cast<const VecT*>(x + offset + h);
    VecT outputs;
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      const int scale_index = h + i;
      const int64_t scale_offset = shared_scale ? scale_index : branch_offset + scale_index;
      const float weight = scale == nullptr ? 1.0f : LoadScale(scale, scale_offset, scale_type);
      outputs.val[i] = from_float<T>(to_float<T>(values.val[i]) * inv_rms * weight);
    }
    *reinterpret_cast<VecT*>(y + offset + h) = outputs;
  }
}

}  // namespace

template <typename T>
Status LaunchMixedScaleBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                         int scale_type, T* y, int64_t groups, int branches,
                                         int hidden, bool shared_scale, float epsilon) {
  ORT_RETURN_IF_NOT(groups <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  if constexpr (sizeof(T) == 2) {
    if (hidden % 2 == 0) {
      MixedScaleBranchwiseRMSNormKernel<T, 2><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
    } else {
      MixedScaleBranchwiseRMSNormKernel<T, 1><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
    }
  } else {
    MixedScaleBranchwiseRMSNormKernel<T, 1><<<static_cast<int>(groups), kThreads, 0, stream>>>(
        x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                                        \
  template Status LaunchMixedScaleBranchwiseRMSNorm<T>(cudaStream_t, const T*, const void*, int, T*, int64_t, \
                                                       int, int, bool, float);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
