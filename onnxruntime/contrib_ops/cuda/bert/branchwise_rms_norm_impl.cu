// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

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

template <typename T>
__global__ void MixedScaleBranchwiseRMSNormKernel(const T* x, const void* scale, int scale_type,
                                                  T* y, int64_t groups, int branches, int hidden,
                                                  bool shared_scale, float epsilon) {
  const int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group >= groups) {
    return;
  }
  const int64_t offset = group * hidden;
  float sum_sq = 0.0f;
  for (int h = 0; h < hidden; ++h) {
    const float value = to_float<T>(x[offset + h]);
    sum_sq += value * value;
  }
  const float inv_rms = rsqrtf(sum_sq / hidden + epsilon);
  for (int h = 0; h < hidden; ++h) {
    const int64_t scale_index = shared_scale ? h : (group % branches) * hidden + h;
    y[offset + h] = from_float<T>(
        to_float<T>(x[offset + h]) * inv_rms * LoadScale(scale, scale_index, scale_type));
  }
}

}  // namespace

template <typename T>
Status LaunchMixedScaleBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                         int scale_type, T* y, int64_t groups, int branches,
                                         int hidden, bool shared_scale, float epsilon) {
  const int64_t block_count = (groups - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  MixedScaleBranchwiseRMSNormKernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(
      x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
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
