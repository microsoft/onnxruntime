// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_impl.h"

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
constexpr int kBFloat16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;

__device__ __forceinline__ float Sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float e = expf(value);
  return e / (1.0f + e);
}

__device__ __forceinline__ int64_t GateIndex(int layout, int64_t row,
                                             int branch, int feature,
                                             int branches, int hidden) {
  if (layout == 0) {
    return 0;
  }
  if (layout == 1 || layout == 2) {
    return row * branches + branch;
  }
  return (row * branches + branch) * hidden + feature;
}

__device__ __forceinline__ float LoadAux(const void* data, int64_t index,
                                         int element_type) {
  if (element_type == kFloat) {
    return static_cast<const float*>(data)[index];
  }
  if (element_type == kFloat16) {
    return __half2float(static_cast<const half*>(data)[index]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

template <typename T>
__global__ void BranchwiseRMSNormKernel(const T* x, const void* scale,
                                        int scale_type, T* y, int64_t groups,
                                        int branches, int hidden,
                                        bool shared_scale, float epsilon) {
  const int64_t group =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
    const int64_t scale_index =
        shared_scale ? h : (group % branches) * hidden + h;
    const float weight = scale == nullptr ? 1.0f : LoadAux(scale, scale_index, scale_type);
    y[offset + h] = from_float<T>(to_float<T>(x[offset + h]) * inv_rms * weight);
  }
}

template <typename T>
__global__ void ScaledSiLUKernel(const T* x, const void* scale, int scale_type,
                                 T* y, int64_t count, float alpha) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const float scale_value = scale == nullptr ? alpha : LoadAux(scale, 0, scale_type);
  const T z = from_float<T>(to_float<T>(x[i]) * scale_value);
  const T sigmoid = from_float<T>(Sigmoid(to_float<T>(z)));
  y[i] = from_float<T>(to_float<T>(z) * to_float<T>(sigmoid));
}

template <typename T>
__global__ void PreMixKernel(const T* streams, const void* pre_mix, int mix_type,
                             T* y, int64_t count, int branches, int hidden,
                             int gate_layout, float reduction_scale) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const int64_t row = i / hidden;
  const int h = static_cast<int>(i % hidden);
  float sum = 0.0f;
  for (int c = 0; c < branches; ++c) {
    const int64_t x_index = (row * branches + c) * hidden + h;
    sum += to_float<T>(streams[x_index]) *
           LoadAux(pre_mix, GateIndex(gate_layout, row, c, h, branches, hidden), mix_type);
  }
  y[i] = from_float<T>(sum * reduction_scale);
}

template <typename T>
__global__ void PostMixKernel(const T* streams, const T* branch_output,
                              const void* post_mix, const void* stream_mix,
                              int mix_type, T* y, int64_t count, int branches,
                              int hidden, int gate_layout) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const int h = static_cast<int>(i % hidden);
  const int c = static_cast<int>((i / hidden) % branches);
  const int64_t row = i / (static_cast<int64_t>(hidden) * branches);
  float value = stream_mix == nullptr ? to_float<T>(streams[i]) : 0.0f;
  if (stream_mix != nullptr) {
    for (int source = 0; source < branches; ++source) {
      value += LoadAux(
                   stream_mix, (row * branches + source) * branches + c, mix_type) *
               to_float<T>(
                   streams[(row * branches + source) * hidden + h]);
    }
  }
  value += LoadAux(
               post_mix, GateIndex(gate_layout, row, c, h, branches, hidden), mix_type) *
           to_float<T>(branch_output[row * hidden + h]);
  y[i] = from_float<T>(value);
}

Status GetBlocks(int64_t count, int& blocks) {
  const int64_t block_count = count == 0 ? 0 : (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(),
                    "CUDA launch requires too many blocks");
  blocks = static_cast<int>(block_count);
  return Status::OK();
}

}  // namespace

template <typename T>
Status LaunchBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                               int scale_type, T* y, int64_t groups, int branches,
                               int hidden, bool shared_scale, float epsilon) {
  if (groups == 0) return Status::OK();
  int blocks;
  ORT_RETURN_IF_ERROR(GetBlocks(groups, blocks));
  BranchwiseRMSNormKernel<<<blocks, kThreads, 0, stream>>>(
      x, scale, scale_type, y, groups, branches, hidden, shared_scale, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchScaledSiLU(cudaStream_t stream, const T* x, const void* scale,
                        int scale_type, T* y, int64_t count, float alpha) {
  if (count == 0) return Status::OK();
  int blocks;
  ORT_RETURN_IF_ERROR(GetBlocks(count, blocks));
  ScaledSiLUKernel<<<blocks, kThreads, 0, stream>>>(x, scale, scale_type, y, count, alpha);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchHyperConnectionPreMix(cudaStream_t stream, const T* streams,
                                   const void* pre_mix, int mix_type, T* y,
                                   int64_t count, int branches, int hidden,
                                   int gate_layout, float reduction_scale) {
  if (count == 0) return Status::OK();
  int blocks;
  ORT_RETURN_IF_ERROR(GetBlocks(count, blocks));
  PreMixKernel<<<blocks, kThreads, 0, stream>>>(
      streams, pre_mix, mix_type, y, count, branches, hidden, gate_layout,
      reduction_scale);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchHyperConnectionPostMix(
    cudaStream_t stream, const T* streams, const T* branch_output,
    const void* post_mix, const void* stream_mix, int mix_type, T* y,
    int64_t count, int branches, int hidden, int gate_layout) {
  if (count == 0) return Status::OK();
  int blocks;
  ORT_RETURN_IF_ERROR(GetBlocks(count, blocks));
  PostMixKernel<<<blocks, kThreads, 0, stream>>>(
      streams, branch_output, post_mix, stream_mix, mix_type, y, count, branches,
      hidden, gate_layout);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                        \
  template Status LaunchBranchwiseRMSNorm<T>(                                  \
      cudaStream_t, const T*, const void*, int, T*, int64_t, int, int, bool,   \
      float);                                                                  \
  template Status LaunchScaledSiLU<T>(cudaStream_t, const T*, const void*, int, \
                                      T*, int64_t, float);                      \
  template Status LaunchHyperConnectionPreMix<T>(                              \
      cudaStream_t, const T*, const void*, int, T*, int64_t, int, int, int,    \
      float);                                                                  \
  template Status LaunchHyperConnectionPostMix<T>(                             \
      cudaStream_t, const T*, const T*, const void*, const void*, int, T*,      \
      int64_t, int, int, int);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
