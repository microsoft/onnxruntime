// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/scaled_silu_impl.h"

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

__device__ __forceinline__ float LoadScale(const void* data, int element_type) {
  if (element_type == kFloat) return static_cast<const float*>(data)[0];
  if (element_type == kFloat16) return __half2float(static_cast<const half*>(data)[0]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[0]);
}

__device__ __forceinline__ float Sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
  const float e = expf(value);
  return e / (1.0f + e);
}

template <typename T>
__global__ void ScaledSiLUKernel(const T* x, const void* scale, int scale_type,
                                 T* y, int64_t count, float alpha) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float scale_value = scale == nullptr ? alpha : LoadScale(scale, scale_type);
  const T z = from_float<T>(to_float<T>(x[i]) * scale_value);
  const T sigmoid = from_float<T>(Sigmoid(to_float<T>(z)));
  y[i] = from_float<T>(to_float<T>(z) * to_float<T>(sigmoid));
}

}  // namespace

template <typename T>
Status LaunchScaledSiLU(cudaStream_t stream, const T* x, const void* scale,
                        int scale_type, T* y, int64_t count, float alpha) {
  if (count == 0) return Status::OK();
  const int64_t block_count = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  ScaledSiLUKernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(x, scale, scale_type, y, count, alpha);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T) \
  template Status LaunchScaledSiLU<T>(cudaStream_t, const T*, const void*, int, T*, int64_t, float);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
