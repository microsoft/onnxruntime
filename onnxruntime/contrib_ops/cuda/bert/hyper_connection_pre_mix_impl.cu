// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_pre_mix_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kFloat = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
constexpr int kFloat16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;

__device__ __forceinline__ float LoadMix(const void* data, int64_t index, int element_type) {
  if (element_type == kFloat) return static_cast<const float*>(data)[index];
  if (element_type == kFloat16) return __half2float(static_cast<const half*>(data)[index]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

template <typename T>
__device__ __forceinline__ float MixSigmoid(float value, int mode) {
  if (mode == 1) {
    const float e = expf(-fabsf(value));
    return to_float<T>(from_float<T>(value >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e)));
  }
  const float e = to_float<T>(from_float<T>(expf(value > 0 ? -value : value)));
  const float denominator = to_float<T>(from_float<T>(1.0f + e));
  const float reciprocal = to_float<T>(from_float<T>(1.0f / denominator));
  return value > 0 ? reciprocal : to_float<T>(from_float<T>(1.0f - reciprocal));
}

__device__ __forceinline__ int64_t GateIndex(int layout, int64_t row, int branch,
                                             int feature, int branches, int hidden) {
  if (layout == 0) return 0;
  if (layout == 1 || layout == 2) return row * branches + branch;
  return (row * branches + branch) * hidden + feature;
}

template <typename T>
__global__ void PreMixKernel(const T* streams, const void* pre_mix, int mix_type,
                             T* y, int64_t count, int branches, int hidden,
                             int gate_layout, float reduction_scale, int sigmoid_mode) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t row = i / hidden;
  const int h = static_cast<int>(i % hidden);
  float sum = 0.0f;
  for (int c = 0; c < branches; ++c) {
    const int64_t x_index = (row * branches + c) * hidden + h;
    float gate = LoadMix(pre_mix, GateIndex(gate_layout, row, c, h, branches, hidden), mix_type);
    if (sigmoid_mode != 0) {
      if (mix_type == kFloat) gate = MixSigmoid<float>(gate, sigmoid_mode);
      else if (mix_type == kFloat16) gate = MixSigmoid<half>(gate, sigmoid_mode);
      else gate = MixSigmoid<__nv_bfloat16>(gate, sigmoid_mode);
    }
    sum += to_float<T>(streams[x_index]) * gate;
  }
  y[i] = from_float<T>(sum * reduction_scale);
}

}  // namespace

template <typename T>
Status LaunchHyperConnectionPreMix(cudaStream_t stream, const T* streams,
                                   const void* pre_mix, int mix_type, T* y,
                                   int64_t count, int branches, int hidden,
                                   int gate_layout, float reduction_scale, int sigmoid_mode) {
  if (count == 0) return Status::OK();
  const int64_t block_count = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  PreMixKernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(
      streams, pre_mix, mix_type, y, count, branches, hidden, gate_layout, reduction_scale, sigmoid_mode);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                                  \
  template Status LaunchHyperConnectionPreMix<T>(cudaStream_t, const T*, const void*, int, T*, int64_t, \
                                                 int, int, int, float, int);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
