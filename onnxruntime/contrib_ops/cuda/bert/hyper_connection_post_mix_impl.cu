// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_post_mix_impl.h"

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

__device__ __forceinline__ float LoadMix(const void* data, int64_t index, int element_type) {
  if (element_type == kFloat) return static_cast<const float*>(data)[index];
  if (element_type == kFloat16) return __half2float(static_cast<const half*>(data)[index]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

__device__ __forceinline__ int64_t GateIndex(int layout, int64_t row, int branch,
                                             int feature, int branches, int hidden) {
  if (layout == 0) return 0;
  if (layout == 1 || layout == 2) return row * branches + branch;
  return (row * branches + branch) * hidden + feature;
}

template <typename T>
__global__ void PostMixKernel(const T* streams, const T* branch_output,
                              const void* post_mix, const void* stream_mix,
                              int mix_type, T* y, int64_t count, int branches,
                              int hidden, int gate_layout) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int h = static_cast<int>(i % hidden);
  const int c = static_cast<int>((i / hidden) % branches);
  const int64_t row = i / (static_cast<int64_t>(hidden) * branches);
  float value = stream_mix == nullptr ? to_float<T>(streams[i]) : 0.0f;
  if (stream_mix != nullptr) {
    for (int source = 0; source < branches; ++source) {
      value += LoadMix(stream_mix, (row * branches + source) * branches + c, mix_type) *
               to_float<T>(streams[(row * branches + source) * hidden + h]);
    }
  }
  value += LoadMix(post_mix, GateIndex(gate_layout, row, c, h, branches, hidden), mix_type) *
           to_float<T>(branch_output[row * hidden + h]);
  y[i] = from_float<T>(value);
}

}  // namespace

template <typename T>
Status LaunchHyperConnectionPostMix(
    cudaStream_t stream, const T* streams, const T* branch_output,
    const void* post_mix, const void* stream_mix, int mix_type, T* y,
    int64_t count, int branches, int hidden, int gate_layout) {
  if (count == 0) return Status::OK();
  const int64_t block_count = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");
  PostMixKernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(
      streams, branch_output, post_mix, stream_mix, mix_type, y, count, branches, hidden, gate_layout);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                                        \
  template Status LaunchHyperConnectionPostMix<T>(cudaStream_t, const T*, const T*, const void*, const void*, \
                                                  int, T*, int64_t, int, int, int);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
