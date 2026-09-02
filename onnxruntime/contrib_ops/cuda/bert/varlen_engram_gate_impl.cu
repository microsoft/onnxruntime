// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_engram_gate.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/engram_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
__global__ void VarlenEngramGateKernel(
    const T* key, const T* query, const T* value,
    const T* key_norm_scale, const T* query_norm_scale,
    const int32_t* cu_seqlens, T* output,
    int rows, int batch_size, int total_tokens, int hc_mult,
    int hidden_size, float epsilon) {
  if (cu_seqlens[0] != 0 || cu_seqlens[batch_size] != total_tokens) {
    return;
  }
  extern __shared__ float shared[];
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int g = row % hc_mult;
    const int token = row / hc_mult;
    const T* key_row = key + static_cast<int64_t>(row) * hidden_size;
    const T* query_row = query + static_cast<int64_t>(row) * hidden_size;
    const T* value_row = value + static_cast<int64_t>(token) * hidden_size;
    const T* key_scale = key_norm_scale + static_cast<int64_t>(g) * hidden_size;
    const T* query_scale = query_norm_scale + static_cast<int64_t>(g) * hidden_size;
    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    float dot = 0.0f;
    for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
      const float k = to_float<T>(key_row[c]);
      const float q = to_float<T>(query_row[c]);
      key_sum_sq += k * k;
      query_sum_sq += q * q;
      dot += k * to_float<T>(key_scale[c]) * q * to_float<T>(query_scale[c]);
    }
    engram_helper::BlockSum3(&key_sum_sq, &query_sum_sq, &dot, shared);
    const float inv_key = rsqrtf(key_sum_sq / hidden_size + epsilon);
    const float inv_query = rsqrtf(query_sum_sq / hidden_size + epsilon);
    dot = dot * inv_key * inv_query / sqrtf(static_cast<float>(hidden_size));
    const float gate = engram_helper::SigmoidFloat(engram_helper::EngramGateArg(dot));
    T* output_row = output + static_cast<int64_t>(row) * hidden_size;
    for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
      output_row[c] = from_float<T>(gate * to_float<T>(value_row[c]));
    }
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenEngramGateKernel(
    cudaStream_t stream, const T* key, const T* query, const T* value,
    const T* key_norm_scale, const T* query_norm_scale,
    const int32_t* cumulative_sequence_length, T* output,
    int batch_size, int total_tokens, int hc_mult, int hidden_size,
    float epsilon) {
  const int64_t rows_64 = static_cast<int64_t>(total_tokens) * hc_mult;
  if (rows_64 == 0 || hidden_size == 0) {
    return Status::OK();
  }
  const int blocks = static_cast<int>(std::min(rows_64, engram_helper::kMaxGridDimX));
  const size_t shared_bytes = 3 * static_cast<size_t>(engram_helper::kThreads) * sizeof(float);
  VarlenEngramGateKernel<T><<<blocks, engram_helper::kThreads, shared_bytes, stream>>>(
      key, query, value, key_norm_scale, query_norm_scale,
      cumulative_sequence_length, output, static_cast<int>(rows_64),
      batch_size, total_tokens, hc_mult, hidden_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                        \
  template Status LaunchVarlenEngramGateKernel<T>(cudaStream_t, const T*, const T*, const T*, \
                                                  const T*, const T*, const int32_t*, T*,     \
                                                  int, int, int, int, float);
INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
