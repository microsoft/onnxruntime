// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/llm/attention_mask_convert.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void ConvertBoolMaskToFloatBiasKernel(
    T* output,
    const bool* input,
    CUDA_LONG size,
    T mask_filter_value) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    // Boolean mask semantics: true = attend (bias 0.0), false = mask out (bias mask_filter_value)
    output[idx] = input[idx] ? static_cast<T>(0.0f) : mask_filter_value;
  }
}

template <typename T>
Status LaunchConvertBoolMaskToFloatBias(
    cudaStream_t stream,
    T* output,
    const bool* input,
    int64_t size,
    T mask_filter_value) {
  if (size <= 0) {
    return Status::OK();
  }

  // Use CeilDiv to safely compute grid size and avoid integer overflow
  int grid_size = static_cast<int>(CeilDiv(size, static_cast<int64_t>(GridDim::maxThreadsPerBlock)));

  ConvertBoolMaskToFloatBiasKernel<T><<<grid_size, GridDim::maxThreadsPerBlock, 0, stream>>>(
      output, input, static_cast<CUDA_LONG>(size), mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status LaunchConvertBoolMaskToFloatBias<float>(
    cudaStream_t stream, float* output, const bool* input, int64_t size, float mask_filter_value);

template Status LaunchConvertBoolMaskToFloatBias<half>(
    cudaStream_t stream, half* output, const bool* input, int64_t size, half mask_filter_value);

template Status LaunchConvertBoolMaskToFloatBias<BFloat16>(
    cudaStream_t stream, BFloat16* output, const bool* input, int64_t size, BFloat16 mask_filter_value);

}  // namespace cuda
}  // namespace onnxruntime
