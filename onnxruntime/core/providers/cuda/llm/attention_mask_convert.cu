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
    int64_t size,
    T mask_filter_value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  constexpr int block_size = 256;
  const int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  ConvertBoolMaskToFloatBiasKernel<T><<<grid_size, block_size, 0, stream>>>(
      output, input, size, mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status LaunchConvertBoolMaskToFloatBias<float>(
    cudaStream_t stream, float* output, const bool* input, int64_t size, float mask_filter_value);

template Status LaunchConvertBoolMaskToFloatBias<half>(
    cudaStream_t stream, half* output, const bool* input, int64_t size, half mask_filter_value);

#if !defined(DISABLE_FLOAT8_TYPES)
template Status LaunchConvertBoolMaskToFloatBias<BFloat16>(
    cudaStream_t stream, BFloat16* output, const bool* input, int64_t size, BFloat16 mask_filter_value);
#endif

}  // namespace cuda
}  // namespace onnxruntime
