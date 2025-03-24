// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/attention_qk.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__ void ConvertAndCopyQK(const int count, const float* input, half* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = __float2half(input[idx]);
  }
}

__global__ void ConvertAndCopyQK(const int count, const half* input, float* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = __half2float(input[idx]);
  }
}

template <typename T>
__global__ void ConvertAndCopyQK(const int count, const T* input, T* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = input[idx];
  }
}

template <typename T, typename QK>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const T* input,
              QK* output) {
  constexpr const bool half2float = std::is_same<T, half>::value && std::is_same<QK, float>::value;
  constexpr const bool float2half = std::is_same<T, float>::value && std::is_same<QK, half>::value;
  static_assert(half2float || float2half, "This function supports either <float,half> or <half,float>");

  constexpr const int block_size = 256;
  int num_blocks = (qk_size + block_size - 1) / block_size;
  ConvertAndCopyQK<<<num_blocks, block_size, 0, stream>>>(qk_size, input, output);

  return CUDA_CALL(cudaGetLastError());
}

template Status CopyQK<float, half>(cudaStream_t stream,
                                    const int qk_size,
                                    const float* input,
                                    half* output);

template Status CopyQK<half, float>(cudaStream_t stream,
                                    const int qk_size,
                                    const half* input,
                                    float* output);

template <>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const float* input,
              float* output) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, qk_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  return Status::OK();
}

template <>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const half* input,
              half* output) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, qk_size * sizeof(half), cudaMemcpyDeviceToDevice, stream));
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
