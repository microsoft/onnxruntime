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

__global__ void ConvertAndCopyQK(const int count, const float* input, nv_bfloat16* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

__global__ void ConvertAndCopyQK(const int count, const nv_bfloat16* input, float* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = __bfloat162float(input[idx]);
  }
}

template <typename T, typename QK>
Status CopyQK(cudaStream_t stream, int qk_size, const T* input, QK* output) {
  if constexpr (std::is_same_v<T, QK>) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        output, input, size_t(qk_size) * sizeof(T),
        cudaMemcpyDeviceToDevice, stream));
    return Status::OK();
  } else {
    constexpr bool h2f = std::is_same_v<T, half> && std::is_same_v<QK, float>;
    constexpr bool f2h = std::is_same_v<T, float> && std::is_same_v<QK, half>;
    constexpr bool b2f = std::is_same_v<T, BFloat16> && std::is_same_v<QK, float>;
    constexpr bool f2b = std::is_same_v<T, float> && std::is_same_v<QK, BFloat16>;

    static_assert(h2f || f2h || b2f || f2b, "CopyQK supports only (float<->half) and (float<->bfloat16).");

    constexpr int block = 256;
    const int grid = (qk_size + block - 1) / block;

    if constexpr (h2f || f2h) {
      ConvertAndCopyQK<<<grid, block, 0, stream>>>(qk_size, input, output);
      return CUDA_CALL(cudaGetLastError());
    } else if constexpr (b2f) {
      ConvertAndCopyQK<<<grid, block, 0, stream>>>(qk_size, reinterpret_cast<const __nv_bfloat16*>(input), output);
      return CUDA_CALL(cudaGetLastError());
    } else if constexpr (f2b) {
      ConvertAndCopyQK<<<grid, block, 0, stream>>>(qk_size, input, reinterpret_cast<__nv_bfloat16*>(output));
      return CUDA_CALL(cudaGetLastError());
    }
  }
}

template Status CopyQK<float, half>(cudaStream_t stream,
                                    const int qk_size,
                                    const float* input,
                                    half* output);

template Status CopyQK<half, float>(cudaStream_t stream,
                                    const int qk_size,
                                    const half* input,
                                    float* output);

template Status CopyQK<BFloat16, float>(cudaStream_t stream,
                                        const int qk_size,
                                        const BFloat16* input,
                                        float* output);

template Status CopyQK<float, BFloat16>(cudaStream_t stream,
                                        const int qk_size,
                                        const float* input,
                                        BFloat16* output);

template Status CopyQK(cudaStream_t stream,
                       const int qk_size,
                       const float* input,
                       float* output);

template Status CopyQK(cudaStream_t stream,
                       const int qk_size,
                       const half* input,
                       half* output);

template Status CopyQK(cudaStream_t stream,
                       const int qk_size,
                       const BFloat16* input,
                       BFloat16* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
