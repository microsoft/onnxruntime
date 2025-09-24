// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_type_conversion.h"
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
  using CudaT = typename OrtToCudaType<T>::type;
  using CudaQK = typename OrtToCudaType<QK>::type;

  if constexpr (std::is_same_v<CudaT, CudaQK>) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        output, input, size_t(qk_size) * sizeof(T),
        cudaMemcpyDeviceToDevice, stream));
    return Status::OK();
  } else {
    constexpr int block = 256;
    const int grid = (qk_size + block - 1) / block;

    ConvertAndCopyQK<<<grid, block, 0, stream>>>(
        qk_size,
        reinterpret_cast<const CudaT*>(input),
        reinterpret_cast<CudaQK*>(output));

    return CUDA_CALL(cudaGetLastError());
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
