// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/math/matrix29_impl.h"

#include <type_traits>

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void Gemm29BiasKernel(T* output, const T* bias, float beta, int64_t count,
                                 int rank, const int64_t* dimensions, const int64_t* bias_strides) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t index = i;
    int64_t offset = rank < 0 ? i : 0;
    for (int d = rank - 1; d >= 0; --d) {
      offset += (index % dimensions[d]) * bias_strides[d];
      index /= dimensions[d];
    }
    using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    output[i] = static_cast<T>(static_cast<AccT>(output[i]) + static_cast<AccT>(beta) * static_cast<AccT>(bias[offset]));
  }
}

template <typename T>
Status Gemm29AddBias(cudaStream_t stream, T* output, const T* bias, float beta,
                     int64_t count, int rank, const int64_t* dimensions, const int64_t* bias_strides) {
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
  Gemm29BiasKernel<<<blocks, 256, 0, stream>>>(output, bias, beta, count, rank, dimensions, bias_strides);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void GroupedBiasKernel(T* output, const T* const* bias, int64_t count, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    output[i] = bias ? bias[i / n][i % n] : T(0.0f);
  }
}

template <typename T>
Status GroupedMatMulBias(cudaStream_t stream, T* output, const T* const* bias, int64_t count, int64_t n) {
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
  GroupedBiasKernel<<<blocks, 256, 0, stream>>>(output, bias, count, n);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void GatherRowsKernel(const T* input, T* packed, const int64_t* order,
                                 int64_t count, int64_t k, int64_t selections) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    packed[i] = input[(order[i / k] / selections) * k + i % k];
  }
}

template <typename T>
Status GroupedMatMulGatherRows(cudaStream_t stream, const T* input, T* packed, const int64_t* order,
                               int64_t count, int64_t k, int64_t selections) {
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
  GatherRowsKernel<<<blocks, 256, 0, stream>>>(input, packed, order, count, k, selections);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void ScatterRowsKernel(const T* packed, T* output, const int64_t* order,
                                  const T* const* bias, int64_t count, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const AccT value = static_cast<AccT>(packed[i]);
    output[order[i / n] * n + i % n] =
        bias ? static_cast<T>(value + static_cast<AccT>(bias[i / n][i % n])) : packed[i];
  }
}

template <typename T>
Status GroupedMatMulScatterRows(cudaStream_t stream, const T* packed, T* output, const int64_t* order,
                                const T* const* bias, int64_t count, int64_t n) {
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / 256 + 1, 65535));
  ScatterRowsKernel<<<blocks, 256, 0, stream>>>(packed, output, order, bias, count, n);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                                               \
  template Status Gemm29AddBias<T>(cudaStream_t, T*, const T*, float, int64_t, int, const int64_t*, const int64_t*); \
  template Status GroupedMatMulBias<T>(cudaStream_t, T*, const T* const*, int64_t, int64_t);                         \
  template Status GroupedMatMulGatherRows<T>(cudaStream_t, const T*, T*, const int64_t*, int64_t, int64_t, int64_t); \
  template Status GroupedMatMulScatterRows<T>(cudaStream_t, const T*, T*, const int64_t*, const T* const*, int64_t, int64_t);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(half)
INSTANTIATE(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
