// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/tensor/gelu_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/unary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct OP_Gelu {
  __device__ __inline__ T operator()(const T& a) const {
    return _Gelu(a);
  }
};

template <>
struct OP_Gelu<half> {
  __device__ __inline__ half operator()(const half& a) const {
    return static_cast<half>(_Gelu(static_cast<float>(a)));
  }
};

template <typename T>
Status LaunchGeluKernel(
    cudaStream_t stream,
    const T* input_data,
    T* output_data,
    size_t count) {
  UnaryElementWiseImpl(stream, input_data, output_data, OP_Gelu<T>(), count);

  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_GELU_IMPL(T)                                                                \
  template Status LaunchGeluKernel<T>(cudaStream_t stream, const T* input_data, T* output_data, \
                                      size_t count);

SPECIALIZED_GELU_IMPL(float);
SPECIALIZED_GELU_IMPL(half);
SPECIALIZED_GELU_IMPL(double);

#undef SPECIALIZED_GELU_IMPL

}  // namespace cuda
}  // namespace onnxruntime
