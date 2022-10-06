// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "cos_grad_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct OP_CosGrad {
  __device__ __inline__ T operator()(const T& dy, const T& y) const {
    return dy * ((T)-1 * sin(y));
  }
};

#define BINARY_ELEMENTWISE_IMPL()                                                  \
  BINARY_ELEMENTWISE_IMPL_DECLARATION() {                                          \
    BinaryElementWiseNoBroadcastImpl(stream,                                           \
                                     lhs_data, rhs_data,                               \
                                     output_data,                                      \
                                     *reinterpret_cast<const OP_CosGrad<T>*>(func_ctx), \
                                     count);                                           \
  }

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL(T) \
  template void CosGradImpl<T>(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data, const OP_CosGrad* func_ctx, size_t count);

#define SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

// template <typename T>
// __global__ void _CosGradImpl(const T* dy, const T* Y, T* output, CUDA_LONG N) {
//   CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
//   output[id] = dy[id] * ((T)-1 *  sin<T>(Y[id]));
// }

// template <typename T>
// void CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t count) {
//   int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
//   CUDA_LONG N = static_cast<CUDA_LONG>(count);
//   _CosGradImpl<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(dy, Y, output, N);
// }

// #define SPECIALIZE_COSGRAD_IMPL(T) /
//   template void CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t count);

// SPECIALIZE_COSGRAD_IMPL(half)
// SPECIALIZE_COSGRAD_IMPL(float)
// SPECIALIZE_COSGRAD_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
