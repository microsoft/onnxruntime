// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qorder_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename FuncT, int NumThreadsPerBlock = GridDim::maxThreadsPerBlock, int NumElementsPerThread = GridDim::maxElementsPerThread>
__global__ void _QOrderUnaryElementWise(
    const int8_t* input_data,
    float input_scale,
    int8_t* output_data,
    float output_scale,
    const FuncT functor,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  float value[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      value[i] = functor(input_scale * static_cast<float>(input_data[id])) / output_scale;
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = static_cast<int8_t>(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename FuncT>
void QOrderUnaryElementWiseImpl(
    cudaStream_t stream,
    const int8_t* input_data,
    const float* input_scale,
    int8_t* output_data,
    const float* output_scale,
    const FuncT& func,
    size_t count) {
  if (count > 0) {
    int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
    _QOrderUnaryElementWise<FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        input_data, *input_scale, output_data, *output_scale, func, static_cast<CUDA_LONG>(count));
  }
}


#define DEFINE_QORDER_OP(name, expr)                                 \
  struct QOrderUnaryOp##name {                                       \
    __device__ __inline__ float operator()(const float& a) const {   \
      return expr;                                                   \
    }                                                                \
  };

#define QORDER_UNARY_OP_IMPL(name)                                                         \
  QORDER_UNARY_OP_DECLARATION(name) {                                                      \
    QOrderUnaryElementWiseImpl<QOrderUnaryOp##name>(stream, input_data, input_scale, output_data, output_scale, \
                               QOrderUnaryOp##name(), count);                              \
  }


#define QORDER_UNARY_OP_NAME_EXPR(name, expr) \
  DEFINE_QORDER_OP(name, expr)                \
  QORDER_UNARY_OP_IMPL(name)

LIST_OF_QORDER_UNARY_OPS()
#undef QORDER_UNARY_OP_NAME_EXPR

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
