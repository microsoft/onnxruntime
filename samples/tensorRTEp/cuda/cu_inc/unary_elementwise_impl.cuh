// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>

namespace onnxruntime {
namespace cuda {

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT value[NumElementsPerThread];

  CUDA_LONG id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      value[i] = input_data[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = functor(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _UnaryElementWise<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_data,
          output_data,
          func,
          N);
}

}  // namespace cuda
}  // namespace onnxruntime
