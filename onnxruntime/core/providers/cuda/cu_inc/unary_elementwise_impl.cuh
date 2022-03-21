// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

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

template <typename InT, typename OutT, typename InT2, typename OutT2, typename FuncT, typename FuncT2, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    const FuncT2 functor2,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT2 value[NumElementsPerThread];

  InT2* input_data2 = (InT2*)input_data;

  CUDA_LONG id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N / 2) {
      value[i] = input_data2[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N / 2) {
      reinterpret_cast<OutT2*>(output_data)[id] = functor2(value[i]);
      id += NumThreadsPerBlock;
    }
  }

  if (start == N / 2 && N % 2 == 1) {
    printf("in the half2 unaryelementwise function N % 2 == 1.... \n");
    output_data[N-1] = functor(input_data[N-1]);
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


template <typename T>
struct VectorizeTypeMap {
  typedef T VectorizeT;
};

template <>
struct VectorizeTypeMap<half> {
  typedef __half2 VectorizeT;
};

template <>
struct VectorizeTypeMap<BFloat16> {
  typedef __nv_bfloat162  VectorizeT;
};

template <>
struct VectorizeTypeMap<float> {
  typedef float2 VectorizeT;
};

template <typename InT, typename OutT, typename FuncT, typename FuncT2>
void UnaryElementWiseImpl(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    const FuncT2& func2,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;
  
  int blocksPerGrid = static_cast<int>(CeilDiv(CeilDiv(count, 2), GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  typedef typename VectorizeTypeMap<InT>::VectorizeT VectorizeInT;
  typedef typename VectorizeTypeMap<OutT>::VectorizeT VectorizeOuT;
  _UnaryElementWise<InT, OutT, VectorizeInT, VectorizeOuT, FuncT, FuncT2, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input_data,
          output_data,
          func,
          func2,
          N);
}

}  // namespace cuda
}  // namespace onnxruntime
