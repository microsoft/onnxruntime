#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace rocm {

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    HIP_LONG N) {
  HIP_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT value[NumElementsPerThread];

  HIP_LONG id = start;
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

template <typename InT, typename OutT, typename InT2, typename OutT2, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWiseVectorize(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    HIP_LONG N) {
  HIP_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT2 value[NumElementsPerThread];

  InT2* input_data2 = (InT2*)input_data;

  HIP_LONG id = start;
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
      reinterpret_cast<OutT2*>(output_data)[id] = functor(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    hipStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_UnaryElementWise<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>), blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream, 
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
struct VectorizeTypeMap<float> {
  typedef float2 VectorizeT;
};

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImplVectorize(
    hipStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;
  
  int blocksPerGrid = static_cast<int>(CeilDiv(CeilDiv(count, 2), GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  typedef typename VectorizeTypeMap<InT>::VectorizeT VectorizeInT;
  typedef typename VectorizeTypeMap<OutT>::VectorizeT VectorizeOuT;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(_UnaryElementWiseVectorize<InT, OutT, VectorizeInT, VectorizeOuT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>), blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream, 
          input_data,
          output_data,
          func,
          N);
}

}  // namespace rocm
}  // namespace onnxruntime

