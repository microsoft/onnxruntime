// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/hip_utils.h"
#include "common.cuh"

namespace onnxruntime {
namespace hip {

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT func,
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
      output_data[id] = func(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  HIP_LONG N = static_cast<HIP_LONG>(count);
  hipLaunchKernelGGL(_UnaryElementWise<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
          input_data,
          output_data,
          func,
          N);
}

}  // namespace hip
}  // namespace onnxruntime
