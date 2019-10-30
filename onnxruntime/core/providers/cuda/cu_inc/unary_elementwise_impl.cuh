// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT, typename FuncT>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT& functor,
    CUDA_LONG N) {
  if (N == 0)  // special case where there's a dim value of 0 in the shape
    return;

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = functor(input_data[id]);
}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _UnaryElementWise<InT, OutT, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_data,
      output_data,
      func,
      N);
}

}  // namespace cuda
}  // namespace onnxruntime
