// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <class T>
struct Func_Assignment {
  __device__ __inline__ void operator()(T* a, const T* b) const {
    *a = *b;
  }
};

template <typename CudaT, typename Tin, typename FuncT>
Status ScatterElementsImpl(
    const int rank,
    const CudaT* input_data,
    const int64_t input_size,
    TArray<int64_t>& buffer_input_dims,
    TArray<int64_t>& buffer_input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    TArray<int64_t>& buffer_indices_dims,
    TArray<fast_divmod>& indices_strides,
    const CudaT* updates,
    const int axis,
    CudaT* output_data,
    const FuncT& func);

}  // namespace cuda
}  // namespace onnxruntime

