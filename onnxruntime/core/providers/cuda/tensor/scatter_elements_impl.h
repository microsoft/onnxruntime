// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin>
Status ScatterElementsImpl(
    cudaStream_t stream,
    const int rank,
    const T* input_data,
    const int64_t input_size,
    TArray<int64_t>& buffer_input_dims,
    TArray<int64_t>& buffer_input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    TArray<int64_t>& buffer_indices_dims,
    TArray<fast_divmod>& indices_strides,
    const T* updates,
    const int axis,
    T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
