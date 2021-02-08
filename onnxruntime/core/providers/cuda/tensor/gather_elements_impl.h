// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

void GatherElementsImpl(
    cudaStream_t stream,
    const int64_t rank,  // both inputs have same rank and this is validated in the main Compute
    const void* input_data,
    const int64_t input_dim_along_axis,
    const TArray<int64_t>& input_strides,
    const void* indices_data,
    const int64_t indices_size,
    const TArray<fast_divmod>& indices_strides,
    const int64_t axis,
    void* output_data,
    size_t element_size,
    size_t index_element_size);

}  // namespace cuda
}  // namespace onnxruntime
