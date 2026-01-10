// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status TriluImpl(
    cudaStream_t stream,
    bool upper,
    size_t element_size,
    int64_t k,
    const void* input_data,
    void* output_data,
    int N,
    const fast_divmod& batch_divmod_indices,
    const fast_divmod& row_col_divmod_indices);

}  // namespace cuda
}  // namespace onnxruntime
