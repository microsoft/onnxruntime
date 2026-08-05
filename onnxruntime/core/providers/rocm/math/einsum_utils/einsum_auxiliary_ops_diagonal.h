// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {

namespace rocm {

void DiagonalImpl(
    hipStream_t stream,
    const void* input_data,
    const int64_t input_rank,
    const int64_t dim_1,
    const int64_t dim_2,
    const TArray<int64_t> input_strides,
    void* output_data,
    const TArray<fast_divmod> output_strides,
    const size_t output_size,
    size_t element_size);

}  // namespace rocm

}  // namespace onnxruntime
