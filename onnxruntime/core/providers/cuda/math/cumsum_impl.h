// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//#include <stdint.h>
//#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

void CumSumImpl(
    const int64_t input_rank,
    const void* input_data,
    const int64_t axis,
    const int64_t input_dim_along_axis,
    const int64_t* input_strides,
    void* output_data,
    const int64_t output_size,
    size_t element_size,
    bool exclusive,
    bool reverse);

}  // namespace cuda
}  // namespace onnxruntime
