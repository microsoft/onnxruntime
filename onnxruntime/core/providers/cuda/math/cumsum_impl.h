// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void CumSumImpl(
    const T* input_data,
    const fast_divmod& input_dim_along_axis,
    const fast_divmod& input_stride_along_axis,
    T* output_data,
    const int64_t output_size,
    const size_t element_size,
    const bool exclusive,
    const bool reverse);

}  // namespace cuda
}  // namespace onnxruntime
