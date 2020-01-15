// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void WhereImpl(
    size_t output_rank_or_simple_broadcast,
    const int64_t* cond_padded_strides,
    const bool* cond_data,
    const int64_t* x_padded_strides,
    const T* x_data,
    const int64_t* y_padded_strides,
    const T* y_data,
    const fast_divmod* fdm_output_strides,
    T* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
