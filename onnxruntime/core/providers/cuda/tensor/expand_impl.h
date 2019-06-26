// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ExpandImpl(
    const size_t shape_rank,
    const size_t N,
    const size_t N_input,
    const T* input_data,
    T* output_data,
    const fast_divmod* fdm_input_dims,
    const fast_divmod* fdm_output_dims);

}  // namespace cuda
}  // namespace onnxruntime
