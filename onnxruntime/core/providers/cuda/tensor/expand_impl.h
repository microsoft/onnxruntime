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
    const int64_t* input_dims,
    const int64_t* output_dims);

}  // namespace cuda
}  // namespace onnxruntime
