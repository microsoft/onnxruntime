// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void TransposeImpl(
    const size_t shape_rank,
    const int64_t* input_strides,
    const int64_t* perm,
    const T* input_data,
    const fast_divmod* fdm_output_strides,
    T* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
