// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void TileImpl(
    const size_t shape_rank,
    const TArray<fast_divmod>& input_shape,
    const TArray<int64_t>& input_strides,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
