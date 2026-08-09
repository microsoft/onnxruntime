// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

void GatherImpl(
    cudaStream_t stream,
    const int64_t input_block_size,
    const int64_t indices_max,
    const fast_divmod& output_block_size,
    const fast_divmod& block_size,
    const void* indices_data,
    size_t index_element_size,
    const void* input_data,
    size_t element_size,
    void* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
