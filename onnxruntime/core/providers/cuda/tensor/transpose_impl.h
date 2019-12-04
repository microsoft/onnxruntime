// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status TransposeImpl(size_t element_size, size_t shape_rank, const int64_t* input_strides, const size_t* perm,
                     const void* input_data, const fast_divmod* fdm_output_strides, void* output_data, size_t N);

}  // namespace cuda
}  // namespace onnxruntime
