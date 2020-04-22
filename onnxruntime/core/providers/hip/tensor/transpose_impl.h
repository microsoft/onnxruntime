// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/shared_inc/hip_utils.h"

namespace onnxruntime {
namespace hip {

// Status TransposeImpl(size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
//                      const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int64_t N);
Status TransposeImpl(size_t element_size, int32_t shape_rank, const int64_t* input_strides,
                     const void* input_data, const fast_divmod* fdm_output_strides, void* output_data, int64_t N);
}  // namespace hip
}  // namespace onnxruntime
