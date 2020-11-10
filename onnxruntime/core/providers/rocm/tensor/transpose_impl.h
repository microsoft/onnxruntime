// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

bool CanDoTranspose3D(int32_t rank, const std::vector<int64_t>& input_dims, const std::vector<size_t>& permutations);
// Status Transpose3DImpl(size_t element_size, const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides, const void* input_data,
//                        void* output_data, int64_t N);
Status Transpose3DImpl(size_t element_size, const int64_t* input_shape, const int64_t* input_strides, const void* input_data,
                       void* output_data, int64_t N);

// Status TransposeImpl(size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
//                      const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int64_t N);
Status TransposeImpl(size_t element_size, int32_t shape_rank, const int64_t* input_strides,
                     const void* input_data, const fast_divmod* fdm_output_strides, void* output_data, int64_t N);
}  // namespace rocm
}  // namespace onnxruntime
