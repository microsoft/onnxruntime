// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

<<<<<<< HEAD
template <typename T>
void TransposeImpl(int32_t rank, int64_t N,
                   const TArray<int64_t>& input_strides, const T* input_data,
                   const TArray<fast_divmod>& output_strides, T* output_data);
=======
Status TransposeImpl(size_t element_size, size_t shape_rank, const int64_t* input_strides, const size_t* perm,
                     const void* input_data, const fast_divmod* fdm_output_strides, void* output_data, size_t N);
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

}  // namespace cuda
}  // namespace onnxruntime
