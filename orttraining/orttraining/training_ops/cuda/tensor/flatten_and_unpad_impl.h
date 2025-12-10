// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void FlattenAndUnpadImpl(cudaStream_t stream,
                         const int64_t total_element_count,
                         const fast_divmod output_element_stride_fdm,
                         const int64_t index_value_upper_bound,
                         const T* input_data,
                         const int64_t* indices_data,
                         T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
