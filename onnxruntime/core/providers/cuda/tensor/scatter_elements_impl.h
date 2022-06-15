// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TIndex>
Status ScatterElementsImpl(cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data,
                           const int64_t input_size, const int64_t input_dim_along_axis,
                           const int64_t input_stride_along_axis, const TArray<int64_t>& masked_input_strides,
                           const TIndex* indices_data, const int64_t indices_size,
                           const TArray<fast_divmod>& indices_fdms, const T* updates_data, T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
