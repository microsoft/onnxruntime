// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status LaunchEinsumDiagonal(cudaStream_t stream,
                            const void* input_data,
                            void* output_data,
                            size_t element_size,
                            size_t output_size,
                            const TArray<int64_t>& input_strides,
                            const TArray<int32_t>& input_axis_to_output_axis,
                            const TArray<fast_divmod>& output_strides);

Status LaunchEinsumTrace(cudaStream_t stream,
                         const void* input_data,
                         void* output_data,
                         size_t element_size,
                         size_t output_size,
                         int64_t trace_dim,
                         int64_t trace_stride,
                         const TArray<int64_t>& input_strides,
                         const TArray<int32_t>& input_axis_to_output_axis,
                         const TArray<fast_divmod>& output_strides);

}  // namespace cuda
}  // namespace onnxruntime
