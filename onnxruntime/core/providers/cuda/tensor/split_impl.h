// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

Status SplitImpl(cudaStream_t stream,
                 const size_t element_size,
                 const int block_size_including_axis_dim,
                 const int block_size_inside_axis_dim,
                 const int64_t* split_sizes,
                 const int64_t* split_sizes_range,
                 const int64_t* axis_dimension_input_output_mapping,
                 const int num_outputs,
                 const void* input_data,
                 void** output_ptr,
                 const size_t N);

}  // namespace cuda
}  // namespace onnxruntime