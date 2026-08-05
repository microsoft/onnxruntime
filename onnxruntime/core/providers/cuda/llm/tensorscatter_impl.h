// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status TensorScatterImpl(
    cudaStream_t stream,
    void* output_data,
    const void* update_data,
    const int64_t* write_indices,
    size_t element_size,
    int64_t prefix_count,
    int64_t prefix_stride_for_batch,
    int64_t max_seq_len,
    int64_t seq_len,
    int64_t suffix_count,
    bool circular);

}  // namespace cuda
}  // namespace onnxruntime
