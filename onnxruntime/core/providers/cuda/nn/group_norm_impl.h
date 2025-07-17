// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename U>
Status GroupNormImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* scale_data,
    const T* bias_data,
    T* output_data,
    int64_t batch_size,
    int64_t num_channels,
    int64_t spatial_size,
    int64_t num_groups,
    int64_t stash_type,
    double epsilon);

}  // namespace cuda
}  // namespace onnxruntime