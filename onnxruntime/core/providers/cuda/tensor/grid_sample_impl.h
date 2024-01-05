// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void GridSampleImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[4],
    const int64_t H_out,
    const int64_t W_out,
    T* output_data);

template <typename T>
void GridSampleImpl3D(
    cudaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims_input[5],
    const int64_t D_out,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data);
}  // namespace cuda
}  // namespace onnxruntime
