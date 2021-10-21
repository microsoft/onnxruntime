// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void RoiAlignImpl(
    cudaStream_t stream,
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    const bool is_mode_avg,
    const int64_t* batch_indices_ptr);

}  // namespace cuda
}  // namespace onnxruntime
