// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void UpampleImpl(const onnxruntime::UpsampleMode upsample_mode,
                 const size_t rank,
                 const int64_t input_dim2,
                 const int64_t* input_pitches,
                 const fast_divmod* output_div_pitches,
                 const fast_divmod* scales_div,
                 const T* input_data,
                 T* output_data,
                 const size_t N);

template <typename T>
void ResizeImpl(
    const onnxruntime::UpsampleMode upsample_mode,
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const T* Xdata,
    T* Ydata,
    const size_t N);
}  // namespace cuda
}  // namespace onnxruntime
