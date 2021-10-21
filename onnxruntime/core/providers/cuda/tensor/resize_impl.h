// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/resize.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const std::vector<int64_t>& output_dims);

template <typename T>
void ResizeImpl(
    cudaStream_t stream,
    const onnxruntime::UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
    onnxruntime::ResizeNearestMode nearest_mode,
    void* dims_mapping);

}  // namespace cuda
}  // namespace onnxruntime
