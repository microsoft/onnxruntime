// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void PadNCHWInputWithPaddingAlongHAndWImpl(
    cudaStream_t stream,
    const int64_t n,  // Batch
    const int64_t c,  // Channel
    const int64_t input_height,
    const int64_t output_height,
    const int64_t input_width,
    const int64_t output_width,
    const int64_t pad_height_start,
    const int64_t pad_width_start,
    const T pad_value,
    const int pad_mode,
    const T* input_data,
    T* output_data,
    const size_t N);

template <typename T>
void PadImpl(
    cudaStream_t stream,
    const size_t shape_rank,
    const TArray<int64_t>& input_dims,
    const TArray<int64_t>& input_strides,
    const TArray<int64_t>& lower_pads,
    const T pad_value,
    const int pad_mode,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
