// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/resize.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ResizeImpl(const onnxruntime::UpsampleMode upsample_mode,
                const size_t rank,
                const int64_t input_dim2,
                const int64_t* input_pitches,
                const fast_divmod* output_div_pitches,
                const float* scales_vals,
                const T* input_data,
                T* output_data,
                const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
