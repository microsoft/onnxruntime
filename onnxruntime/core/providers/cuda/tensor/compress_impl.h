// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

void PrefixSumImpl(const int8_t* condition_data,
                   int32_t* condition_cumulative_sum,
                   const size_t length);

Status CompressImpl(const size_t element_bytes,
                    const int32_t valid_condition_length,
                    const int32_t axis_right_stride,
                    const int32_t input_axis_dim_length,
                    const int32_t output_axis_dim_length,
                    const int32_t* condition_cumulative_sum,
                    const bool* condition_data,
                    const void* input_data,
                    void* output_data,
                    const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
