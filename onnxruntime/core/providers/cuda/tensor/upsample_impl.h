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
void UpampleImpl(cudaStream_t stream,
                 const onnxruntime::UpsampleMode upsample_mode,
                 const size_t rank,
                 const int64_t input_dim2,
                 const TArray<int64_t>& input_pitches,
                 const TArray<fast_divmod>& output_div_pitches,
                 const TArray<fast_divmod>& scales_div,
                 const T* input_data,
                 T* output_data,
                 const size_t N);

}  // namespace cuda
}  // namespace onnxruntime
