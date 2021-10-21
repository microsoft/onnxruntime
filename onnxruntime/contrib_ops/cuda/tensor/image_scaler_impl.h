// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void ImageScalerImpl(
    cudaStream_t stream,
    const T* input_data,
    const float scale,
    const float* bias_data,
    const int64_t dims[4],
    T* output_data,
    const size_t N);

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
