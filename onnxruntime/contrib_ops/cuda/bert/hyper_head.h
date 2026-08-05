// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchHyperHeadKernel(
    cudaStream_t stream, T* output, const T* hidden, const float* weight,
    const float* bias, const float* scale, int rows, int streams,
    int hidden_size, float epsilon, int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime