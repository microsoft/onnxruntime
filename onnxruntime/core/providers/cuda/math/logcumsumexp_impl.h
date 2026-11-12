// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status LogCumSumExpImpl(cudaStream_t stream, const T* input, T* output,
                        int64_t width, int64_t inner, int64_t lanes);

}  // namespace cuda
}  // namespace onnxruntime
