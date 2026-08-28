// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void LpNormImpl(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t norm_size,
    int64_t num_norms,
    int64_t stride,
    int p);

}  // namespace cuda
}  // namespace onnxruntime
