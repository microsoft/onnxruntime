// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t N);

}
}  // namespace onnxruntime
