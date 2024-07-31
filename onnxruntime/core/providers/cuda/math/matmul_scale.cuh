// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
float ComputeScale(cudaStream_t stream, const Tensor* tensor);

}  // namespace cuda
}  // namespace onnxruntime
