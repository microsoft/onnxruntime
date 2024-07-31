// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, T* h_scale_coef);

}  // namespace cuda
}  // namespace onnxruntime
