// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/status.h"
#include <cuda.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchBiasAddKernel(cudaStream_t stream, int32_t grid_size, int32_t num_channels,
                         T const* input, T const* bias, T const* residual, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
