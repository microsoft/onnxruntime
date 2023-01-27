// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/status.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchSplitGeluKernel(cudaStream_t stream, int32_t grid_size, int32_t half_hidden_size, T const* input, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
