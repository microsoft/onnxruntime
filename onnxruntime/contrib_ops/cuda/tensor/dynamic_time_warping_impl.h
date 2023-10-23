// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetDynamicTimeWarpingBufferSize(size_t batch, size_t rows, size_t cols, size_t& max_index_len);

Status LaunchDynamicTimeWarping(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t batch,
    size_t rows,
    size_t cols,
    const float* input,
    void* buffer,
    size_t& result_len);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
