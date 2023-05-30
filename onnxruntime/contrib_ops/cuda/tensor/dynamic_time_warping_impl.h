// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetDynamicTimeWarpingBufferSize(size_t batch, size_t rows, size_t cols, size_t& max_index_len) {
  max_index_len = rows + cols + 1;
  size_t cost_buffer_size = static_cast<size_t>((rows + 1) * (cols + 1));
  return batch * max_index_len * 2 * sizeof(int32_t) + // two index arrays
         sizeof(int64_t) + // final index array length
         batch* cost_buffer_size * sizeof(float) + // cost buffer
         batch* cost_buffer_size * sizeof(int8_t); // trace buffer
}

Status LaunchDynamicTimeWarping(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t batch,
    size_t rows,
    size_t cols,
    const float* input,
    void* buffer,
    size_t& result_len
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
