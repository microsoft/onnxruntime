// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// dummy kernel
bool LaunchGpt3AttentionKernel(
    cudaStream_t stream,
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int element_count,
    size_t element_size
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
