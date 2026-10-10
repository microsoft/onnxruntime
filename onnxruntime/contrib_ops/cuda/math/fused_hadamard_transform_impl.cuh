// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

cudaError_t LaunchFusedHadamardTransform(
    cudaStream_t stream,
    const half* input,
    const half* sign,
    half* output,
    int64_t block_count,
    int64_t sign_size,
    int block_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime