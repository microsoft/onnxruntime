// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

Status LaunchUnfoldTensor(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t element_size,
    const void* input,
    void* output,
    int64_t leading_dim,
    int64_t tailing_dim,
    int64_t dim_size,
    int64_t unfold_size,
    int64_t step
);

}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
