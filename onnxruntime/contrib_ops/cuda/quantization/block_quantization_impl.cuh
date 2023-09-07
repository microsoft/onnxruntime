// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T, class S>
Status CudaBlockQuantize(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const T* x,
    unsigned const block_size,
    unsigned const block_count,
    S* scale,
    int8_t* y);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
