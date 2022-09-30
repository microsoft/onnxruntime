// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status QOrderedLayerNorm(
    cudaStream_t stream, const cudaDeviceProp& device_prop, cublasLtOrder_t order,
    const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
    const T* gamma, const T* beta, const float epsilon,
    unsigned batch, unsigned rows, unsigned cols);

}
}  // namespace contrib
}  // namespace onnxruntime
