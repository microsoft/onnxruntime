// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void BitmaskDropoutKernelImpl(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const int64_t N,
    const float ratio,
    PhiloxGenerator& generator,
    const T* X_data,
    T* Y_data,
    uint32_t* mask_data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime