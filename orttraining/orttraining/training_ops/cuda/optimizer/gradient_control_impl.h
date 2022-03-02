// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    cudaStream_t stream,
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);
}
}  // namespace onnxruntime

