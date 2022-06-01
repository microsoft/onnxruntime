// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace cuda {

template <typename T>
void DropoutGradientKernelImpl(cudaStream_t stream, const int64_t N, const T* dY_data, const void* mask_data,
                               const float ratio, T* dX_data, bool use_bitmask);

}  // namespace cuda
}  // namespace onnxruntime
