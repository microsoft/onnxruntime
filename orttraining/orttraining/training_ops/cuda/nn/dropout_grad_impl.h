// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

template <typename T, bool UseBitmask>
void DropoutGradientKernelImpl(cudaStream_t stream, const int64_t N, const T* dY_data, const void* mask_data,
                               const float ratio, T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime
