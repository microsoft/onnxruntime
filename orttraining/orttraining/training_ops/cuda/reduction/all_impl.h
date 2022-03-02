// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
template <typename T>
void LaunchAllKernel(cudaStream_t stream, const T* data, const int size, bool* output);
}
}  // namespace onnxruntime

