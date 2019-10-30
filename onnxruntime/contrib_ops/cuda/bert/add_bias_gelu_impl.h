// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
bool LaunchAddBiasGeluKernel(const T* input, const T* bias, T* output, int m, int n);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
