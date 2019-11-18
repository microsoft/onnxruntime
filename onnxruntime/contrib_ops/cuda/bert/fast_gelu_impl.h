// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
bool LaunchFastGeluKernel(cudaStream_t stream, int input_length, int bias_length, const T* input, const T* bias, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
