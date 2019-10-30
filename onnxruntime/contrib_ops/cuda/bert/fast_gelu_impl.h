// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
bool computeGelu(cudaStream_t stream, int m, int n, const T* input, const T* bias, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
