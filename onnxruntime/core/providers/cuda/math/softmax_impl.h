// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
template<class T>
void launchSoftmaxKernel( const T* input, T* output, int N, int D);
}  // namespace cuda
}  // namespace onnxruntime