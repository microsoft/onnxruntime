// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, int64_t input_size, int64_t axis, int64_t alpha,
                                        const Tensor& X, const Tensor& S, Tensor& Y);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
