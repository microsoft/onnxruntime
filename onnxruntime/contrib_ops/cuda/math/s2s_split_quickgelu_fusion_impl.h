// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, int dim, int64_t input_size, float alpha, const T* input_data, T* output_data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
