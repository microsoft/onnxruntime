// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, const size_t element_size,
                                        const int num_outputs,
                                        const void* input_data, void* output_data, const size_t input_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
