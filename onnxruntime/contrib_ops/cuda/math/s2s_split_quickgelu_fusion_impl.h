// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream,
                                        const int num_outputs,
                                        const void* input_data, void* output_data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
