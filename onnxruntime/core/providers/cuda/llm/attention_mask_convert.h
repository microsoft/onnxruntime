// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

// Convert boolean attention mask to float attention bias.
// Boolean semantics: true = attend (output 0.0), false = mask out (output mask_filter_value)
template <typename T>
Status LaunchConvertBoolMaskToFloatBias(
    cudaStream_t stream,
    T* output,              // [size] output buffer (float type)
    const bool* input,      // [size] input buffer (bool type)
    int64_t size,           // total number of elements
    T mask_filter_value);   // value to use for masked positions (typically -10000.0f)

}  // namespace cuda
}  // namespace onnxruntime
