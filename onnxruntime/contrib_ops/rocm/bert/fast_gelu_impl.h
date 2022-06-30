// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove cudaDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
bool LaunchFastGeluKernel(hipStream_t stream, int input_length, int bias_length,
                          const T* input, const T* bias, T* output, bool use_half2);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
