// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
// Throw if "ptr" is not allocated on the CUDA device obtained by cudaGetDevice.
void CheckIfMemoryOnCurrentGpuDevice(const void* ptr);
}  // onnxruntime