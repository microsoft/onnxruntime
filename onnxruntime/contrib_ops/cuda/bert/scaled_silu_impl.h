// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
Status LaunchScaledSiLU(cudaStream_t stream, const T* x, const void* scale,
                        int scale_type, T* y, int64_t count, float alpha);

}  // namespace onnxruntime::contrib::cuda
