// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status Impl_CosGrad(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t N);

}
}  // namespace onnxruntime
