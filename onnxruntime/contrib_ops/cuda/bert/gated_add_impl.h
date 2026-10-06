// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchGatedAddKernel(cudaStream_t stream,
                            T* output,
                            const T* x,
                            const T* y,
                            const T* gate,
                            int64_t count,
                            int64_t hidden_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime