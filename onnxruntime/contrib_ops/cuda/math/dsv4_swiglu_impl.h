// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchDSV4SwiGLU(cudaStream_t stream, int64_t count, float limit, const T* gate,
                        const T* up, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
