// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// `cols` is the width of one half when gate and up are the two halves of a single
// `[.., 2 * cols]` row; pass 0 when they are separate, densely packed tensors.
template <typename T>
Status LaunchDSV4SwiGLU(cudaStream_t stream, int64_t count, int64_t cols, float limit,
                        const T* gate, const T* up, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
