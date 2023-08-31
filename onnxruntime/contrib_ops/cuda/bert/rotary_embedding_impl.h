// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchRotaryEmbeddingKernel(Stream* stream, const T* input, int64_t batch_size, int64_t num_heads, int64_t seqlen, int64_t head_dim, int64_t seqlen_with_past,
                                   const int64_t* pos, const T* cos_buffer, const T* sin_buffer, T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
