// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename U>
Status LaunchGemmaRotaryEmbeddingGradKernel(
    cudaStream_t stream,
    T* q_grad,
    T* q_rot_grad,
    T* k_grad,
    T* k_rot_grad,
    const T* go0,
    const T* go1,
    const U* emb,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

}  // namespace cuda
}  // namespace onnxruntime
