// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U>
Status LaunchGemmaRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output1,
    T* output2,
    const U* emb,
    const T* q,
    const T* q_rot,
    const T* k,
    const T* k_rot,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
