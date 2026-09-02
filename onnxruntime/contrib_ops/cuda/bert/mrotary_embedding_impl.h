// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// mrope_section holds the 3 (T, H, W) section sizes; their sum must equal
// rotary_embedding_dim / 2. mrope_layout: 0 = Sectioned/Chunked, 1 = Interleaved.
template <typename T>
Status LaunchMRotaryEmbeddingKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const int64_t* position_ids,  // (3, batch_size, sequence_length)
    const T* cos_cache,
    const T* sin_cache,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int rotary_embedding_dim,
    const int max_sequence_length,
    const bool interleaved,
    const int3 mrope_section,
    const int mrope_layout,
    const float scale,
    const int max_threads_per_block,
    const bool is_input_bnsh_format);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
