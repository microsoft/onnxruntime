// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Launch the chunk-parallel linear attention kernel.
// Splits the sequence into chunks and processes intra-chunk in parallel,
// propagating state between chunks.
//
// Parameters:
//   stream        - CUDA stream
//   update_rule   - which recurrence to use
//   query         - (B, H, T, d_k)
//   key           - (B, H, T, d_k)
//   value         - (B, H, T, d_v)
//   initial_state - (B, H, d_k, d_v) or nullptr (zeros)
//   decay         - (B, H, T, d_k) or nullptr
//   beta          - (B, H, T, 1) or nullptr
//   output        - (B, H, T, d_v)
//   final_state   - (B, H, d_k, d_v)
//   workspace     - temporary GPU memory
//   scale         - scaling factor
//   batch_size, num_heads, seq_len, d_k, d_v, chunk_size
//   decay_broadcasted - true if decay is per-key-dim, false if scalar per head
template <typename T>
Status LaunchLinearAttentionChunkParallelKernel(
    cudaStream_t stream,
    LinearAttentionUpdateRule update_rule,
    const T* query,
    const T* key,
    const T* value,
    const T* initial_state,
    const T* decay,
    const T* beta,
    T* output,
    T* final_state,
    void* workspace,
    size_t workspace_size,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    bool decay_broadcasted);

// Compute workspace size needed for chunk-parallel kernel
size_t GetLinearAttentionChunkParallelWorkspaceSize(
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    size_t element_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
