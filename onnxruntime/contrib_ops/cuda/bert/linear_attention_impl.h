// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Update rule types matching the CPU definition
enum class LinearAttentionUpdateRuleCuda {
  kLinear,
  kGated,
  kDelta,
  kGatedDelta,
};

template <typename T>
void LaunchLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,       // (B, H, T, d_k)
    const T* key,         // (B, H, T, d_k)
    const T* value,       // (B, H, T, d_v)
    const T* past_state,  // (B, H, d_k, d_v) or nullptr
    const T* decay,       // (B, H, T, decay_key_dim) or nullptr
    const T* beta,        // (B, H, T, 1) or nullptr
    T* output,            // (B, H, T, d_v)
    T* present_state,     // (B, H, d_k, d_v)
    int batch_size,
    int num_heads,
    int seq_len,
    int key_dim,
    int value_dim,
    int decay_key_dim,    // last dim of decay (1 or key_dim)
    float scale,
    LinearAttentionUpdateRuleCuda update_rule);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
