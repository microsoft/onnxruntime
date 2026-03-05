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

// Launch the fused linear attention recurrent step kernel.
// All pointers are device pointers.
//
// Parameters:
//   stream       - CUDA stream
//   update_rule  - which recurrence to use
//   query        - (B, H, 1, d_k)
//   key          - (B, H, 1, d_k)
//   value        - (B, H, 1, d_v)
//   past_state   - (B, H, d_k, d_v)  [input]
//   decay        - (B, H, 1, d_k) or nullptr; broadcastable from (B,H,1,1)
//   beta         - (B, H, 1, 1) or nullptr
//   output       - (B, H, 1, d_v)
//   present_state - (B, H, d_k, d_v) [output]
//   scale        - scaling factor
//   batch_size, num_heads, d_k, d_v - dimensions
//   decay_broadcasted - true if decay has full (B,H,1,d_k) shape, false if (B,H,1,1)
template <typename T>
Status LaunchLinearAttentionRecurrentKernel(
    cudaStream_t stream,
    LinearAttentionUpdateRule update_rule,
    const T* query,
    const T* key,
    const T* value,
    const T* past_state,
    const T* decay,
    const T* beta,
    T* output,
    T* present_state,
    float scale,
    int batch_size,
    int num_heads,
    int d_k,
    int d_v,
    bool decay_broadcasted);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
