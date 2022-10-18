// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Fused kernel of Add (bias) and Transpose.
// Shape of inputs and outputs:
//     biases:  (num_matrices, num_heads * head_size)
// format 0:
//     input:   (num_matrices, batch_size, sequence_length, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
// format 1:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
// format 2:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (batch_size, sequence_length, num_heads, num_matrices, head_size)

template <typename T>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const T* input, const T* biases, T* output, bool enable_half4, const int v_head_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
