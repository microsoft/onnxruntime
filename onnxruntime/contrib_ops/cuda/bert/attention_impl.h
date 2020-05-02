// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
size_t GetAttentionWorkspaceSize(size_t element_size, int batchsize, int num_heads, int head_size, int sequence_length);

bool LaunchAttentionKernel(
    const void* input,         // Input tensor
    const int* mask_index,     // Mask index (length of each sequence). NULL means no mask.
    void* output,              // Output tensor
    int batch_size,            // Batch size (B)
    int sequence_length,       // Sequence length (S)
    int num_heads,             // Number of attention heads (N)
    int head_size,             // Hidden layer size per head (H)
    void* workspace,           // Temporary buffer
    cublasHandle_t& cublas,    // Cublas handle
    const size_t element_size, // Element size of input tensor
    bool is_unidirectional     // Whether there is unidirecitonal mask.
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
