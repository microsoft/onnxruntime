// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetLongformerAttentionWorkspaceSize(
    size_t element_size,
    int batchsize,
    int num_heads,
    int head_size,
    int sequence_length,
    int max_num_global);

bool LaunchLongformerAttentionKernel(
    const cudaDeviceProp& device_prop,  // Device Properties
    cudaStream_t stream,                // CUDA stream
    const void* input,                  // Input tensor
    const void* attention_mask,         // Attention mask with shape (B, S)
    const void* global_input,           // Global attention input, or nullptr when max_num_global == 0.
    const int* global_attention,        // Global attention flags with shape (B, S)
    void* output,                       // Output tensor
    int batch_size,                     // Batch size (B)
    int sequence_length,                // Sequence length (S)
    int num_heads,                      // Number of attention heads (N)
    int head_size,                      // Hidden layer size per head (H)
    int window,                         // One sided attention window (W)
    int max_num_global,                 // Maximum number of global tokens (G)
    void* workspace,                    // Temporary buffer
    cublasHandle_t& cublas,             // Cublas handle
    const size_t element_size           // Element size of input tensor
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
