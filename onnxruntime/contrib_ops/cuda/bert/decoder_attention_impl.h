// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& prop,       // Device Properties
    Stream* stream,                   // ORT Stream
    cublasHandle_t& cublas,           // Cublas handle
    const size_t element_size,        // Element size of input tensor
    const int batch_size,             // Batch size (B)
    const int sequence_length,        // Sequence length (S)
    const int kv_sequence_length,     // Key/Value/Cache sequence length
    const int num_heads,              // Number of attention heads (N)
    const int head_size,              // Hidden size per head (H)
    const bool static_kv,             // Whether cross attention or not
    const bool use_past,              // Whether use cache or not
    const bool has_layer_state,       // Whether output cache or not
    const bool has_key_padding_mask,  // Whether use key_padding_mask or not
    const float mask_filter_value,    // Mask filter value
    const void* gemm_query_buffer,    // Query buffer
    const void* gemm_kv_buffer,       // Key and value buffer
    const bool* key_padding_mask,     // Key padding mask
    const void* key_cache,            // Input key cache
    const void* value_cache,          // Input value cache
    void* qkv_buffer,                 // Temporary buffer
    void* workspace_buffer,           // Temporary buffer
    void* output,                     // Output tensor
    void* new_key_cache,              // New_key_cache tensor
    void* new_value_cache             // New_value_cache tensor
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
