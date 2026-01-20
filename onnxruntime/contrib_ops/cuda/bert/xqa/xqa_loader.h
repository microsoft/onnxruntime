// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Wrapper for XQA MHA launch
// Only supports decoding (S=1) for now.
template <typename T>
Status LaunchXQAKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,        // [B, NumHeads, HeadSize]
    const void* key_cache,    // [B, MaxSeqLen, NumKVHeads, HeadSize] (or BNSH, but XQA usually expects contiguous or paged)
    const void* value_cache,  // [B, MaxSeqLen, NumKVHeads, HeadSize]
    void* output,             // [B, NumHeads, HeadSize]
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int actual_seq_len,  // Current sequence length (past + 1)
    const int max_seq_len,     // Max sequence length of cache
    const float scale,
    const bool is_bsnh,           // Layout of KV cache
    const int* seq_lens,          // Array of sequence lengths [BatchSize]
    const float* kv_cache_scale,  // KV cache dequant scale (nullptr for FP16/BF16, per-tensor float for INT8)
    const int kv_quant_type,      // 0=FP16/BF16, 1=INT8, 2=FP8
    void* workspace = nullptr,    // Scratch memory
    size_t workspace_size = 0     // Size of scratch memory
);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
