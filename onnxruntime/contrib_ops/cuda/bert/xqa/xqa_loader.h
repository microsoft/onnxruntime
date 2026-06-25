// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Quantization type for XQA
enum class XqaQuantType {
  kNone = 0,  // no quantization, use FP16/BF16
  kInt8 = 1,
  kFp8 = 2
};

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
    const int max_seq_len,  // Max sequence length of cache
    const float scale,
    const bool is_bsnh,           // Layout of KV cache
    const int* past_seq_lens,     // Past sequence lengths [BatchSize]
    const float* kv_cache_scale,  // KV cache dequant scale (nullptr for FP16/BF16, per-tensor float for INT8)
    const XqaQuantType kv_quant_type,
    void* workspace = nullptr,  // Scratch memory
    size_t workspace_size = 0   // Size of scratch memory
);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int max_seq_len,
    XqaQuantType kv_quant_type,
    bool is_bf16 = false);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
