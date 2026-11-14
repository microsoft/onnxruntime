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
  kFp8 = 2,
  kInt4 = 3
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
    const int local_window_size,   // Sliding window size; -1 means global attention (no sliding window)
    const bool is_bsnh,            // Layout of KV cache
    const int* past_seq_lens,      // Past sequence lengths [BatchSize]
    const float* attention_sinks,  // Attention sink per query head, nullptr if not used
    const float* k_cache_scale,    // K cache dequant scale (nullptr for FP16/BF16, per-tensor float for INT8/FP8)
    const float* v_cache_scale,    // V cache dequant scale; may differ from k_cache_scale
                                   // For a quantized cache, a null scale means "1": the caller has folded a
                                   // per-channel scale into Q (for K) or rescales the output itself (for V).
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

// Dynamic shared memory (bytes) the XQA decode kernel requires for the given head size and
// query/KV group size on this device. The value is read from the loaded kernel module, so it is
// accurate even when the kernel is a PTX image JIT-compiled for the running SM. Returns 0 when it
// cannot be determined (e.g. unsupported head size / group size). Callers should skip XQA when the
// returned value exceeds device_prop.sharedMemPerBlockOptin (otherwise the launch fails with
// cudaErrorInvalidValue, e.g. consumer Blackwell sm_120 running a kernel built only for sm_90).
size_t GetXQARequiredSharedMemoryBytes(
    const cudaDeviceProp& device_prop,
    int head_size,
    int num_heads,
    int kv_num_heads);

// Multi-token (speculative verify) XQA over the same contiguous KV cache as LaunchXQAKernel, for a uniform query
// length 2 <= sequence_length <= kXqaSpecDecMaxSequenceLength. Query and output are token-major
// [B, S, num_heads, head_size]. The S new tokens must already be appended at [past_seq_lens[b], past_seq_lens[b] + S);
// query token s attends causally to cache positions < past_seq_lens[b] + s + 1 (within local_window_size if > 0).
// Supports FP16/BF16 caches, head_size 128 and group sizes 4, 8 and 16 without attention sinks.
constexpr int kXqaSpecDecMaxSequenceLength = 16;

bool IsXQASpecDecSupported(int head_size, int num_heads, int kv_num_heads, int sequence_length);

Status LaunchXQASpecDecKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    bool is_bf16,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int max_seq_len,
    float scale,
    int local_window_size,
    bool is_bsnh,
    const int* past_seq_lens,
    void* workspace,
    size_t workspace_size);

size_t GetXQASpecDecWorkspaceSize(
    const cudaDeviceProp& device_prop,
    bool is_bf16,
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int max_seq_len);

// Same contract as GetXQARequiredSharedMemoryBytes for the speculative kernel; 0 when unsupported.
size_t GetXQASpecDecRequiredSharedMemoryBytes(bool is_bf16, int head_size, int num_heads, int kv_num_heads);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
