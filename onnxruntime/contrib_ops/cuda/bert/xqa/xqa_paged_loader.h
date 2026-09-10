// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Number of tokens the paged XQA kernels are compiled for.
//
// XQA requires tokensPerPage to divide the kernel's CTA tile in the sequence dimension
// (mha_impl.cuh: `nbPagesPerCtaTile = exactDiv(ctaTile.x, tokensPerPage)`), which caps it at 128.
// PagedAttention's block_size is independent of this: a block of `block_size` tokens is presented
// to XQA as `block_size / kXqaTokensPerPage` consecutive pages. That remap is exact because the KV
// pool is contiguous -- [num_blocks, block_size, kv_num_heads, head_size] -- and XQA's
// PAGED_KV_CACHE_LAYOUT == 1 page is exactly [tokens_per_page, kv_num_heads, head_size]. So block b
// covers pages [b * kPagesPerBlock, (b + 1) * kPagesPerBlock).
constexpr int kXqaTokensPerPage = 128;

// Paged-KV XQA decode launcher. Unlike LaunchXQAKernel (contiguous per-request cache) this reads
// K and V from a shared block pool addressed through a page table.
// kInt4 uses packed UINT8 heads with static FP32 PER_CHANNEL scales folded into Q and the output
// by the caller. The K fold is divided by max|k_scale| and k_cache_scale carries that normalizer;
// v_cache_scale is null because the V scale is applied to the output. It supports FP16 query/output,
// head_size 256, and group_size 6 only. Other quantized types use FP32 per-tensor scales or the same
// normalized PER_CHANNEL folding. The INT4 shared-memory and scratch layouts match native FP16 XQA.
//
// Preconditions: one query token per sequence, head_size in {64, 128, 256}, group_size in
// {4, 6, 8, 16, 32}, supported FP16/INT8/FP8/INT4 cache, block_size % kXqaTokensPerPage == 0.
// PagedAttention currently routes native FP16 cache only for head_size=256 and group_size=6.
Status LaunchXQAPagedKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,        // [batch_size, num_heads, head_size]
    const void* key_cache,    // [num_blocks, block_size, kv_num_heads, head_size]
    const void* value_cache,  // [num_blocks, block_size, kv_num_heads, head_size]
    void* output,             // [batch_size, num_heads, head_size]
    const int* page_table,    // [batch_size, max_pages_per_seq], in units of kXqaTokensPerPage
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_pages_per_seq,   // page-table stride; max_seq_len = max_pages_per_seq * kXqaTokensPerPage
    const float scale,             // softmax scale applied to Q*K.T
    const int local_window_size,   // -1 => global attention
    const int* past_seq_lens,      // [batch_size]; the kernel attends to past_seq_lens[i] + 1 tokens
    const float* attention_sinks,  // [num_heads] fp32, nullptr if unused
    const float* k_cache_scale,    // per-tensor scale or folded-scale normalizer; nullptr means "1"
    const float* v_cache_scale,    // per-tensor scale; nullptr means "1" (applied to output)
    const XqaQuantType kv_quant_type,
    const bool is_bf16,  // dtype of query and output
    void* workspace,
    size_t workspace_size);

// Multi-token speculative-verification launcher. The implementation is deliberately limited to
// the DFlash2 target geometry: FP16/BF16 query/output, H256, group size 6, and matching native or
// INT8/FP8 paged KV, or packed INT4 with FP16 query/output and the PER_CHANNEL scale folding above.
Status LaunchXQAPagedSpecDecKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,  // [token_count, num_heads, head_size]
    const void* key_cache,
    const void* value_cache,
    void* output,  // [token_count, num_heads, head_size]
    const int* page_table,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_pages_per_seq,
    const float scale,
    const int local_window_size,
    const int* past_seq_lens,
    const int max_query_len,
    const int* cumulative_seqlens_q,
    const uint32_t* spec_dec_mask,
    const float* attention_sinks,
    const float* k_cache_scale,
    const float* v_cache_scale,
    const XqaQuantType kv_quant_type,
    const bool is_bf16,  // dtype of query and output; native cache has the same dtype
    void* workspace,
    size_t workspace_size);

size_t GetXQAPagedSpecDecWorkspaceSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int kv_num_heads,
    int max_pages_per_seq,
    int max_query_len,
    XqaQuantType kv_quant_type);

size_t GetXQAPagedSpecDecRequiredSharedMemoryBytes(XqaQuantType kv_quant_type);

// Workspace bytes required by LaunchXQAPagedKernel (semaphores + multi-block scratch). The paged
// and contiguous kernels share the CTA tile and the scratch layout, so this is GetXQAScratchSize
// called with max_seq_len = max_pages_per_seq * kXqaTokensPerPage.

// Dynamic shared memory the paged kernel requests, read from the loaded module. Returns 0 when the
// selected CUDA image has no compatible kernel or the size cannot be determined. Callers must skip
// XQA when this returns 0 or exceeds device_prop.sharedMemPerBlockOptin.
size_t GetXQAPagedRequiredSharedMemoryBytes(
    const cudaDeviceProp& device_prop,
    int head_size,
    int num_heads,
    int kv_num_heads,
    XqaQuantType kv_quant_type,
    bool is_bf16);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
