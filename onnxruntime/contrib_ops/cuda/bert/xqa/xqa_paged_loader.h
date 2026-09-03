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
//
// Preconditions: one query token per sequence, head_size in {64, 128, 256}, group_size in
// {4, 6, 8, 16, 32}, supported FP16/INT8/FP8 cache, block_size % kXqaTokensPerPage == 0.
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
    const float* k_cache_scale,    // per-tensor dequant scale; nullptr means "1" (folded into Q)
    const float* v_cache_scale,    // per-tensor dequant scale; nullptr means "1" (applied to output)
    const XqaQuantType kv_quant_type,
    const bool is_bf16,  // dtype of query and output
    void* workspace,
    size_t workspace_size);

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
