// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ORT-side adaptation of the vendored FlashMLA decode kernel (see flash_mla/README.md).
//
// FlashMLA solves exactly the shape PagedLatentAttentionKernel + PagedLatentReduce solve for
// DeepSeek-V4-Flash decode -- absorbed MLA, one latent KV head, head_size 576 with the value
// occupying the leading 512 channels, paged cache with a 64-token page -- but on Hopper it does it
// with wgmma and TMA instead of a hand-rolled split-KV loop.
//
// Nothing in here includes flash_mla.h: that header declares symbols that only exist when the
// sm90a object library is built, so it stays confined to the .cu. Callers see plain parameters.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/common/status.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Everything the launcher needs that is not a pointer. Split out from the pointers so the
// host-side gate and the workspace sizing can be evaluated before any buffer exists.
struct FlashMlaLatentConfig {
  int batch_size = 0;
  int num_heads = 0;      // query heads per rank; becomes FlashMLA's `ngroups`
  int kv_num_heads = 0;   // must be 1 -- the latent cache is MQA
  int head_size = 0;      // must be 576
  int v_head_size = 0;    // must be 512
  int block_size = 0;     // page size; must be 64
  int max_num_blocks_per_seq = 0;
  int seqlen_q = 0;       // query tokens per sequence; must be uniform across the batch
  int num_sm_parts = 0;   // filled in by ComputeFlashMlaNumSmParts
  float scale = 0.f;
};

// True when this step can run on FlashMLA. Deliberately strict: every condition below is something
// the kernel assumes rather than checks, so a false positive is silent corruption, not a failure.
// `is_bf16` and `cache_is_unquantized` come from the caller's template types.
bool FlashMlaLatentSupported(const FlashMlaLatentConfig& config, bool is_bf16, bool cache_is_unquantized,
                             bool has_head_sink, bool has_kv_indices, float softcap, int local_window_size,
                             int sm_major);

// FlashMLA splits the KV range across `num_sm_parts` persistent tiles rather than a fixed split
// count. Upstream's own helper ignores the KV length and always returns the SM count, which is far
// too many for a short context; this follows the benchmark's formula instead, which divides the
// SMs by the number of query tiles so each part gets real work.
//
// Must be derived only from replay-invariant bounds -- it sizes a workspace and is baked into a
// captured CUDA graph.
int ComputeFlashMlaNumSmParts(int seqlen_q, int num_heads, int kv_num_heads, int multi_processor_count);

// Bytes of scratch the launcher needs: the tile-scheduler metadata, the per-sequence split counts,
// the softmax LSE, the split accumulators, and the per-sequence total KV lengths.
size_t GetFlashMlaLatentWorkspaceBytes(const FlashMlaLatentConfig& config);

// Single source of truth for "does this step run on FlashMLA, and with what config". The workspace
// is allocated in paged_attention.cc but the kernel is launched from paged_attention_impl.cu; if
// those two sites decided independently they could disagree, and the failure mode would be a
// kernel writing into a buffer that was never allocated. Both call this instead.
//
// Returns false (leaving *config untouched) when the env switch is off or any precondition fails.
// Every argument is a host-side shape or attribute, so the answer is replay-invariant and safe to
// bake into a captured CUDA graph.
bool TryBuildFlashMlaLatentConfig(int batch_size, int token_count, int num_heads, int kv_num_heads,
                                  int head_size, int v_head_size, int block_size, int max_num_blocks_per_seq,
                                  float scale, float softcap, int local_window_size, bool has_head_sink,
                                  bool has_kv_indices, bool is_bf16, bool cache_is_unquantized, int sm_major,
                                  int multi_processor_count, FlashMlaLatentConfig* config);

// `workspace` must be at least GetFlashMlaLatentWorkspaceBytes(config) and 256-byte aligned.
// `past_seqlens` is the KV length at the start of the step; the launcher forms the total length
// itself. `key_cache` is used for both K and V -- the value is the leading v_head_size channels of
// the same row, which is what makes this MLA rather than ordinary paged attention.
template <typename T>
Status LaunchFlashMlaLatentAttention(const T* query, const T* key_cache, const int* past_seqlens,
                                     const int* block_table, T* output, void* workspace,
                                     const FlashMlaLatentConfig& config, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
