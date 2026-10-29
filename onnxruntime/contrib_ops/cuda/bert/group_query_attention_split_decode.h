// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Single-token decode attention for GroupQueryAttention on a BNSH fp16 KV cache that already holds
// the new token (PrepareQKV has appended it).
//
// The flash path for this case launches a split-KV kernel over all query heads, a combine kernel and
// two memsets of the split accumulators. This kernel runs one CTA per (batch, kv head, split); each
// warp streams its own 8-key steps of the split straight from global memory into tensor-core mma
// fragments, with the 8 query heads of the KV head as the rows of one m16 tile, so K and V are read
// once for the whole group. The warps of a CTA are combined in shared memory and the last CTA of a
// (batch, kv head) merges the per-split softmax partials, so there is no second kernel and no memset.
// Requires sm_80 (the kernel body is empty on older architectures; the Flash path gates it).
//
// A CUDA graph freezes the grid, so the kernel is launched with GqaSplitDecodeMaxSplits() splits per
// (batch, kv head), which depends only on the batch size, the KV head count, the KV buffer length and
// the GPU (about kGqaSplitDecodeCtasPerSm CTAs per SM). Each batch then uses
// ceil(total_seq_len / kGqaSplitDecodeMinKeysPerSplit) of them (capped by the launched count), read
// from total_seq_lens on the device; the remaining CTAs exit at once.
constexpr int kGqaSplitDecodeMinKeysPerSplit = 128;
constexpr int kGqaSplitDecodeCtasPerSm = 1;
constexpr int kGqaSplitDecodeMaxSplits = 64;
constexpr int kGqaSplitDecodeGroup = 8;
constexpr int kGqaSplitDecodeHeadSize = 128;

// Persistent buffers, sized for `batch_size` x `kv_num_heads` (x `max_splits` for the partials):
//   partials: per-split softmax partials (contents need no initialization)
//   counters: one per (batch, kv head); must be zero before the first launch, the kernel leaves them zero.
int GqaSplitDecodeMaxSplits(int batch_size, int kv_num_heads, int capacity, int num_sms);
size_t GqaSplitDecodePartialsBytes(int batch_size, int kv_num_heads, int max_splits);
size_t GqaSplitDecodeCountersBytes(int batch_size, int kv_num_heads);

// Decode-step preparation for the split decode path, replacing UnpackRoPEAppend for one token:
// per (head, batch) one warp loads the head, applies QK-Norm (fp32 statistics, result rounded to
// fp16, as UnpackRoPEAppend does), half-split RoPE over the full head, then writes Q to q_out and
// K/V into the BNSH cache at past_seq_lens[b]. Inputs are either packed_qkv
// [batch, 1, (num_heads + 2 * kv_num_heads) * head_size] or separate q/k/v (packed_qkv == nullptr).
// position_ids [batch] (nullable: past_seq_lens[b] is the position); cos/sin [rope_max_pos, head_size / 2]
// (nullable when do_rotary is false); q_norm/k_norm [head_size] (nullable).
Status LaunchGqaSplitDecodePrep(cudaStream_t stream, const half* packed_qkv, const half* q, const half* k,
                                const half* v, half* q_out, half* k_cache, half* v_cache, const int* past_seq_lens,
                                const int64_t* position_ids, const half* cos_cache, const half* sin_cache,
                                int rope_max_pos, bool do_rotary, const half* q_norm, const half* k_norm, float q_eps,
                                float k_eps,
                                int batch_size, int num_heads, int kv_num_heads, int capacity);

// q:       [batch, 1, num_heads, head_size] (BSNH, rotated/normalized)
// k_cache: [batch, kv_num_heads, capacity, head_size] (BNSH), v_cache likewise
// output:  [batch, 1, num_heads * head_size]
Status LaunchGqaSplitDecode(cudaStream_t stream, const half* q, const half* k_cache, const half* v_cache, half* output,
                            const int* total_seq_lens, float scale, int batch_size, int kv_num_heads, int capacity,
                            int max_splits, float* partials, unsigned int* counters);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
