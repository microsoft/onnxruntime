// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/flash_mla_latent.h"

#include <algorithm>

#include "core/providers/cuda/cuda_common.h"

#if defined(ORT_ENABLE_FLASH_MLA)
#include "contrib_ops/cuda/bert/flash_mla/flash_mla.h"
#include "cutlass/numeric_types.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// FlashMLA hardcodes a 64-token page, a 576-wide latent row and a 512-wide value.
constexpr int kFlashMlaPageSize = 64;
constexpr int kFlashMlaHeadSize = 576;
constexpr int kFlashMlaVHeadSize = 512;

// Matches upstream's get_mla_metadata caller. It is the number of pages the scheduler pads its
// per-part work estimate by, so that a part which lands on a sequence boundary is not starved.
constexpr int kFixedOverheadNumBlocks = 5;

constexpr size_t kAlign = 256;

size_t AlignUp(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// Byte offsets of each slice of the workspace. Kept in one place so the sizing function and the
// launcher cannot disagree.
struct WorkspaceLayout {
  size_t seqlens_k;     // int[batch]
  size_t tile_md;       // int[num_sm_parts * TileSchedulerMetaDataSize]
  size_t num_splits;    // int[batch + 1]
  size_t lse;           // float[batch * seqlen_q * num_heads]
  size_t lse_accum;     // float[(batch + num_sm_parts) * num_heads * seqlen_q]
  size_t o_accum;       // float[(batch + num_sm_parts) * num_heads * seqlen_q * v_head_size]
  size_t total;
};

WorkspaceLayout ComputeWorkspaceLayout(const FlashMlaLatentConfig& c) {
#if defined(ORT_ENABLE_FLASH_MLA)
  constexpr int kMetaDataSize = TileSchedulerMetaDataSize;
#else
  constexpr int kMetaDataSize = 8;
#endif
  const size_t rows = static_cast<size_t>(c.batch_size + c.num_sm_parts) * c.num_heads * c.seqlen_q;
  WorkspaceLayout w{};
  size_t offset = 0;
  w.seqlens_k = offset;
  offset += AlignUp(sizeof(int) * static_cast<size_t>(c.batch_size));
  w.tile_md = offset;
  offset += AlignUp(sizeof(int) * static_cast<size_t>(c.num_sm_parts) * kMetaDataSize);
  w.num_splits = offset;
  offset += AlignUp(sizeof(int) * static_cast<size_t>(c.batch_size + 1));
  w.lse = offset;
  offset += AlignUp(sizeof(float) * static_cast<size_t>(c.batch_size) * c.seqlen_q * c.num_heads);
  w.lse_accum = offset;
  offset += AlignUp(sizeof(float) * rows);
  w.o_accum = offset;
  offset += AlignUp(sizeof(float) * rows * static_cast<size_t>(c.v_head_size));
  w.total = offset;
  return w;
}

// FlashMLA wants the TOTAL KV length per sequence, while ORT carries the length as of the start of
// the step. One tiny kernel rather than a host round trip, because the value has to be correct
// under CUDA graph replay where the host cannot see it.
__global__ void BuildSeqlensKKernel(int* seqlens_k, const int* past_seqlens, int batch_size, int seqlen_q) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) {
    seqlens_k[b] = past_seqlens[b] + seqlen_q;
  }
}

}  // namespace

bool FlashMlaLatentSupported(const FlashMlaLatentConfig& config, bool is_bf16, bool cache_is_unquantized,
                             bool has_head_sink, bool has_kv_indices, float softcap, int local_window_size,
                             int sm_major) {
#if !defined(ORT_ENABLE_FLASH_MLA)
  ORT_UNUSED_PARAMETER(config);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(cache_is_unquantized);
  ORT_UNUSED_PARAMETER(has_head_sink);
  ORT_UNUSED_PARAMETER(has_kv_indices);
  ORT_UNUSED_PARAMETER(softcap);
  ORT_UNUSED_PARAMETER(local_window_size);
  ORT_UNUSED_PARAMETER(sm_major);
  return false;
#else
  // Only the bf16 headdim-576 instantiation is compiled in, and it is Hopper-only.
  if (!is_bf16 || sm_major != 9) {
    return false;
  }
  // A quantized cache would need the descale pointers and the fp8 instantiation.
  if (!cache_is_unquantized) {
    return false;
  }
  if (config.head_size != kFlashMlaHeadSize || config.v_head_size != kFlashMlaVHeadSize) {
    return false;
  }
  // The page size is baked into the kernel's TMA descriptors, and the value-is-a-prefix-of-the-key
  // trick only holds for a single latent KV head.
  if (config.block_size != kFlashMlaPageSize || config.kv_num_heads != 1) {
    return false;
  }
  // FlashMLA indexes Q as a dense [batch, seqlen_q, ngroups, head_size], so every sequence in the
  // batch must contribute the same number of query tokens. Decode and MTP verify both do.
  if (config.batch_size <= 0 || config.seqlen_q <= 0 || config.num_heads <= 0) {
    return false;
  }
  // Features the kernel has no equivalent for. Each of these would be silently ignored rather than
  // rejected, so they must be screened here.
  if (has_head_sink || has_kv_indices || softcap != 0.f || local_window_size > 0) {
    return false;
  }
  return true;
#endif
}

int ComputeFlashMlaNumSmParts(int seqlen_q, int num_heads, int kv_num_heads, int multi_processor_count) {
  if (kv_num_heads <= 0 || multi_processor_count <= 0) {
    return 1;
  }
  // Query tiles this step produces per KV head. The kernel's tile is 64 rows of the flattened
  // (token, head) axis, which is why the 64 is not the page size despite matching it.
  const int per_head_k = seqlen_q * num_heads / kv_num_heads;
  const int query_tiles = std::max(1, (per_head_k + 63) / 64);
  return std::max(1, multi_processor_count / kv_num_heads / query_tiles);
}

size_t GetFlashMlaLatentWorkspaceBytes(const FlashMlaLatentConfig& config) {
  return ComputeWorkspaceLayout(config).total;
}

template <typename T>
Status LaunchFlashMlaLatentAttention(const T* query, const T* key_cache, const int* past_seqlens,
                                     const int* block_table, T* output, void* workspace,
                                     const FlashMlaLatentConfig& config, cudaStream_t stream) {
#if !defined(ORT_ENABLE_FLASH_MLA)
  ORT_UNUSED_PARAMETER(query);
  ORT_UNUSED_PARAMETER(key_cache);
  ORT_UNUSED_PARAMETER(past_seqlens);
  ORT_UNUSED_PARAMETER(block_table);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(workspace);
  ORT_UNUSED_PARAMETER(config);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FlashMLA was not compiled into this build.");
#else
  static_assert(sizeof(T) == 2, "FlashMLA is instantiated for bf16 only.");

  const WorkspaceLayout w = ComputeWorkspaceLayout(config);
  auto* base = reinterpret_cast<uint8_t*>(workspace);
  int* seqlens_k = reinterpret_cast<int*>(base + w.seqlens_k);
  int* tile_md = reinterpret_cast<int*>(base + w.tile_md);
  int* num_splits = reinterpret_cast<int*>(base + w.num_splits);

  {
    constexpr int kThreads = 128;
    const int blocks = (config.batch_size + kThreads - 1) / kThreads;
    BuildSeqlensKKernel<<<blocks, kThreads, 0, stream>>>(seqlens_k, past_seqlens, config.batch_size,
                                                         config.seqlen_q);
  }

  Mla_metadata_params meta{};
  meta.seqlens_k_ptr = seqlens_k;
  meta.tile_scheduler_metadata_ptr = tile_md;
  meta.num_splits_ptr = num_splits;
  meta.batch_size = config.batch_size;
  meta.block_size_n = config.block_size;
  meta.fixed_overhead_num_blocks = kFixedOverheadNumBlocks;
  meta.num_sm_parts = config.num_sm_parts;
  get_mla_metadata_func(meta, stream);

  // Strides are in elements, not bytes. The query heads become FlashMLA's `ngroups`, so the
  // flattened query axis is seqlen_q * num_heads and `h` (the KV head count) is 1.
  const int64_t page_stride = static_cast<int64_t>(config.block_size) * config.head_size;

  Flash_fwd_mla_params p{};
  p.b = config.batch_size;
  p.seqlen_q = config.seqlen_q * config.num_heads;
  p.d = config.head_size;
  p.d_v = config.v_head_size;
  p.h = 1;
  p.h_h_k_ratio = 1;
  p.ngroups = config.num_heads;
  p.is_causal = true;
  p.scale_softmax = config.scale;
  // log2(e), spelled out rather than taken from M_LOG2E, which MSVC only defines with
  // _USE_MATH_DEFINES. The kernel exponentiates base 2, so it wants the scale pre-multiplied.
  p.scale_softmax_log2 = config.scale * 1.4426950408889634f;
  p.cu_seqlens_k = seqlens_k;  // despite the name, upstream reads this as a per-sequence length

  p.q_ptr = const_cast<T*>(query);
  p.k_ptr = const_cast<T*>(key_cache);
  p.v_ptr = const_cast<T*>(key_cache);  // the value is the leading d_v channels of the same row
  p.o_ptr = output;
  p.softmax_lse_ptr = base + w.lse;

  p.q_batch_stride = static_cast<int64_t>(config.head_size) * config.num_heads * config.seqlen_q;
  p.k_batch_stride = page_stride;  // indexed through block_table, so this is the per-page stride
  p.v_batch_stride = page_stride;
  p.o_batch_stride = static_cast<int64_t>(config.v_head_size) * config.num_heads * config.seqlen_q;
  p.q_row_stride = config.head_size;
  p.k_row_stride = config.head_size;
  p.v_row_stride = config.head_size;
  p.o_row_stride = config.v_head_size;
  p.q_head_stride = config.head_size;
  p.k_head_stride = config.head_size;
  p.v_head_stride = config.head_size;

  p.block_table = const_cast<int*>(block_table);
  p.block_table_batch_stride = config.max_num_blocks_per_seq;
  p.page_block_size = config.block_size;

  p.tile_scheduler_metadata_ptr = tile_md;
  p.num_sm_parts = config.num_sm_parts;
  p.num_splits_ptr = num_splits;
  p.softmax_lseaccum_ptr = base + w.lse_accum;
  p.oaccum_ptr = base + w.o_accum;

  run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, cutlass::bfloat16_t, kFlashMlaHeadSize>(p, stream);
  return CUDA_CALL(cudaGetLastError());
#endif
}

template Status LaunchFlashMlaLatentAttention<onnxruntime::BFloat16>(
    const onnxruntime::BFloat16* query, const onnxruntime::BFloat16* key_cache, const int* past_seqlens,
    const int* block_table, onnxruntime::BFloat16* output, void* workspace, const FlashMlaLatentConfig& config,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
