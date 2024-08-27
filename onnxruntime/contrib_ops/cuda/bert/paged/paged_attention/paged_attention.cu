// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/paged/paged_attention/paged_attention.h"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/paged_attention.cuh"

namespace onnxruntime::contrib::paged {

template <typename TIO, typename TKV, typename TSB, int NumQueriesPerCta>
void launch_paged_attention_kernel(
    stream_t stream,
    dev_props_ptr dev_props,
    TIO* out_ptr,
    const TIO* q_ptr,
    const TKV* k_cache_ptr,
    const TKV* v_cache_ptr,
    const TSB* scalebias_ptr,
    const int* page_table_ptr,
    const int* context_lens_ptr,
    const float* alibi_slopes_ptr,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int page_size,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len,
    const int num_queries_per_kv
) {
  constexpr int NumThreads = 128;
  constexpr int ChunkSize = 32;  // only useful for fp8

  if constexpr (std::is_same_v<TSB, void>) {
    if (scalebias_ptr != nullptr) {
      throw std::runtime_error("scalebias is not supported by this kernel");
    }
  }

  if (num_queries_per_kv % NumQueriesPerCta != 0) {
    throw std::runtime_error(
        std::string("Unsupported NumQueriesPerCta ") + std::to_string(NumQueriesPerCta) +
        " for num_queries_per_kv " + std::to_string(num_queries_per_kv)
    );
  }

#define LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NUM_THREADS, HEAD_SIZE, PAGE_SIZE, CHUNK_SIZE, NUM_QUERIES_PER_CTA, TIO, TKV, TSB) \
  paged_attention_kernel<NUM_THREADS, HEAD_SIZE, PAGE_SIZE, CHUNK_SIZE, NUM_QUERIES_PER_CTA, TIO, TKV, TSB>                        \
      <<<(num_heads / NUM_QUERIES_PER_CTA) * num_seqs, NumThreads, 0, stream>>>(                                                   \
          out_ptr,                                                                                                                 \
          q_ptr,                                                                                                                   \
          k_cache_ptr,                                                                                                             \
          v_cache_ptr,                                                                                                             \
          scalebias_ptr,                                                                                                           \
          page_table_ptr,                                                                                                          \
          context_lens_ptr,                                                                                                        \
          alibi_slopes_ptr,                                                                                                        \
          scale,                                                                                                                   \
          num_seqs,                                                                                                                \
          num_heads,                                                                                                               \
          num_kv_heads,                                                                                                            \
          max_num_pages_per_seq,                                                                                                   \
          q_stride,                                                                                                                \
          max_context_len                                                                                                          \
      );                                                                                                                           \
  break;

  do {
    if (page_size == 32) {
      if (head_size == 64) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 64, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 80) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 80, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 96) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 96, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 112) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 112, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 128) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 128, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 256) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 256, 32, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else if (page_size == 16) {
      if (head_size == 64) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 64, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 80) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 80, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 96) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 96, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 112) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 112, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 128) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 128, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 256) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 256, 16, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else if (page_size == 8) {
      if (head_size == 64) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 64, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 80) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 80, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 96) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 96, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 112) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 112, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 128) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 128, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else if (head_size == 256) {
        LAUNCH_PAGED_ATTENTION_KERNEL_AND_BREAK(NumThreads, 256, 8, ChunkSize, NumQueriesPerCta, TIO, TKV, TSB);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else {
      throw std::runtime_error(std::string("Unsupported page size: ") + std::to_string(page_size));
    }
  } while (0);
}

template <typename TIO, typename TKV, typename TSB>
void launch_paged_attention_kernel(
    stream_t stream,
    dev_props_ptr dev_props,
    TIO* out_ptr,
    const TIO* q_ptr,
    const TKV* k_cache_ptr,
    const TKV* v_cache_ptr,
    const TSB* scalebias_ptr,
    const int* page_table_ptr,
    const int* context_lens_ptr,
    const float* alibi_slopes_ptr,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int page_size,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len
) {
  int max_num_queries_per_kv = num_heads / num_kv_heads;

#define LAUNCH_AND_RETURN(NUM_QUERIES_PER_KV)                              \
  launch_paged_attention_kernel<TIO, TKV, TSB, NUM_QUERIES_PER_KV>(        \
      stream, dev_props,                                                   \
      out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, scalebias_ptr,             \
      page_table_ptr, context_lens_ptr, alibi_slopes_ptr,                  \
      scale, num_seqs, num_heads, num_kv_heads, head_size, page_size,      \
      max_num_pages_per_seq, q_stride, max_context_len, NUM_QUERIES_PER_KV \
  );                                                                       \
  return;

  if (max_num_queries_per_kv % 8 == 0 && !std::is_same_v<TKV, float>) {
    LAUNCH_AND_RETURN(8);
  }
  if (max_num_queries_per_kv % 4 == 0) {
    LAUNCH_AND_RETURN(4);
  }
  LAUNCH_AND_RETURN(1);
#undef LAUNCH_AND_RETURN
}

template void launch_paged_attention_kernel<float, float, void>(
    stream_t stream,
    dev_props_ptr dev_props,
    float* out_ptr,
    const float* q_ptr,
    const float* k_cache_ptr,
    const float* v_cache_ptr,
    const void* scalebias_ptr,
    const int* page_table_ptr,
    const int* context_lens_ptr,
    const float* alibi_slopes_ptr,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int page_size,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len
);

template void launch_paged_attention_kernel<half, half, void>(
    stream_t stream,
    dev_props_ptr dev_props,
    half* out_ptr,
    const half* q_ptr,
    const half* k_cache_ptr,
    const half* v_cache_ptr,
    const void* scalebias_ptr,
    const int* page_table_ptr,
    const int* context_lens_ptr,
    const float* alibi_slopes_ptr,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int page_size,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len
);

template void launch_paged_attention_kernel<half, cute::float_e4m3_t, half>(
    stream_t stream,
    dev_props_ptr dev_props,
    half* out_ptr,
    const half* q_ptr,
    const cute::float_e4m3_t* k_cache_ptr,
    const cute::float_e4m3_t* v_cache_ptr,
    const half* scalebias_ptr,
    const int* page_table_ptr,
    const int* context_lens_ptr,
    const float* alibi_slopes_ptr,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int page_size,
    const int max_num_pages_per_seq,
    const int q_stride,
    const int max_context_len
);

}  // namespace onnxruntime::contrib::paged
