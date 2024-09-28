// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention.h"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_dp.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_dpo_reduction.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_ws.cuh"

namespace onnxruntime::contrib::paged {

struct DataParallelOutOfPlace {};
struct DataParallelInPlace {};
struct WorkStealing {};

namespace detail {

template <typename TIO, typename TKV, typename TSB, typename Sch, int NumQueriesPerCta>
struct LBPAttentionKernel {
  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
      TIO* out_ptr,
      const TIO* q_ptr,
      const TKV* k_cache_ptr,
      const TKV* v_cache_ptr,
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

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int head_size, int max_context_len);
  static void destroy_workspace(stream_t stream, void* workspace);
  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len);
};

template <typename TIO, typename TKV, typename TSB, int NumQueriesPerCta>
struct LBPAttentionKernel<TIO, TKV, TSB, WorkStealing, NumQueriesPerCta> {
  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
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
    constexpr int NumThreads = 128;
    constexpr int NumCtaPerProcessor = 2;
    dim3 grid(NumCtaPerProcessor * dev_props->multiProcessorCount);
    dim3 block(NumThreads);
#define LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(HEAD_SIZE, PAGE_SIZE)     \
  lbp_attention_work_stealing_kernel<NumThreads, HEAD_SIZE, PAGE_SIZE, TIO, TKV, TSB> \
      <<<grid, block, 0, stream>>>(                                                   \
          workspace,                                                                  \
          out_ptr,                                                                    \
          q_ptr,                                                                      \
          k_cache_ptr,                                                                \
          v_cache_ptr,                                                                \
          scalebias_ptr,                                                              \
          page_table_ptr,                                                             \
          context_lens_ptr,                                                           \
          alibi_slopes_ptr,                                                           \
          scale,                                                                      \
          num_seqs,                                                                   \
          num_heads,                                                                  \
          num_kv_heads,                                                               \
          max_num_pages_per_seq,                                                      \
          q_stride                                                                    \
      );                                                                              \
  break;

    // TODO: zero-ing out workspace

    do {
      if (page_size == 32) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(64, 32);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(80, 32);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(96, 32);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(112, 32);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(128, 32);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(256, 32);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 16) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(64, 16);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(80, 16);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(96, 16);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(112, 16);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(128, 16);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(256, 16);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 8) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(64, 8);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(80, 8);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(96, 8);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(112, 8);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(128, 8);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_WORK_STEALING_KERNEL_AND_BREAK(256, 8);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else {
        throw std::runtime_error(std::string("Unsupported page size: ") + std::to_string(page_size));
      }
    } while (0);
  }

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int head_size, int max_context_len) {
    // TODO: workspace logic leaked, move to ws header
    size_t total_bytes = sizeof(WorkStealingWorkspace);
    if (size != nullptr) {
      *size = total_bytes;
    }
    if (workspace != nullptr) {
      CUDA_CHECK(cudaMallocAsync(workspace, total_bytes, stream));
    }
  }

  static void destroy_workspace(stream_t stream, void* workspace) {
    CUDA_CHECK(cudaFreeAsync(workspace, stream));
  }

  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len) {
    auto* ws = reinterpret_cast<WorkStealingWorkspace*>(workspace);
    size_t size_in_bytes = reinterpret_cast<char*>(&ws->max_sum_) - reinterpret_cast<char*>(ws);
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, size_in_bytes, stream));
  }
};

template <typename TIO, typename TKV, typename TSB, int NumQueriesPerCta>
struct LBPAttentionKernel<TIO, TKV, TSB, DataParallelOutOfPlace, NumQueriesPerCta> {
  using Config = DataParallelConfig<false, false, NumQueriesPerCta>;
  using Config1 = DataParallelConfig<false, false, 1>;
  using Workspace = DataParallelWorkspace<Config>;

  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
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
    constexpr int NumThreads = 128;

    const int num_queries_per_kv = num_heads / num_kv_heads;
    if (num_queries_per_kv % NumQueriesPerCta != 0) {
      throw std::runtime_error(
          std::string("Unsupported NumQueriesPerCta ") + std::to_string(NumQueriesPerCta) +
          " for num_queries_per_kv " + std::to_string(num_queries_per_kv)
      );
    }

    int max_task_chunks = ceil_div(max_context_len, Config::TaskChunkSeqLen);
    dim3 grid((num_heads / NumQueriesPerCta) * num_seqs, max_task_chunks);
    dim3 block(NumThreads);
#define LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(HEAD_SIZE, PAGE_SIZE)             \
  lbp_attention_data_parallel_kernel<NumThreads, HEAD_SIZE, PAGE_SIZE, TIO, TKV, TSB, Config> \
      <<<grid, block, 0, stream>>>(                                                           \
          workspace,                                                                          \
          out_ptr,                                                                            \
          q_ptr,                                                                              \
          k_cache_ptr,                                                                        \
          v_cache_ptr,                                                                        \
          scalebias_ptr,                                                                      \
          page_table_ptr,                                                                     \
          context_lens_ptr,                                                                   \
          alibi_slopes_ptr,                                                                   \
          scale,                                                                              \
          num_seqs,                                                                           \
          num_heads,                                                                          \
          num_kv_heads,                                                                       \
          max_num_pages_per_seq,                                                              \
          q_stride,                                                                           \
          max_context_len                                                                     \
      );                                                                                      \
  launch_lbp_attention_reduction_kernel<NumThreads, HEAD_SIZE, TIO, Config1>(                 \
      stream, dev_props,                                                                      \
      workspace,                                                                              \
      out_ptr,                                                                                \
      context_lens_ptr,                                                                       \
      num_seqs,                                                                               \
      num_heads,                                                                              \
      max_context_len                                                                         \
  );                                                                                          \
  break;

    do {
      if (page_size == 32) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 32);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 32);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 32);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 32);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 32);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 32);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 16) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 16);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 16);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 16);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 16);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 16);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 16);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 8) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 8);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 8);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 8);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 8);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 8);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 8);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else {
        throw std::runtime_error(std::string("Unsupported page size: ") + std::to_string(page_size));
      }
    } while (0);
#undef LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK
  }

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int head_size, int max_context_len) {
    DataParallelWorkspace<Config>::create(stream, workspace, size, num_seqs, num_heads, head_size, max_context_len);
  }

  static void destroy_workspace(stream_t stream, void* workspace) {
    DataParallelWorkspace<Config>::destroy(stream, workspace);
  }

  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len) {
    DataParallelWorkspace<Config>::init(stream, workspace, num_seqs, num_heads, head_size, max_context_len);
  }
};

template <typename TIO, typename TKV, typename TSB, int NumQueriesPerCta>
struct LBPAttentionKernel<TIO, TKV, TSB, DataParallelInPlace, NumQueriesPerCta> {
  using Config = DataParallelConfig<true, true>;
  using Workspace = DataParallelWorkspace<Config>;

  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
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
    constexpr int NumThreads = 128;
    int max_task_chunks = ceil_div(max_context_len, Config::TaskChunkSeqLen);

    const int num_queries_per_kv = num_heads / num_kv_heads;
    if (num_queries_per_kv % NumQueriesPerCta != 0) {
      throw std::runtime_error(
          std::string("Unsupported NumQueriesPerCta ") + std::to_string(NumQueriesPerCta) +
          " for num_queries_per_kv " + std::to_string(num_queries_per_kv)
      );
    }

    dim3 grid((num_heads / NumQueriesPerCta) * num_seqs, max_task_chunks);
    dim3 block(NumThreads);
#define LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(HEAD_SIZE, PAGE_SIZE)             \
  lbp_attention_data_parallel_kernel<NumThreads, HEAD_SIZE, PAGE_SIZE, TIO, TKV, TSB, Config> \
      <<<grid, block, 0, stream>>>(                                                           \
          workspace,                                                                          \
          out_ptr,                                                                            \
          q_ptr,                                                                              \
          k_cache_ptr,                                                                        \
          v_cache_ptr,                                                                        \
          scalebias_ptr,                                                                      \
          page_table_ptr,                                                                     \
          context_lens_ptr,                                                                   \
          alibi_slopes_ptr,                                                                   \
          scale,                                                                              \
          num_seqs,                                                                           \
          num_heads,                                                                          \
          num_kv_heads,                                                                       \
          max_num_pages_per_seq,                                                              \
          q_stride,                                                                           \
          max_context_len                                                                     \
      );                                                                                      \
  break;

    do {
      if (page_size == 32) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 32);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 32);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 32);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 32);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 32);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 32);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 16) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 16);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 16);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 16);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 16);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 16);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 16);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else if (page_size == 8) {
        if (head_size == 64) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(64, 8);
        } else if (head_size == 80) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(80, 8);
        } else if (head_size == 96) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(96, 8);
        } else if (head_size == 112) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(112, 8);
        } else if (head_size == 128) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(128, 8);
        } else if (head_size == 256) {
          LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK(256, 8);
        } else {
          throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
        }
      } else {
        throw std::runtime_error(std::string("Unsupported page size: ") + std::to_string(page_size));
      }
    } while (0);
#undef LAUNCH_LBP_ATTENTION_DATA_PARALLEL_KERNEL_AND_BREAK
  }

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int head_size, int max_context_len) {
    DataParallelWorkspace<Config>::create(stream, workspace, size, num_seqs, num_heads, head_size, max_context_len);
  }

  static void destroy_workspace(stream_t stream, void* workspace) {
    DataParallelWorkspace<Config>::destroy(stream, workspace);
  }

  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int head_size, int max_context_len) {
    DataParallelWorkspace<Config>::init(stream, workspace, num_seqs, num_heads, head_size, max_context_len);
  }
};

}  // namespace detail

template <typename TIO, typename TKV, typename TSB, typename Sch>
struct LBPAttentionKernel {
  using LBPAttentionKernelImpl1 = detail::LBPAttentionKernel<TIO, TKV, TSB, Sch, 1>;
  using LBPAttentionKernelImpl4 = detail::LBPAttentionKernel<TIO, TKV, TSB, Sch, 4>;
  using LBPAttentionKernelImpl8 = detail::LBPAttentionKernel<TIO, TKV, TSB, Sch, 8>;

  inline static bool is_gqa8_supported(int num_heads, int num_kv_heads) {
    int max_num_queries_per_kv = num_heads / num_kv_heads;
    return max_num_queries_per_kv % 8 == 0 && !std::is_same_v<TKV, float>;
  }

  inline static bool is_gqa4_supported(int num_heads, int num_kv_heads) {
    int max_num_queries_per_kv = num_heads / num_kv_heads;
    return max_num_queries_per_kv % 4 == 0;
  }

  inline static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
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
#define LAUNCH_AND_RETURN(NUM_QUERIES_PER_KV)                         \
  LBPAttentionKernelImpl##NUM_QUERIES_PER_KV::launch(                 \
      stream, dev_props, workspace,                                   \
      out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, scalebias_ptr,        \
      page_table_ptr, context_lens_ptr, alibi_slopes_ptr,             \
      scale, num_seqs, num_heads, num_kv_heads, head_size, page_size, \
      max_num_pages_per_seq, q_stride, max_context_len                \
  );                                                                  \
  return;

    if (is_gqa8_supported(num_heads, num_kv_heads)) {
      LAUNCH_AND_RETURN(8);
    }
    if (is_gqa4_supported(num_heads, num_kv_heads)) {
      LAUNCH_AND_RETURN(4);
    }
    LAUNCH_AND_RETURN(1);
#undef LAUNCH_AND_RETURN
  }

  inline static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len) {
    if (is_gqa8_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl8::create_workspace(stream, workspace, size, num_seqs, num_heads, head_size, max_context_len);
      return;
    }
    if (is_gqa4_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl4::create_workspace(stream, workspace, size, num_seqs, num_heads, head_size, max_context_len);
      return;
    }
    return LBPAttentionKernelImpl1::create_workspace(stream, workspace, size, num_seqs, num_heads, head_size, max_context_len);
  }

  inline static void destroy_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len) {
    if (is_gqa8_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl8::destroy_workspace(stream, workspace);
      return;
    }
    if (is_gqa4_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl4::destroy_workspace(stream, workspace);
      return;
    }
    return LBPAttentionKernelImpl1::destroy_workspace(stream, workspace);
  }

  inline static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len) {
    if (is_gqa8_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl8::init_workspace(stream, workspace, num_seqs, num_heads, head_size, max_context_len);
      return;
    }
    if (is_gqa4_supported(num_heads, num_kv_heads)) {
      LBPAttentionKernelImpl4::init_workspace(stream, workspace, num_seqs, num_heads, head_size, max_context_len);
      return;
    }
    return LBPAttentionKernelImpl1::init_workspace(stream, workspace, num_seqs, num_heads, head_size, max_context_len);
  }
};

// template class LBPAttentionKernel<float, float, void, DataParallelOutOfPlace>;
template class LBPAttentionKernel<half, half, void, DataParallelOutOfPlace>;
template class LBPAttentionKernel<half, cute::float_e4m3_t, half, DataParallelOutOfPlace>;

// template class LBPAttentionKernel<float, float, void, DataParallelInPlace>;
template class LBPAttentionKernel<half, half, void, DataParallelInPlace>;
template class LBPAttentionKernel<half, cute::float_e4m3_t, half, DataParallelInPlace>;

// template class LBPAttentionKernel<float, float, void, WorkStealing>;
template class LBPAttentionKernel<half, half, void, WorkStealing>;
template class LBPAttentionKernel<half, cute::float_e4m3_t, half, WorkStealing>;

}  // namespace onnxruntime::contrib::paged
