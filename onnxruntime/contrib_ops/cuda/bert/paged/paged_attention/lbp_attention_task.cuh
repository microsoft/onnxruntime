// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/algorithms.cuh"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/mutex.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/attention_common.cuh"
#include "contrib_ops/cuda/bert/paged/type_convert.cuh"
#include "contrib_ops/cuda/bert/paged/warp_utilities.cuh"

#ifndef TASK_DPRINTF1
#define DUMMY_TASK_DPRINTF1(...)
#define TASK_DPRINTF1 DUMMY_TASK_DPRINTF1
#endif

namespace onnxruntime::contrib::paged {

using namespace cute;

__forceinline__ __device__ void
init_max_sum(float2* max_sum) {
  *max_sum = float2{std::numeric_limits<float>::lowest(), 0.0f};
}

__forceinline__ __device__ void
atomic_init_max_sum(float2* max_sum) {
  union {
    float2 f32x2;
    uint64_t packed;
  };
  f32x2 = float2{std::numeric_limits<float>::lowest(), 0.0f};
  atomicExch(reinterpret_cast<unsigned long long*>(max_sum), packed);
}

template <int TaskChunkSeqLen_, int PageSize_>
struct TaskChunk {
  static constexpr int SeqLen = TaskChunkSeqLen_;
  static constexpr int PageSize = PageSize_;
  static_assert(SeqLen % PageSize == 0);
  static constexpr int NumPages = ceil_div(SeqLen, PageSize);

  // start and end of the chunk id
  int start;
  int end;

  __forceinline__ __device__ int
  start_tok_idx(int global_start_tok_idx) const {
    return max(start * SeqLen, global_start_tok_idx);
  }

  __forceinline__ __device__ int
  end_tok_idx(int global_end_tok_idx) const {
    return min(end * SeqLen, global_end_tok_idx);
  }

  __forceinline__ __device__ int
  start_logical_page_id(int global_start_tok_idx) const {
    constexpr int PageSize = TaskChunk::PageSize;
    return start_tok_idx(global_start_tok_idx) / PageSize;
  }

  __forceinline__ __device__ int
  end_logical_page_id(int global_end_tok_idx) const {
    constexpr int PageSize = TaskChunk::PageSize;
    return ceil_div(end_tok_idx(global_end_tok_idx), PageSize);
  }

  __forceinline__ __device__ bool
  is_valid() const {
    return start < end;
  }

  template <int Offset, typename StagingTensor>
  __forceinline__ __device__ void
  preload_page_table_for_chunk(
      const int* page_table,
      StagingTensor& staging,
      int global_start_tok_idx,
      int global_end_tok_idx
  ) {
    constexpr int NumElemPerThread = size(typename StagingTensor::layout_type{});
    constexpr int PageSize = TaskChunk::PageSize;
    const int lpid_in_chunk = NumElemPerThread * threadIdx.x;
    const int global_end_lpid = ceil_div(global_end_tok_idx, PageSize);

    if (lpid_in_chunk < TaskChunk::NumPages) {
      CUTE_UNROLL
      for (int i = 0; i < NumElemPerThread; i++) {
        const int lpid_to_load = start_logical_page_id(global_start_tok_idx) + lpid_in_chunk + Offset + i;
        staging(i) = lpid_to_load < global_end_lpid ? page_table[lpid_to_load] : -1;
      }
    }
  }

  template <typename StagingTensor>
  __forceinline__ __device__ void
  commit_preloaded_page_table(const StagingTensor& staging, int* chunk_page_table) {
    constexpr int NumElemPerThread = size(typename StagingTensor::layout_type{});
    const int lpid_in_chunk = NumElemPerThread * threadIdx.x;

    if (lpid_in_chunk < TaskChunk::NumPages) {
      CUTE_UNROLL
      for (int i = 0; i < NumElemPerThread; i++) {
        chunk_page_table[lpid_in_chunk + i] = staging[i];
      }
    }
  }
};

template <typename WorkerImpl>
struct IWorker {
  __forceinline__ __device__ void
  take_work(int& chunk_start, int& chunk_end) {
    static_cast<WorkerImpl*>(this)->take_work(chunk_start, chunk_end);
  }

  __forceinline__ __device__ void
  broadcast_work(void* broadcast_buffer, int& chunk_start, int& chunk_end) {
    static_cast<WorkerImpl*>(this)->broadcast_work(broadcast_buffer, chunk_start, chunk_end);
  }
};

template <
    int TaskChunkSeqLen_ = 256,
    bool InplaceFlashAcc_ = true,
    bool UseHeadSeq_ = true,
    bool SingleChunk_ = false,
    int NumQueriesPerCta_ = 1>
struct TaskConfig {
  inline static constexpr int TaskChunkSeqLen = TaskChunkSeqLen_;
  inline static constexpr bool InplaceFlashAcc = InplaceFlashAcc_;
  inline static constexpr bool UseHeadSeq = UseHeadSeq_;
  inline static constexpr bool SingleChunk = SingleChunk_;
  inline static constexpr int NumQueriesPerCta = NumQueriesPerCta_;
};

template <typename T_ = half, int ChunkSize_ = 32>
struct KVConfig {
  using TSB = T_;
  inline static constexpr int ChunkSize = ChunkSize_;
};

struct DefaultKV {
  using TSB = void;
  inline static constexpr int ChunkSize = 32;
};

struct Unused {
  template <typename... Coords>
  __forceinline__ __device__ constexpr Unused
  operator()(const Coords&... coords) const {
    return Unused{};
  }

  template <typename... Coords>
  __forceinline__ __device__ constexpr Unused
  operator()(const Coords&... coords) {
    return Unused{};
  }
};

template <
    int NumThreads,
    int HeadSize,
    int PageSize,
    typename TI,
    typename TO_,
    typename TKV,
    typename Worker,
    typename Config,
    typename KVConfig = DefaultKV>
struct PagedAttentionTask {
  using TQ = TI;
  using TO = TO_;
  using TK = TKV;
  using TV = TKV;
  using TSB_ = typename KVConfig::TSB;

  static constexpr bool HasSB = !std::is_same_v<typename KVConfig::TSB, void>;
  using TSB = std::conditional_t<HasSB, typename KVConfig::TSB, Unused>;

  using TaskChunk = onnxruntime::contrib::paged::TaskChunk<Config::TaskChunkSeqLen, PageSize>;
  using FlashAccCta = onnxruntime::contrib::paged::FlashAccCta<NumThreads, HeadSize>;

  __forceinline__ __device__ static void
  attention(
      void* __restrict__ broadcast_buffer,
      Worker* __restrict__ worker,
      const int seq_idx,
      const int head_idx,
      const int kv_head_idx,
      TO* __restrict__ out,                    // [num_seqs, num_heads, head_size]
      float2* __restrict__ out_max_sum,        // [num_seqs, num_heads], running max and sum of gmem out
      const TI* __restrict__ q,                // [num_seqs, num_heads, head_size]
      const TKV* __restrict__ k_cache,         // [num_pages, num_kv_heads, head_size/x, page_size, x]
      const TKV* __restrict__ v_cache,         // [num_pages, num_kv_heads, head_size, page_size]
      const TSB_* __restrict__ kv_scalebias,   // [num_pages, 2, num_kv_heads, 2, head_size/chunk_size, page_size], optional
      const int* __restrict__ page_table,      // [num_seqs, max_num_pages_per_seq]
      const int* __restrict__ context_lens,    // [num_seqs]
      const float* __restrict__ alibi_slopes,  // [num_heads]
      const float scale,
      const int num_seqs,
      const int num_heads,
      const int num_kv_heads,
      const int max_num_pages_per_seq,
      const int q_stride
  ) {
    // static_assert(Config::NumQueriesPerCta == 1);
    __shared__ int chunk_page_table[ceil_div(Config::TaskChunkSeqLen, PageSize)];
    __shared__ float attn_scores[Config::TaskChunkSeqLen];
    __shared__ float smem_out[HeadSize];
    __shared__ float2 max_sum[2];
    __shared__ TSB sSB_buffer[NumWarps][PageSize * ScaleBiasNumChunks * 2];
    // NOTE: __shared__ float2 smem_max_sum compiles but crash with unaligned access
    float2* smem_max_sum = &max_sum[0];
    float2* chunk_max_sum = &max_sum[1];

    // zero-ing out __shared__ buffers
    // attn_scores will be initialize by softmax_cta
    // smem_out will not be loaded if sum in smem_max_sum is 0.0f
    if (threadIdx.x == 0) {
      *smem_max_sum = float2{std::numeric_limits<float>::lowest(), 0.0f};
    }
    // chunk_max_sum must be initialize per chunk-wise

    const int num_tokens = context_lens[seq_idx];
    const int64_t dummy_shape = 1073741824;  // 2^30, can be used for the slowest dim
    const auto num_pages = dummy_shape;

    auto gO = make_tensor(make_gmem_ptr(out), make_layout(make_shape(num_seqs, num_heads, Int<HeadSize>{}), LayoutRight{}))(_, head_idx, _);                                     // [num_seqs, num_heads, head_size]
    const auto gQ = make_tensor(make_gmem_ptr(q), make_layout(make_shape(num_seqs, num_heads, Int<HeadSize>{}), make_stride(q_stride, Int<HeadSize>{}, _1{})))(_, head_idx, _);  // [num_seqs, num_heads, head_size]
    const auto gK = [&]() {
      auto l_raw = make_layout(make_shape(Int<x>{}, Int<PageSize>{}, Int<HeadSize / x>{}, num_kv_heads, num_pages));
      auto l = group<1, 3>(select<1, 0, 2, 3, 4>(l_raw));  // [page_size, (x * head_size/x), num_kv_heads, num_pages]
      return make_tensor(make_gmem_ptr(k_cache), l)(/*tok_idx_in_page*/ _, /*dim_idx_in_head*/ _, kv_head_idx, /*physical_page_id*/ _);
    }();
    const auto gV = [&]() {
      auto l = select<1, 0, 2, 3>(make_layout(make_shape(Int<PageSize>{}, Int<HeadSize>{}, num_kv_heads, num_pages)));
      return make_tensor(make_gmem_ptr(v_cache), l)(/*dim_idx_in_head*/ _, /*tok_idx_in_page*/ _, kv_head_idx, /*physical_page_id*/ _);
    }();
    const auto gSB = [&]() {
      if constexpr (HasSB) {
        auto l = make_layout(make_shape(Int<PageSize * ScaleBiasNumChunks * 2>{}, num_kv_heads, _2{}, num_pages));
        return make_tensor(make_gmem_ptr(kv_scalebias), l)(/*copy_dim*/ _, kv_head_idx, /*k_or_v*/ _, /*physical_page_id*/ _);
      } else {
        return Unused();
      }
    }();
    // K: (tok_idx_in_page,(dim_idx_in_head),physical_page_id) -> val_idx
    // V: (dim_idx_in_head,  tok_idx_in_page,physical_page_id) -> val_idx

    auto [tSB_should_copy, tSB_copy_src, tSB_copy_dst] = [&]() {
      if constexpr (HasSB) {
        auto sSB = make_tensor(make_smem_ptr(sSB_buffer[warp_id()]), make_layout(Int<PageSize * ScaleBiasNumChunks * 2>{}));
        auto cSB = make_identity_tensor(shape(sSB));
        ScaleBiasWarpCopy tiled_copy{};
        auto thr_copy = tiled_copy.get_thread_slice(lane_id());
        const auto src_view = thr_copy.partition_S(gSB);
        auto dst_view = thr_copy.partition_S(sSB);
        auto coord = thr_copy.partition_S(cSB);
        bool should_copy = elem_less(coord(_0{}, _0{}), shape(cSB));
        static_assert(size<1>(src_view) == 1);
        static_assert(size<1>(dst_view) == 1);
        return std::make_tuple(
            should_copy,
            coalesce(src_view(_, /*iter*/ _0{}, /*k_or_v*/ _, /*physical_page_id*/ _), make_shape(_1{}, _1{}, _1{})),
            coalesce(dst_view(_, /*iter*/ _0{}))
        );
      } else {
        return std::make_tuple(false, Unused(), Unused());
      }
    }();
    auto tSB_copy_staging = make_tensor<TSB>(Int<ScaleBiasCopyValPerThread>{});
    auto sSB = make_tensor(
        make_smem_ptr(sSB_buffer[warp_id()]),
        Layout<Shape<Int<PageSize>, Shape<Int<KVConfig::ChunkSize>, Int<ScaleBiasNumChunks>>, _2>, Stride<_1, Stride<_0, Int<PageSize>>, Int<PageSize * ScaleBiasNumChunks>>>{}
    );

    const auto seq_page_table = &(make_tensor(make_gmem_ptr(page_table), make_layout(make_shape(num_seqs, max_num_pages_per_seq), LayoutRight{}))(seq_idx, _0{}));
    const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

    // cooperative load q to sA (sQ)
    constexpr const auto SmemQLayout = make_layout(Int<HeadSize>{});
    __shared__ TQ sQ_buffer[cosize(SmemQLayout)];

    auto sQ = make_tensor(make_smem_ptr(sQ_buffer), SmemQLayout);
    auto cQ = make_identity_tensor(shape(SmemQLayout));
    {
      constexpr int QLoadVec = cute::min(ceil_div(HeadSize, NumThreads), 8);  // load n elems per thread
      auto load_q_head = make_tv_layout(SmemQLayout, make_layout(Int<NumThreads>{}), make_layout(Int<QLoadVec>{}));

      auto coord = make_coord(make_coord(threadIdx.x, _), _);
      auto thr_ld = gQ(seq_idx, _).compose(load_q_head)(coord);
      auto thr_st = sQ.compose(load_q_head)(coord);
      auto thr_cQ = cQ.compose(load_q_head)(coord);
      CUTE_UNROLL
      for (int i = 0; i < QLoadVec; i++) {
        if (elem_less(thr_cQ(i), shape(cQ))) {
          thr_st(i) = thr_ld(i);
        }
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    }

    const auto gemv1_thr_copy = Gemv1TiledCopy{}.get_thread_slice(lane_id());
    const auto gemv2_thr_copy = Gemv2TiledCopy{}.get_thread_slice(lane_id());

    // gemv1 A
    const auto gemv1_sA = make_tensor(make_smem_ptr(sQ_buffer), make_layout(make_shape(_1{}, Int<HeadSize>{}), make_stride(_0{}, _1{})));  // [1,k], m=1, broadcast
    const auto gemv1_tA_view = gemv1_thr_copy.partition_S(gemv1_sA);                                                                       // (val,j,p)
    // gemv1 B
    const auto gemv1_page_coord = make_identity_tensor(make_shape(Int<PageSize>{}, Int<HeadSize>{}));
    const auto gemv1_tB_view = gemv1_thr_copy.partition_S(gK);                 // (val,j,p,physical_page_id) -> idx
    const auto gemv1_tB_coord = gemv1_thr_copy.partition_S(gemv1_page_coord);  // (val,j,p)                  -> coord
    const auto gemv1_tSB_view = [&]() {
      if constexpr (HasSB) {
        return gemv1_thr_copy.partition_S(sSB);
      } else {
        return Unused();
      }
    }();

    // gemv2 A
    const auto gemv2_sA = make_tensor(make_smem_ptr(attn_scores), make_layout(make_shape(_1{}, Int<PageSize>{}, Int<TaskChunk::NumPages>{})));  // [1,k,lid_in_chunk], m=1
    const auto gemv2_tA_view = gemv2_thr_copy.partition_S(gemv2_sA);                                                                            // (val,i,p,lid_in_chunk), p is token_idx_in page
    // gemv2 B
    const auto gemv2_page_coord = make_identity_tensor(make_shape(Int<HeadSize>{}, Int<PageSize>{}));                  // (n,k)
    const auto gemv2_tB_view = coalesce(gemv2_thr_copy.partition_S(gV), make_shape(_1{}, _1{}, _1{}, _1{}));           // (val,j,p,physical_page_id) -> idx
    const auto gemv2_tB_coord = coalesce(gemv2_thr_copy.partition_S(gemv2_page_coord), make_shape(_1{}, _1{}, _1{}));  // (val,j,p)                  -> coord
    const auto gemv2_tSB_view = [&]() {
      if constexpr (HasSB) {
        auto sSB_gemv2_nk = make_tensor(sSB.data(), select<1, 0, 2>(sSB.layout()));
        return gemv2_thr_copy.partition_S(sSB_gemv2_nk);
      } else {
        return Unused();
      }
    }();
    static_assert(rank(gemv2_tB_view) == 4 && size<2>(gemv2_tB_view) == 1, "iter mode p is assumed to be 1");
    static_assert(rank(gemv2_tB_coord) == 3 && size<2>(gemv2_tB_coord) == 1, "iter mode p is assumed to be 1");

    constexpr int NumPidsPreloadPerThread = ceil_div(TaskChunk::NumPages, NumThreads);
    static_assert(TaskChunk::NumPages <= NumPidsPreloadPerThread * NumThreads);
    auto preload_pids = make_tensor<int>(Int<NumPidsPreloadPerThread>{});

    TaskChunk chunk{-1, -1};
    worker->take_work(chunk.start, chunk.end);
    worker->broadcast_work(broadcast_buffer, chunk.start, chunk.end);
    chunk.preload_page_table_for_chunk<0>(seq_page_table, preload_pids, 0, num_tokens);

    while (chunk.is_valid()) {
      TASK_DPRINTF1(
          "  worker[%d]: work on tok[%d,%d) of seq:%d, head:%d, kv_head:%d\n",
          worker->worker_id(), chunk.start_tok_idx(0), chunk.end_tok_idx(num_tokens), seq_idx, head_idx, kv_head_idx
      );

      if (threadIdx.x == 0) {
        *chunk_max_sum = float2{std::numeric_limits<float>::lowest(), 0.0f};
      }
      chunk.commit_preloaded_page_table(preload_pids, chunk_page_table);
      __syncthreads();

      load_sb_to_reg<0>(tSB_copy_src, tSB_should_copy, chunk_page_table[warp_id()], tSB_copy_staging);
      // output for first GEMV
      float* logits = attn_scores;
      float qk_max = std::numeric_limits<float>::lowest();
#pragma unroll 1
      for (int lid_in_chunk = warp_id(); lid_in_chunk < TaskChunk::NumPages; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];
        const int64_t next_physical_page_id = lid_in_chunk + NumWarps < TaskChunk::NumPages ? chunk_page_table[lid_in_chunk + NumWarps] : -1;
        store_sb_to_smem(tSB_copy_staging, tSB_should_copy, physical_page_id, tSB_copy_dst);        // if curr is valid, store
        load_sb_to_reg<0>(tSB_copy_src, tSB_should_copy, next_physical_page_id, tSB_copy_staging);  // if next is valid, load
        if (physical_page_id == -1) {
          continue;
        }

        const auto tA_view = gemv1_tA_view;
        auto tA = make_tensor_like(tA_view(_, _0{}, _0{}));

        const auto tB_view = gemv1_tB_view(_, _, _, physical_page_id);
        const auto& tSB_view = gemv1_tSB_view;
        static_assert(size<1>(gemv1_tB_view) == 1);  // IterN over PageSize can be omitted
        auto tB = make_fragment_like<TQ>(tB_view(_, _0{}, _0{}));

        float qk{};
        CUTE_UNROLL
        for (int p = 0; p < size<2>(tB_view); p++) {  // IterK over HeadSize
          copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TQ) * size(tA), 128)>{}, tA_view(_, _0{}, p), tA);
          if (elem_less(gemv1_tB_coord(_0{}, _0{}, p), shape(gemv1_page_coord))) {
            load_tB(tB, tB_view(_, _0{}, p), tSB_view(_, _0{}, p, _));
            qk += inner_product<float>(tA, tB);
          }
          schedule_barrier();
        }
        qk *= scale;
        // reduce in thread group to get the full qk
        qk = warp::reduce<Gemv1ThrK, /*Strided=*/Gemv1TransThrLayout>(qk, [](float a, float b) { return a + b; });
        auto tid_in_group = get<1>(Gemv1ThrLayout{}.get_hier_coord(int(threadIdx.x)));
        auto tok_idx_in_page = get<0>(gemv1_tB_coord(_0{}, _0{}, _0{}));
        const int token_idx = (chunk.start_logical_page_id(0) + lid_in_chunk) * PageSize + tok_idx_in_page;
        qk += (alibi_slope != 0) ? alibi_slope * (token_idx - num_tokens + 1) : 0;
        if (tid_in_group == 0) {
          logits[token_idx % Config::TaskChunkSeqLen] = token_idx < num_tokens ? qk : 0.f;  // TODO: start boundary
          qk_max = token_idx < num_tokens ? fmaxf(qk_max, qk) : qk_max;
        }
      }

      if constexpr (!Config::SingleChunk) {
        chunk.preload_page_table_for_chunk<TaskChunk::NumPages>(seq_page_table, preload_pids, 0, num_tokens);
      }

      // reuse broadcast buffer for reduction
      static_assert(2 * NumWarps <= (BROADCAST0_BUFFER_SIZE_IN_BYTES / sizeof(float)));
      int prefix = chunk.start_tok_idx(0) - (chunk.start_tok_idx(0) % Config::TaskChunkSeqLen);
      softmax_cta<NumThreads, Gemv1ThrK, NumWarps, Config::TaskChunkSeqLen>(
          static_cast<float*>(broadcast_buffer), qk_max, logits,
          chunk.start_tok_idx(0) - prefix,
          chunk.end_tok_idx(num_tokens) - prefix,
          chunk_max_sum
      );

      TaskChunk next_chunk{-1, -1};
      if constexpr (!Config::SingleChunk) {
        worker->take_work(next_chunk.start, next_chunk.end);
      }

      load_sb_to_reg<1>(tSB_copy_src, tSB_should_copy, chunk_page_table[warp_id()], tSB_copy_staging);
      auto acc = make_tensor<float>(Int<size<1>(gemv2_tB_view)>{});
      clear(acc);
      int num_pages_in_chunk = chunk.end_logical_page_id(num_tokens) - chunk.start_logical_page_id(0);
#pragma unroll 1
      for (int lid_in_chunk = warp_id(); lid_in_chunk < num_pages_in_chunk; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];
        const int64_t next_physical_page_id = lid_in_chunk + NumWarps < TaskChunk::NumPages ? chunk_page_table[lid_in_chunk + NumWarps] : -1;
        store_sb_to_smem(tSB_copy_staging, tSB_should_copy, physical_page_id, tSB_copy_dst);        // if curr is valid, store
        load_sb_to_reg<1>(tSB_copy_src, tSB_should_copy, next_physical_page_id, tSB_copy_staging);  // if next is valid, load

        const auto tA_view = coalesce(filter_zeros(gemv2_tA_view(_, _, _, lid_in_chunk)));
        const auto tB_view = gemv2_tB_view(_, _, _0{}, physical_page_id);
        const auto tB_coord = gemv2_tB_coord(_, _, _0{});
        const auto tSB_view = [&]() {
          if constexpr (HasSB) {
            return gemv2_tSB_view(_, _, _0{}, _);
          } else {
            return Unused();
          }
        }();

        static_assert(rank(tA_view) == 1);  // tA_view is not related to j
        static_assert(size<1>(gemv2_tB_view) == size(acc));
        static_assert(size<2>(gemv2_tB_view) == 1);  // IterK over PageSize can be omitted

        auto tA = make_fragment_like(tA_view);
        auto tB = make_fragment_like<TQ>(tB_view(_, _0{}));

        bool is_full_chunk = chunk.end_tok_idx(num_tokens) - chunk.start_tok_idx(0) == Config::TaskChunkSeqLen;
        bool is_full_page = lid_in_chunk != 0 && lid_in_chunk != num_pages_in_chunk - 1;
        if (is_full_chunk || is_full_page) {
          copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(float) * Gemv2ValK, 128)>{}, tA_view, tA);
          CUTE_UNROLL
          for (int j = 0; j < size<1>(tB_view); j++) {  // IterN over HeadSize
            if (elem_less(tB_coord(_0{}, j), shape(gemv2_page_coord))) {
              load_tB(tB, tB_view(_, j), tSB_view(_, j, _));
              acc(j) += inner_product<float>(tA, tB);
            }
            schedule_barrier();
          }
        } else {
          enforce_uniform();
          auto token_idx_in_page = get<1>(tB_coord(_0{}, _0{}));
          auto logical_page_id = chunk.start_logical_page_id(0) + lid_in_chunk;
          int valid_tokens = num_tokens - (logical_page_id * PageSize + token_idx_in_page);

          auto pred = make_tensor_like<bool>(tA);
          CUTE_UNROLL
          for (int p = 0; p < size(pred); p++) {
            pred(p) = p < valid_tokens;
          }

          auto masked_tB = make_fragment_like(tB);

          copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(float) * Gemv2ValK, 128)>{}, tA_view, tA);
          CUTE_UNROLL
          for (int j = 0; j < size<1>(tB_view); j++) {  // IterN over HeadSize
            if (elem_less(tB_coord(_0{}, j), shape(gemv2_page_coord))) {
              load_tB(tB, tB_view(_, j), tSB_view(_, j, _));
              copy_if(pred, tB, masked_tB);
              acc(j) += inner_product<float>(tA, masked_tB);
            }
          }
        }
      }

      acc = warp::reduce<Gemv2ThrK>(acc, [](float a, float b) { return a + b; });

      __syncthreads();  // sync to reuse attn_scores
      auto chunk_out = attn_scores;
      static_assert(HeadSize <= Config::TaskChunkSeqLen);

      constexpr auto OutThrCoord = make_identity_tensor(make_shape(Int<Gemv2ThrK>{}, Int<Gemv2ThrN>{}));
      const auto thr_coord = OutThrCoord(lane_id());
      const auto thr_k = get<0>(thr_coord);  // thr_k == 0 is the leading threads in previous warp reduction
      const auto thr_n = get<1>(thr_coord);

      if (warp_id() == 0 && thr_k == 0) {
        CUTE_UNROLL
        for (int v = 0; v < size(acc); v++) {
          int dim_idx = thr_n + v * Gemv2ThrN;  // strided
          if (dim_idx < HeadSize) {
            chunk_out[dim_idx] = acc(v);
          }
        }
      }
      __syncthreads();
      CUTE_UNROLL
      for (int warp = 1; warp < NumWarps; warp++) {
        if (warp_id() == warp && thr_k == 0) {
          CUTE_UNROLL
          for (int v = 0; v < size(acc); v++) {
            int dim_idx = thr_n + v * Gemv2ThrN;  // strided
            if (dim_idx < HeadSize) {
              chunk_out[dim_idx] += acc(v);
            }
          }
        }
        __syncthreads();
      }

      FlashAccCta::acc(smem_out, smem_max_sum, chunk_out, chunk_max_sum);
      if constexpr (Config::SingleChunk) {
        break;
      } else {
        worker->broadcast_work(broadcast_buffer, next_chunk.start, next_chunk.end);
        chunk = next_chunk;
      }
    }

    __syncthreads();
    int max_sum_idx = [&]() {
      if constexpr (Config::UseHeadSeq) {
        return head_idx * Config::MaxNumSeqs + seq_idx;
      } else {
        return seq_idx * num_heads + head_idx;
      }
    }();
    TO* gmem_out = &gO(seq_idx, 0);
    if constexpr (Config::InplaceFlashAcc) {
      FlashAccCta::atomic_acc(broadcast_buffer, gmem_out, &out_max_sum[max_sum_idx], smem_out, smem_max_sum);
    } else {
      FlashAccCta::write_inc(gmem_out, out_max_sum ? &out_max_sum[max_sum_idx] : nullptr, smem_out, smem_max_sum);
    }
  }

protected:
  static constexpr int NumWarps = NumThreads / constant::WarpSize;
  static constexpr int x = 16 / sizeof(TK);

  static constexpr int Gemv1ThrN = PageSize;
  static constexpr int Gemv1ThrK = ceil_div(constant::WarpSize, Gemv1ThrN);
  static constexpr int Gemv1ValN = 1;
  static constexpr int Gemv1ValK = cute::min(next_power_of_two(ceil_div(HeadSize, Gemv1ThrK)), x);
  static_assert(
      Gemv1ThrK == 1 || Gemv1ThrK == 2 || Gemv1ThrK == 4 || Gemv1ThrK == 8 ||
      Gemv1ThrK == 32 || Gemv1ThrK == constant::WarpSize
  );
  static_assert(constant::WarpSize % PageSize == 0);
  static_assert(x % Gemv1ValK == 0);

  static constexpr bool Gemv1TransThrLayout = true;
  using Gemv1ThrLayout = decltype(make_layout(
      make_shape(Int<Gemv1ThrN>{}, Int<Gemv1ThrK>{}),
      std::conditional_t<Gemv1TransThrLayout, LayoutLeft, LayoutRight>{}
  ));
  using Gemv1TiledCopy = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TK) * Gemv1ValK, 128)>, TK>{},
      Gemv1ThrLayout{},
      make_layout(make_shape(Int<Gemv1ValN>{}, Int<Gemv1ValK>{}), LayoutRight{})
  ));

  // LDG is slow enough, so target v cache to use ldg.128
  // loading from logits is LDS (maybe with different datatype)
  static constexpr int Gemv2ValK = cute::min(16 / sizeof(TV), PageSize);
  static constexpr int Gemv2ThrK = PageSize / Gemv2ValK;
  static constexpr int Gemv2ThrN = constant::WarpSize / Gemv2ThrK;
  static constexpr int Gemv2ValN = 1;

  using Gemv2TiledCopy = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TV) * Gemv2ValK, 128)>, TV>{},
      make_layout(make_shape(Int<Gemv2ThrN>{}, Int<Gemv2ThrK>{}), LayoutRight{}),
      make_layout(make_shape(_1{}, Int<Gemv2ValK>{}), LayoutRight{})
  ));

  // (PageSize * ScaleBiasNumChunks,2):(1,?), 2 for scale or bias
  [[maybe_unused]] static constexpr int ScaleBiasNumChunks = ceil_div(HeadSize, KVConfig::ChunkSize);
  // Thr and Val config along ScaleBiasNumChunks(C) and PageSize(P) axis
  [[maybe_unused]] static constexpr int ScaleBiasCopyValPerThread = ceil_div(PageSize * next_power_of_two(ScaleBiasNumChunks) * 2, constant::WarpSize);
  static_assert((PageSize * ScaleBiasNumChunks * 2) % ScaleBiasCopyValPerThread == 0);

  using ScaleBiasWarpCopy = std::conditional_t<
      HasSB,
      decltype(make_tiled_copy(
          Copy_Atom<UniversalCopy<TSB>, TSB>{},
          make_layout(Int<constant::WarpSize>{}),
          make_layout(Int<ScaleBiasCopyValPerThread>{})
      )),
      void>;

  template <int KVIdx, typename STensor, typename DTensor>
  __forceinline__ __device__ static void
  load_sb_to_reg(const STensor& src, bool should_copy, int physical_page_id, DTensor& dst) {
    if constexpr (HasSB) {
      if (should_copy) {
        if (physical_page_id >= 0) {
          copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(128, 8 * sizeof(TSB) * ScaleBiasCopyValPerThread)>{}, src(_, Int<KVIdx>{}, physical_page_id), dst);
        }
      }
    }
  }

  template <typename STensor, typename DTensor>
  __forceinline__ __device__ static void
  store_sb_to_smem(const STensor& src, bool should_copy, int physical_page_id, DTensor& dst) {
    if constexpr (HasSB) {
      if (should_copy) {
        if (physical_page_id >= 0) {
          copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(128, 8 * sizeof(TSB) * ScaleBiasCopyValPerThread)>{}, src, dst);
        }
      }
    }
  }

  template <typename TensorBDst, typename TensorBSrc, typename TensorSB>
  __forceinline__ __device__ static void
  load_tB(TensorBDst& tB, const TensorBSrc& tB_view, const TensorSB& tSB_view) {
    using Src = std::remove_const_t<typename TensorBSrc::element_type>;
    using Dst = std::remove_const_t<typename TensorBDst::element_type>;
    constexpr int NElem = size(typename TensorBDst::layout_type{});

    if constexpr (HasSB) {
      auto tB_copy = make_tensor_like<Src>(tB);
      copy(AutoVectorizingCopyWithAssumedAlignment<8 * sizeof(Src) * NElem>{}, tB_view, tB_copy);

      constexpr auto SBTileLayout = coalesce(get<0>(typename TensorSB::layout_type{}));
      if constexpr (
          rank(SBTileLayout) == 1 && depth(SBTileLayout) == 0 &&  // no hierarchy
          stride(SBTileLayout) == _0{}                            // is broadcasting
      ) {
        auto scale = tSB_view(_0{}, _0{});
        auto bias = tSB_view(_0{}, _1{});
        tensor_convert(tB_copy, scale, bias, tB);
      } else {
        auto tSB = make_tensor<TSB>(make_layout(make_shape(Int<NElem>{}, _2{})));
        copy(AutoVectorizingCopyWithAssumedAlignment<8 * sizeof(TSB)>{}, tSB_view, tSB);
        tensor_convert(tB_copy, tSB(_, _0{}), tSB(_, _1{}), tB);
      }

    } else {
      static_assert(std::is_same_v<Src, Dst>);
      copy(AutoVectorizingCopyWithAssumedAlignment<8 * sizeof(Src) * NElem>{}, tB_view, tB);
    }
  }
};

}  // namespace onnxruntime::contrib::paged

#ifdef DUMMY_TASK_DPRINTF1
#undef DUMMY_TASK_DPRINTF1
#undef TASK_DPRINTF1
#endif
