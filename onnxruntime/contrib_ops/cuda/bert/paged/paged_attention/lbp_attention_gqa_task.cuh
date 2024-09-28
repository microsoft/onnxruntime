// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/algorithms.cuh"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/attention_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_task.cuh"
#include "contrib_ops/cuda/bert/paged/type_convert.cuh"
#include "contrib_ops/cuda/bert/paged/warp_utilities.cuh"

#ifndef TASK_DPRINTF1
#define DUMMY_TASK_DPRINTF1(...)
#define TASK_DPRINTF1 DUMMY_TASK_DPRINTF1
#endif

namespace onnxruntime::contrib::paged {

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
struct PagedGroupQueryAttentionTask : public PagedAttentionTask<NumThreads, HeadSize, PageSize, TI, TO_, TKV, Worker, Config, KVConfig> {
  using BaseTask = PagedAttentionTask<NumThreads, HeadSize, PageSize, TI, TO_, TKV, Worker, Config, KVConfig>;
  using TQ = TI;
  using TO = TO_;
  using TK = TKV;
  using TV = TKV;
  using TSB_ = typename BaseTask::TSB_;

  using BaseTask::HasSB;
  using TSB = typename BaseTask::TSB;

  using TaskChunk = onnxruntime::contrib::paged::TaskChunk<Config::TaskChunkSeqLen, PageSize>;
  using FlashAccCta = onnxruntime::contrib::paged::FlashAccCta<NumThreads, HeadSize>;
  using FlashAccWarp = onnxruntime::contrib::paged::FlashAccWarp<NumThreads, HeadSize>;

  template <
      typename TensorInEngine, typename TensorInLayout,
      typename TensorOutEngine, typename TensorOutLayout>
  __forceinline__ __device__ static void
  inplace_convert_attn_scores(
      const Tensor<TensorInEngine, TensorInLayout>& logits,
      Tensor<TensorOutEngine, TensorOutLayout>& attn_scores
  ) {
    using TLogits = typename TensorInEngine::element_type;
    using TScores = typename TensorOutEngine::element_type;
    if constexpr (!std::is_same_v<TQ, float>) {
      const auto in = coalesce_tensor(logits);
      auto out = coalesce_tensor(attn_scores);
      static_assert(rank(in) == 1 && rank(out) == 1);  // can be viewed as 1d contiguous

      static_assert(size(TensorInLayout{}) % NumThreads == 0);
      constexpr int NumElemsPerThread = size(TensorInLayout{}) / NumThreads;
      constexpr int NumIters = ceil_div(NumElemsPerThread, 64);  // Max NumElemsPerThreadPerIter is 64
      constexpr int Val = NumElemsPerThread / NumIters;          // NumElemsPerThreadPerIter

      const auto tiled_copy = make_tiled_copy(
          Copy_Atom<UniversalCopy<TQ>, TQ>{},  // dummy
          make_layout(Int<NumThreads>{}),
          make_layout(Int<Val>{})
      );
      const auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
      const auto ld = thr_copy.partition_S(in);
      auto st = thr_copy.partition_D(out);
      auto regs = make_tensor<TLogits>(make_layout(size<0>(ld)));
      auto cvts = make_tensor_like<TScores>(regs);

      CUTE_UNROLL
      for (int iter = 0; iter < size<1>(ld); iter++) {
        copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TLogits) * Val, 128)>{}, ld(_, iter), regs);
        CUTE_UNROLL
        for (int i = 0; i < size(regs); i++) {
          cvts(i) = type_convert<TO>(regs(i));
        }
        __syncthreads();  // ensure load because we are doing conversion inplace
        copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TLogits) * Val, 128)>{}, cvts, st(_, iter));
      }
      __syncthreads();
    }
  }

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
    __shared__ int chunk_page_table[ceil_div(Config::TaskChunkSeqLen, PageSize)];
    __shared__ float logits_or_attn_scores_buffer[Config::NumQueriesPerCta * Config::TaskChunkSeqLen];
    // taking account other smem, use 40KB instead of 48KB
    static_assert(Config::NumQueriesPerCta * Config::TaskChunkSeqLen * sizeof(float) < 40 * 1024,
                  "Kernels relying on shared memory allocations over 48 KB per block are architecture-specific,"
                  "and must use dynamic shared memory rather than statically sized shared memory arrays.");
    __shared__ float smem_out[Config::NumQueriesPerCta][HeadSize];
    __shared__ float2 smem_max_sum[Config::NumQueriesPerCta];
    __shared__ float2 chunk_max_sum[Config::NumQueriesPerCta];
    __shared__ TSB sSB_buffer[NumWarps][PageSize * ScaleBiasNumChunks * 2];
    __shared__ float alibi_slopes_smem[Config::NumQueriesPerCta];

    // zero-ing out __shared__ buffers
    // attn_scores will be initialize by softmax_cta/softmax_warp
    // smem_out will not be loaded if sum in smem_max_sum is 0.0f
    if (threadIdx.x < Config::NumQueriesPerCta) {
      smem_max_sum[threadIdx.x] = float2{std::numeric_limits<float>::lowest(), 0.0f};
      alibi_slopes_smem[threadIdx.x] = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx + threadIdx.x];
    }
    // chunk_max_sum must be initialize per chunk-wise

    const int num_tokens = context_lens[seq_idx];
    const int64_t dummy_shape = 1073741824;  // 2^30, can be used for the slowest dim
    const auto num_pages = dummy_shape;

    auto gO = make_tensor(make_gmem_ptr(out), make_layout(make_shape(Int<HeadSize>{}, num_heads, num_seqs)))(_, _, seq_idx);                                                    // [num_seqs, num_heads, head_size]
    const auto gQ = make_tensor(make_gmem_ptr(q), make_layout(make_shape(Int<HeadSize>{}, num_heads, num_seqs), make_stride(_1{}, Int<HeadSize>{}, q_stride)))(_, _, seq_idx);  // [num_seqs, num_heads, head_size]
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

    // cooperative load q to sA (sQ)
    constexpr const auto SmemQLayout = make_layout(make_shape(Int<HeadSize>{}, Int<Config::NumQueriesPerCta>{}));
    __shared__ TQ sQ_buffer[cosize(SmemQLayout)];
    {
      auto sQ = make_tensor(make_smem_ptr(sQ_buffer), SmemQLayout);
      auto cQ = make_identity_tensor(shape(SmemQLayout));
      constexpr int QLoadVec = cute::min(next_power_of_two(ceil_div(size(SmemQLayout), NumThreads)), 8);  // load n elems per thread
      static_assert(HeadSize % QLoadVec == 0);
      constexpr int ThrD = HeadSize / QLoadVec;
      constexpr int ThrH = ceil_div(NumThreads, ThrD);
      constexpr int ValD = QLoadVec;
      constexpr int ValH = 1;
      const auto tiled_copy = make_tiled_copy(
          Copy_Atom<UniversalCopy<TQ>, TQ>{},
          make_layout(make_shape(Int<ThrD>{}, Int<ThrH>{})),
          make_layout(make_shape(Int<ValD>{}, Int<ValH>{}))
      );
      const auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
      const auto ctaQ = local_tile(gQ, SmemQLayout.shape(), make_coord(_0{}, head_idx / Config::NumQueriesPerCta));
      const auto thr_cQ = thr_copy.partition_S(cQ);
      const auto thr_ld = thr_copy.partition_S(ctaQ);
      auto thr_st = thr_copy.partition_D(sQ);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(thr_ld); i++) {
        CUTE_UNROLL
        for (int j = 0; j < size<2>(thr_ld); j++) {
          if (elem_less(thr_cQ(_0{}, i, j), shape(sQ))) {
            copy(thr_ld, thr_st);
          }
        }
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    }

    const auto gemv1_thr_copy = Gemv1TiledCopy{}.get_thread_slice(lane_id());
    const auto gemv2_thr_copy = Gemv2TiledCopy{}.get_thread_slice(lane_id());

    // gemv1 A
    const auto gemv1_sA = make_tensor(make_smem_ptr(sQ_buffer), Layout<Shape<_1, Int<HeadSize>, Int<Config::NumQueriesPerCta>>, Stride<_0, _1, Int<HeadSize>>>{});
    const auto gemv1_tA_view = gemv1_thr_copy.partition_S(gemv1_sA);
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
    TQ* attn_scores_ptr = reinterpret_cast<TQ*>(&logits_or_attn_scores_buffer[0]);
    auto attn_scores = make_tensor(make_smem_ptr(attn_scores_ptr), Layout<Shape<Int<Config::TaskChunkSeqLen>, Int<Config::NumQueriesPerCta>>>{});
    const auto gemv2_sA = make_tensor(make_smem_ptr(attn_scores_ptr), Layout<Shape<_1, Int<PageSize>, Int<TaskChunk::NumPages>, Int<Config::NumQueriesPerCta>>>{});
    const auto gemv2_tA_view = gemv2_thr_copy.partition_S(gemv2_sA);
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

      BaseTask::template load_sb_to_reg<0>(tSB_copy_src, tSB_should_copy, chunk_page_table[warp_id()], tSB_copy_staging);
      // output for first GEMV
      float* logits_ptr = &logits_or_attn_scores_buffer[0];
      auto logits = make_tensor(make_smem_ptr(logits_ptr), Layout<Shape<Int<Config::TaskChunkSeqLen>, Int<Config::NumQueriesPerCta>>>{});
#pragma unroll 1
      for (int lid_in_chunk = warp_id(); lid_in_chunk < TaskChunk::NumPages; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];
        const int64_t next_physical_page_id = lid_in_chunk + NumWarps < TaskChunk::NumPages ? chunk_page_table[lid_in_chunk + NumWarps] : -1;
        store_sb_to_smem(tSB_copy_staging, tSB_should_copy, physical_page_id, tSB_copy_dst);                           // if curr is valid, store
        BaseTask::template load_sb_to_reg<0>(tSB_copy_src, tSB_should_copy, next_physical_page_id, tSB_copy_staging);  // if next is valid, load
        if (physical_page_id == -1) {
          continue;
        }

        const auto tA_view = gemv1_tA_view;
        auto tA = make_tensor_like(gemv1_tA_view(_, _0{}, _0{}, _0{}));

        const auto tB_view = gemv1_tB_view(_, _, _, physical_page_id);
        const auto tSB_view = gemv1_tSB_view;
        static_assert(size<1>(gemv1_tB_view) == 1);  // IterN over PageSize can be omitted
        auto tB = make_fragment_like<TQ>(tB_view(_, _0{}, _0{}));

        auto qk = make_tensor<float>(Int<Config::NumQueriesPerCta>{});
        clear(qk);
        CUTE_UNROLL
        for (int p = 0; p < size<2>(tB_view); p++) {  // IterK over HeadSize
          if (elem_less(gemv1_tB_coord(_0{}, _0{}, p), shape(gemv1_page_coord))) {
            BaseTask::template load_tB(tB, tB_view(_, _0{}, p), tSB_view(_, _0{}, p, _));
            CUTE_UNROLL
            for (int i = 0; i < Config::NumQueriesPerCta; i++) {
              copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(TQ) * size(tA), 128)>{}, tA_view(_, _0{}, p, i), tA);
              qk(i) += inner_product<float>(tA, tB);
              schedule_barrier();
            }
          }
        }
        auto tid_in_group = get<1>(Gemv1ThrLayout{}.get_hier_coord(int(threadIdx.x)));
        auto tok_idx_in_page = get<0>(gemv1_tB_coord(_0{}, _0{}, _0{}));
        const int token_idx = (chunk.start_logical_page_id(0) + lid_in_chunk) * PageSize + tok_idx_in_page;
        CUTE_UNROLL
        for (int i = 0; i < Config::NumQueriesPerCta; i++) {
          qk(i) *= scale;
          // reduce in thread group to get the full qk
          qk(i) = warp::reduce<Gemv1ThrK, /*Strided=*/Gemv1TransThrLayout>(qk(i), [](float a, float b) { return a + b; });
          qk(i) += alibi_slopes ? alibi_slopes_smem[i] * (token_idx - num_tokens + 1) : 0;
          if (tid_in_group == 0) {
            logits(token_idx % Config::TaskChunkSeqLen, i) = token_idx < num_tokens ? qk(i) : 0.f;  // TODO: start boundary
          }
        }
      }

      if constexpr (!Config::SingleChunk) {
        chunk.preload_page_table_for_chunk<TaskChunk::NumPages>(seq_page_table, preload_pids, 0, num_tokens);
      }

      // reuse broadcast buffer for reduction
      static_assert(2 * NumWarps <= (BROADCAST0_BUFFER_SIZE_IN_BYTES / sizeof(float)));
      int prefix = chunk.start_tok_idx(0) - (chunk.start_tok_idx(0) % Config::TaskChunkSeqLen);
      __syncthreads();
      CUTE_UNROLL
      for (int i = warp_id(); i < Config::NumQueriesPerCta; i += NumWarps) {
        softmax_warp<Config::TaskChunkSeqLen>(
            &logits(_0{}, i),
            chunk.start_tok_idx(0) - prefix,
            chunk.end_tok_idx(num_tokens) - prefix,
            &chunk_max_sum[i]
        );
      }
      __syncthreads();
      inplace_convert_attn_scores(logits, attn_scores);

      TaskChunk next_chunk{-1, -1};
      if constexpr (!Config::SingleChunk) {
        worker->take_work(next_chunk.start, next_chunk.end);
      }


      BaseTask::template load_sb_to_reg<1>(tSB_copy_src, tSB_should_copy, chunk_page_table[warp_id()], tSB_copy_staging);
      auto acc = make_tensor<float>(make_shape(Int<size<1>(gemv2_tB_view)>{}, Int<Config::NumQueriesPerCta>{}));
      clear(acc);
      int num_pages_in_chunk = chunk.end_logical_page_id(num_tokens) - chunk.start_logical_page_id(0);
#pragma unroll 1
      for (int lid_in_chunk = warp_id(); lid_in_chunk < num_pages_in_chunk; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];
        const int64_t next_physical_page_id = lid_in_chunk + NumWarps < TaskChunk::NumPages ? chunk_page_table[lid_in_chunk + NumWarps] : -1;
        store_sb_to_smem(tSB_copy_staging, tSB_should_copy, physical_page_id, tSB_copy_dst);                           // if curr is valid, store
        BaseTask::template load_sb_to_reg<1>(tSB_copy_src, tSB_should_copy, next_physical_page_id, tSB_copy_staging);  // if next is valid, load

        const auto tA_view = filter_zeros(gemv2_tA_view(_, _, _, lid_in_chunk, _))(_, _1{}, _1{}, _);
        const auto tB_view = gemv2_tB_view(_, _, _0{}, physical_page_id);
        const auto tB_coord = gemv2_tB_coord(_, _, _0{});
        const auto tSB_view = [&]() {
          if constexpr (HasSB) {
            return gemv2_tSB_view(_, _, _0{}, _);
          } else {
            return Unused();
          }
        }();

        static_assert(rank(tA_view) == 2);
        static_assert(size<1>(gemv2_tB_view) == size<0>(acc));
        static_assert(size<2>(gemv2_tB_view) == 1);  // IterK over PageSize can be omitted

        auto tA = make_fragment_like(tA_view(_, _0{}));
        auto tB = make_fragment_like<TQ>(tB_view(_, _0{}));

        bool is_full_chunk = chunk.end_tok_idx(num_tokens) - chunk.start_tok_idx(0) == Config::TaskChunkSeqLen;
        bool is_full_page = lid_in_chunk != 0 && lid_in_chunk != num_pages_in_chunk - 1;
        if (is_full_chunk || is_full_page) {
          CUTE_UNROLL
          for (int j = 0; j < size<1>(tB_view); j++) {
            if (elem_less(tB_coord(_0{}, j), shape(gemv2_page_coord))) {
              BaseTask::template load_tB(tB, tB_view(_, j), tSB_view(_, j, _));
              CUTE_UNROLL
              for (int i = 0; i < Config::NumQueriesPerCta; i++) {
                copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(float) * Gemv2ValK, 128)>{}, tA_view(_, i), tA);
                acc(j, i) += inner_product<float>(tA, tB);
                schedule_barrier();
              }
            }
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

          CUTE_UNROLL
          for (int j = 0; j < size<1>(tB_view); j++) {
            if (elem_less(tB_coord(_0{}, j), shape(gemv2_page_coord))) {
              BaseTask::template load_tB(tB, tB_view(_, j), tSB_view(_, j, _));
              copy_if(pred, tB, masked_tB);
              CUTE_UNROLL
              for (int i = 0; i < Config::NumQueriesPerCta; i++) {
                copy(AutoVectorizingCopyWithAssumedAlignment<cute::min(8 * sizeof(float) * Gemv2ValK, 128)>{}, tA_view(_, i), tA);
                acc(j, i) += inner_product<float>(tA, masked_tB);
              }
            }
          }
        }
      }

      CUTE_UNROLL
      for (int i = 0; i < Config::NumQueriesPerCta; i++) {
        CUTE_UNROLL
        for (int v = 0; v < size<0>(acc); v++) {
          acc(v, i) = warp::reduce<Gemv2ThrK>(acc(v, i), [](float a, float b) { return a + b; });
        }
      }

      __syncthreads();  // sync to reuse logits
      auto& chunk_out = logits;
      static_assert(HeadSize <= Config::TaskChunkSeqLen);

      constexpr auto OutThrCoord = make_identity_tensor(make_shape(Int<Gemv2ThrK>{}, Int<Gemv2ThrN>{}));
      const auto thr_coord = OutThrCoord(lane_id());
      const auto thr_k = get<0>(thr_coord);  // thr_k == 0 is the leading threads in previous warp reduction
      const auto thr_n = get<1>(thr_coord);

      if (warp_id() == 0 && thr_k == 0) {
        CUTE_UNROLL
        for (int v = 0; v < size<0>(acc); v++) {
          int dim_idx = thr_n + v * Gemv2ThrN;  // strided
          if (dim_idx < HeadSize) {
            CUTE_UNROLL
            for (int i = 0; i < Config::NumQueriesPerCta; i++) {
              chunk_out(dim_idx, i) = acc(v, i);
            }
          }
        }
      }
      __syncthreads();
      CUTE_UNROLL
      for (int warp = 1; warp < NumWarps; warp++) {
        if (warp_id() == warp && thr_k == 0) {
          CUTE_UNROLL
          for (int v = 0; v < size<0>(acc); v++) {
            int dim_idx = thr_n + v * Gemv2ThrN;  // strided
            if (dim_idx < HeadSize) {
              CUTE_UNROLL
              for (int i = 0; i < Config::NumQueriesPerCta; i++) {
                chunk_out(dim_idx, i) += acc(v, i);
              }
            }
          }
        }
        __syncthreads();
      }

      for (int i = warp_id(); i < Config::NumQueriesPerCta; i += NumWarps) {
        FlashAccWarp::acc(smem_out[i], &smem_max_sum[i], &chunk_out(_0{}, i), &chunk_max_sum[i]);
      }
      if constexpr (Config::SingleChunk) {
        break;
      } else {
        worker->broadcast_work(broadcast_buffer, next_chunk.start, next_chunk.end);
        chunk = next_chunk;
      }
    }

    __syncthreads();
    auto get_max_sum_idx = [=](int i) {  // i in range [0,NumQueriesPerCta)
      if constexpr (Config::UseHeadSeq) {
        return (head_idx + i) * Config::MaxNumSeqs + seq_idx;
      } else {
        return seq_idx * num_heads + (head_idx + i);
      }
    };
    if constexpr (Config::InplaceFlashAcc) {
      for (int i = 0; i < Config::NumQueriesPerCta; ++i) {
        TO* gmem_out = &gO(_0{}, head_idx + i);
        FlashAccCta::atomic_acc(broadcast_buffer, gmem_out, &out_max_sum[get_max_sum_idx(i)], smem_out[i], &smem_max_sum[i]);
      }
    } else {
      for (int i = warp_id(); i < Config::NumQueriesPerCta; i += NumWarps) {
        TO* gmem_out = &gO(_0{}, head_idx + i);
        FlashAccWarp::write_inc(gmem_out, out_max_sum ? &out_max_sum[get_max_sum_idx(i)] : nullptr, smem_out[i], &smem_max_sum[i]);
      }
    }
  }

protected:
  using BaseTask::NumWarps;
  using BaseTask::x;

  using BaseTask::Gemv1ThrK;
  using BaseTask::Gemv1ThrN;
  using BaseTask::Gemv1TransThrLayout;
  using BaseTask::Gemv1ValK;
  using BaseTask::Gemv1ValN;
  using Gemv1ThrLayout = typename BaseTask::Gemv1ThrLayout;
  using Gemv1TiledCopy = typename BaseTask::Gemv1TiledCopy;

  using BaseTask::Gemv2ThrK;
  using BaseTask::Gemv2ThrN;
  using BaseTask::Gemv2ValK;
  using BaseTask::Gemv2ValN;
  using Gemv2TiledCopy = typename BaseTask::Gemv2TiledCopy;

  using BaseTask::ScaleBiasCopyValPerThread;
  using BaseTask::ScaleBiasNumChunks;
  using ScaleBiasWarpCopy = typename BaseTask::ScaleBiasWarpCopy;

  using BaseTask::store_sb_to_smem;
};

}  // namespace onnxruntime::contrib::paged

#ifdef DUMMY_TASK_DPRINTF1
#undef DUMMY_TASK_DPRINTF1
#undef TASK_DPRINTF1
#endif
