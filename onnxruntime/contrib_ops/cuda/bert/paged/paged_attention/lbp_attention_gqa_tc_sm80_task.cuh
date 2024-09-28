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

#ifndef TASK_DPRINTF1
#define DUMMY_TASK_DPRINTF1(...)
#define TASK_DPRINTF1 DUMMY_TASK_DPRINTF1
#endif

namespace onnxruntime::contrib::paged {

template <int KSize, int NumElemsPerLoad, typename MMA_Atom>
struct KVectorizeTiler {};

template <int KSize, int NumElemsPerLoad>
struct KVectorizeTiler<KSize, NumElemsPerLoad, SM80_16x8x8_F32F16F16F32_TN> {
  // These are MMA_Atom dependent
  static constexpr int MmaThrK = 4;  // 4 thr consecutive
  static constexpr int MmaValK = 2;  // 2 val consecutive
  static_assert(KSize % (MmaThrK * NumElemsPerLoad) == 0);

  // shuffle and packed
  static constexpr int NumMmaTilePacked = NumElemsPerLoad / MmaValK;
  using Tiler = decltype(select<0, 2, 1, 3>(make_layout(Shape<_2, Int<NumMmaTilePacked>, _4, Int<KSize / (NumMmaTilePacked * 8)>>{})));
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
struct PagedGroupQueryAttentionTcSm80Task : public PagedAttentionTask<NumThreads, HeadSize, PageSize, TI, TO_, TKV, Worker, Config, KVConfig> {
  static_assert(PageSize == 8 || PageSize == 16 || PageSize == 32);
  constexpr static int NumQueriesPerCtaMma = 8;                    // physical limit, control smem and mma tile
  static_assert(Config::NumQueriesPerCta <= NumQueriesPerCtaMma);  // number of queries this task actually process

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

  static_assert(sizeof(TKV) <= 2, "not implemented for float kv");

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
    __align__(128) __shared__ float smem_out[NumQueriesPerCtaMma][HeadSize];
    __shared__ float2 smem_max_sum[NumQueriesPerCtaMma];
    __shared__ float2 chunk_max_sum[NumQueriesPerCtaMma];
    __align__(128) __shared__ TSB sSB_buffer[NumWarps][PageSize * ScaleBiasNumChunks * 2];
    __shared__ float alibi_slopes_smem[NumQueriesPerCtaMma];
    __align__(128) __shared__ float logits_or_attn_scores_buffer[NumQueriesPerCtaMma * (Config::TaskChunkSeqLen + 4)];
    static_assert(NumQueriesPerCtaMma * Config::TaskChunkSeqLen * sizeof(float) < 40 * 1024,
                  "Kernels relying on shared memory allocations over 48 KB per block are architecture-specific,"
                  "and must use dynamic shared memory rather than statically sized shared memory arrays.");

    // zero-ing out __shared__ buffers
    // attn_scores will be initialize by softmax_cta
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
      auto l = group<1, 3>(select<1, 0, 2, 3, 4>(l_raw));  // [page_size, (x, head_size/x), num_kv_heads, num_pages]
      return make_tensor(make_gmem_ptr(k_cache), l)(/*tok_idx_in_page*/ _, /*dim_idx_in_head*/ _, kv_head_idx, /*physical_page_id*/ _);
    }();
    const auto gV = [&]() {
      auto l = select<1, 0, 2, 3>(make_layout(make_shape(Int<PageSize>{}, Int<HeadSize>{}, num_kv_heads, num_pages)));
      return make_tensor(make_gmem_ptr(v_cache), l)(/*dim_idx_in_head*/ _, /*tok_idx_in_page*/ _, kv_head_idx, /*physical_page_id*/ _);
    }();
    const auto gSB = [&]() {
      if constexpr (HasSB) {
        auto l = make_layout(make_shape(Int<PageSize>{}, Int<ScaleBiasNumChunks>{}, _2{}, num_kv_heads, _2{}, num_pages));
        auto t = make_tensor(make_gmem_ptr(kv_scalebias), l)(/*tok_idx_in_page*/ _, /*chunk_idx*/ _, /*s_or_b*/ _, kv_head_idx, /*k_or_v*/ _, /*physical_page_id*/ _);
        return t.compose(_, Layout<Shape<Int<KVConfig::ChunkSize>, Int<ScaleBiasNumChunks>>, Stride<_0, _1>>{}, _, _, _);
      } else {
        return Unused();
      }
    }();
    auto gKSB = gSB(_, _, _, _0{}, _);
    auto gVSB = [&]() {
      if constexpr (HasSB) {
        return select_tensor<1, 0, 2, 3>(gSB(_, _, _, _1{}, _));
      } else {
        return Unused();
      }
    }();

    const auto seq_page_table = &(make_tensor(make_gmem_ptr(page_table), make_layout(make_shape(num_seqs, max_num_pages_per_seq), LayoutRight{}))(seq_idx, _0{}));

    // cooperative load q to sA (sQ)
    constexpr const auto SmemQLayout = make_layout(make_shape(Int<NumQueriesPerCtaMma>{}, Int<HeadSize>{}), make_stride(Int<HeadSize + 8>{}, _1{}));
    __align__(128) __shared__ TQ sQ_buffer[cosize(SmemQLayout)];
    auto sQ = make_tensor(make_smem_ptr(sQ_buffer), SmemQLayout);
    {
      constexpr auto valid_sQ_shape = make_shape(Int<HeadSize>{}, Int<Config::NumQueriesPerCta>{});      // transposed view for copy of valid q tile
      auto sQ = make_tensor(make_smem_ptr(sQ_buffer), select<1, 0>(SmemQLayout));                        // transposed view for copy
      constexpr int QLoadVec = cute::min(next_power_of_two(ceil_div(size(sQ.shape()), NumThreads)), 8);  // load n elems per thread
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
      const auto ctaQ = local_tile(gQ, valid_sQ_shape, make_coord(_0{}, head_idx / Config::NumQueriesPerCta));
      const auto cQ = make_identity_tensor(valid_sQ_shape);
      const auto thr_cQ = thr_copy.partition_S(cQ);
      const auto thr_ld = thr_copy.partition_S(ctaQ);
      auto thr_st = thr_copy.partition_D(sQ);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(thr_ld); i++) {
        CUTE_UNROLL
        for (int j = 0; j < size<2>(thr_ld); j++) {
          if (elem_less(thr_cQ(_0{}, i, j), valid_sQ_shape)) {
            copy(thr_ld(_, i, j), thr_st(_, i, j));
          }
        }
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    }

    float* logits_ptr = &logits_or_attn_scores_buffer[0];
    using LogitsLayout = Layout<Shape<Int<Config::TaskChunkSeqLen>, Int<NumQueriesPerCtaMma>>, Stride<_1, Int<Config::TaskChunkSeqLen + 4>>>;
    auto logits = make_tensor(make_smem_ptr(logits_ptr), LogitsLayout{});
    auto cL = make_identity_tensor(shape(logits));
    auto logits_paged = coalesce_tensor(flat_divide(logits, make_tile(Int<PageSize>{}, Int<NumQueriesPerCtaMma>{})));  // (tid_in_page,hidx_in_group,lpid)
    auto cL_paged = coalesce_tensor(flat_divide(cL, make_tile(Int<PageSize>{}, Int<NumQueriesPerCtaMma>{})));          // (tid_in_page,hidx_in_group,lpid)

    auto tiled_mma1 = make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{});
    auto thr_mma1 = tiled_mma1.get_thread_slice(lane_id());
    using KVectorizeTiler1 = KVectorizeTiler<HeadSize, 4, SM80_16x8x8_F32F16F16F32_TN>;
    constexpr auto KTiler1 = typename KVectorizeTiler1::Tiler{};
    auto tCgK = thr_mma1.partition_A(gK.compose(_, KTiler1, _));  // GEMM1 A
    auto tCsQ = thr_mma1.partition_B(sQ.compose(_, KTiler1));     // GEMM1 B
    auto tCsL = thr_mma1.partition_C(logits_paged);               // GEMM1 C
    auto tCcL = thr_mma1.partition_C(cL_paged);                   // GEMM1 C coord
    auto tCgKSB = [&]() {
      if constexpr (HasSB) {
        constexpr int HeadSizePadded = ceil_div(HeadSize, KVConfig::ChunkSize) * KVConfig::ChunkSize;
        using KVectorizeTiler1Padded = KVectorizeTiler<HeadSizePadded, 4, SM80_16x8x8_F32F16F16F32_TN>;
        constexpr auto KTiler1 = typename KVectorizeTiler1Padded::Tiler{};
        return thr_mma1.partition_A(gKSB.compose(_, KTiler1, _, _));
      } else {
        return Unused();
      }
    }();

    float* attn_scores_ptr = &logits_or_attn_scores_buffer[0];
    using AttnScoresLayout = Layout<Shape<Int<PageSize>, Int<Config::TaskChunkSeqLen / PageSize>, Int<NumQueriesPerCtaMma>>, Stride<_1, Int<PageSize>, Int<Config::TaskChunkSeqLen + 4>>>;
    auto attn_scores = make_tensor(make_smem_ptr(attn_scores_ptr), AttnScoresLayout{});

    auto tiled_mma2 = make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{});
    auto thr_mma2 = tiled_mma2.get_thread_slice(lane_id());
    constexpr bool Gemm2IsFullMma = PageSize % 16 == 0;  // otherwise, half of mma input and out is valid
    using KVectorizeTiler2 = KVectorizeTiler<PageSize, Gemm2IsFullMma ? 4 : 2, SM80_16x8x8_F32F16F16F32_TN>;
    constexpr auto KTiler2 = typename KVectorizeTiler2::Tiler{};
    auto tCgV = thr_mma2.partition_A(gV.compose(_, KTiler2, _));
    auto tCsS = thr_mma2.partition_B(select_tensor<2, 0, 1>(attn_scores).compose(_, KTiler2, _));
    auto tCgVSB = [&]() {
      if constexpr (HasSB) {
        return thr_mma2.partition_A(gVSB.compose(_, KTiler2, _, _));
      } else {
        return Unused();
      }
    }();

    static_assert(HeadSize <= Config::TaskChunkSeqLen);
    auto chunk_out = logits.compose(make_tile(Int<HeadSize>{}, _));  // reuse logits as chunk_out
    auto tCsO = thr_mma2.partition_C(chunk_out);

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

      // GEMM1 ---------------------------------------------------------------------------------------------------------
      for (int lid_in_chunk = warp_id(); lid_in_chunk < TaskChunk::NumPages; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];
        if (physical_page_id == -1) {
          continue;
        }

        auto tCgK_page = tCgK(_, _, _, physical_page_id);
        auto tCgK_vec_view = tiled_divide(tCgK_page, make_tile(make_tile(_2{}, _1{}), _1{}, Int<KVectorizeTiler1::NumMmaTilePacked>{}));  // ((Frag),(FragIters...), IterM, IterK)
        auto tCgK_vec0 = make_tensor_like<half>(tCgK_vec_view(_, make_coord(_0{}, _0{}), _, _0{}));                                       // ((HalfFrag),IterM)
        auto tCgK_vec1 = make_tensor_like<half>(tCgK_vec_view(_, make_coord(_0{}, _0{}), _, _0{}));                                       // ((HalfFrag),IterM)
        auto tCgKSB_vec_view = [&]() {                                                                                                    // ((Frag),(FragIters...), IterM, IterK,SB)
          if constexpr (HasSB) {
            auto tCgKSB_page = tCgKSB(_, _, _, /*s_or_b*/ _, physical_page_id);
            return tiled_divide(tCgKSB_page, make_tile(make_tile(_2{}, _1{}), _1{}, Int<KVectorizeTiler1::NumMmaTilePacked>{}));
          } else {
            return Unused();
          }
        }();

        auto tCrA = thr_mma1.make_fragment_A(append_tensor<3>(tCgK(/*Frag*/ _, /*IterM*/ _, /*IterK*/ _0{}, /*lpid*/ _0{})));
        auto tCrB = thr_mma1.make_fragment_B(append_tensor<3>(tCsQ(/*Frag*/ _, /*IterN*/ _, /*IterK*/ _0{})));
        auto tCrC = thr_mma1.make_fragment_C(append_tensor<3>(tCsL(/*Frag*/ _, /*IterM*/ _, /*IterN*/ _, /*lpid*/ _0{})));

        clear(tCrC);
        if constexpr (!Gemm2IsFullMma) {
          clear(tCgK_vec1);
        }
        for_each(make_int_sequence<size<3>(tCgK_vec_view)>{}, [&](auto p) {
          if constexpr (!HasSB) {  // direct copy
            copy(tCgK_vec_view(_, make_coord(_0{}, _0{}), _, p), tCgK_vec0);
            if constexpr (Gemm2IsFullMma) {
              copy(tCgK_vec_view(_, make_coord(_0{}, _1{}), _, p), tCgK_vec1);
            }
          } else {  // copy then dequantize
            auto copy_then_dequantize = [&](auto&& half_frag_idx, auto& out) {
              auto tCgK_copy = make_tensor_like<TK>(tCgK_vec0);
              auto tCgKSB_vec = make_tensor_like(tCgKSB_vec_view(_, make_coord(_0{}, _0{}), _, _0{}, /*s_or_b*/ _));
              copy(tCgK_vec_view(_, make_coord(_0{}, half_frag_idx), _, p), tCgK_copy);
              copy(tCgKSB_vec_view(_, make_coord(_0{}, half_frag_idx), _, p, _), tCgKSB_vec);
#if 0
              // FIXME: slow and incorrect! incorrectness is cause by recast in tensor_convert
              tensor_convert(tCgK_copy, tCgKSB_vec(_, _, _0{}), tCgKSB_vec(_, _, _1{}), out);
#else
              // scalebias must be broadcasting
              static_assert(layout(tCgKSB_vec)(_0{}) == layout(tCgKSB_vec)(_1{}));
              static_assert(layout(tCgKSB_vec)(_0{}) == layout(tCgKSB_vec)(_2{}));
              static_assert(layout(tCgKSB_vec)(_0{}) == layout(tCgKSB_vec)(_3{}));

              const auto tCgK_copy4 = recast<array<float_e4m3_t, 4>>(tCgK_copy);
              const auto out2x2 = recast<half2x2>(out);
              CUTE_UNROLL
              for (int v_ = 0; v_ < size(tCgK_copy4); v_++) {
                int v = layout(tCgK_copy4)(v_);
                auto b = tCgKSB_vec(_, _, _1{})(4 * v);
                auto s = tCgKSB_vec(_, _, _0{})(4 * v);
                half2 scale2{s, s};
                half2 bias2{b, b};
                auto in2x2 = fast_type_convert<half2x2>(tCgK_copy4(v));
                out2x2(v)[0] = __hfma2(scale2, in2x2[0], bias2);
                out2x2(v)[1] = __hfma2(scale2, in2x2[1], bias2);
              }
#endif
            };

            copy_then_dequantize(_0{}, tCgK_vec0);
            if constexpr (Gemm2IsFullMma) {
              copy_then_dequantize(_1{}, tCgK_vec1);
            }
          }
          CUTE_UNROLL
          for (int pp = 0; pp < KVectorizeTiler1::NumMmaTilePacked; pp++) {
            copy(tCgK_vec0(make_coord(_, _, pp), _), tCrA(make_coord(_, _0{}), _, _));
            copy(tCgK_vec1(make_coord(_, _, pp), _), tCrA(make_coord(_, _1{}), _, _));
            copy(tCsQ(_, _, make_coord(pp, p)), tCrB);
            gemm(tiled_mma1, tCrC, tCrA, tCrB, tCrC);
          }
        });
        transform(tCrC, [&](auto v) { return v * scale; });
        if (alibi_slopes) {
          for_each(make_int_sequence<size<0>(tCrC)>{}, [&](auto&& v) {  // IterFrag
            const auto frag_coord = layout<0>(tCrC).get_hier_coord(v);
            for_each(make_int_sequence<size<1>(tCrC)>{}, [&](auto&& i) {  // IterM
              const auto [tok_id_in_chunk, hidx_in_group] = tCcL(frag_coord, i, _0{}, lid_in_chunk);
              const int token_idx = chunk.start_logical_page_id(0) * PageSize + tok_id_in_chunk;
              tCrC(frag_coord, i, _0{}) += alibi_slopes_smem[hidx_in_group] * (token_idx - num_tokens + 1);
            });
          });
          // NOTE: we don't need to handle oob here, softmax_warp automatically ignore them and zero them out
        }
        if constexpr (Gemm2IsFullMma) {
          copy(tCrC, tCsL(_, _, _, lid_in_chunk));
        } else {
          // only store the first half of the output when PageSize == 8
          copy(tCrC(make_coord(_, _0{}), _, _), tCsL(make_coord(_, _0{}), _, _, lid_in_chunk));
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

      TaskChunk next_chunk{-1, -1};
      if constexpr (!Config::SingleChunk) {
        worker->take_work(next_chunk.start, next_chunk.end);
      }

      // GEMM2 ---------------------------------------------------------------------------------------------------------
      auto tCrC = thr_mma2.make_fragment_C(tCsO);
      clear(tCrC);
      int num_pages_in_chunk = chunk.end_logical_page_id(num_tokens) - chunk.start_logical_page_id(0);
#pragma unroll(HeadSize < 128 ? 2 : 1)
      for (int lid_in_chunk = warp_id(); lid_in_chunk < num_pages_in_chunk; lid_in_chunk += NumWarps) {
        const int64_t physical_page_id = chunk_page_table[lid_in_chunk];

        auto tCrA = thr_mma2.make_fragment_A(append_tensor<3>(tCgV(_, _, _, _0{})));
        auto tCrB = thr_mma2.make_fragment_B(append_tensor<3>(tCsS(_, _, _, _0{})));

        auto tCgV_page = tCgV(_, _, _, physical_page_id);
        auto tCgVSB_page = tCgVSB(_, _, _, /*s_or_b*/ _, physical_page_id);
        for_each(make_int_sequence<size<1>(tCrC)>{}, [&](auto i) {
          auto tCgV_vec = make_tensor_like<half>(tCgV_page(_, _0{}, _));

          if constexpr (!HasSB) {
            copy(tCgV_page(_, i, _), tCgV_vec);
          } else {  // copy then dequantize
            auto tCgV_copy = make_tensor_like<TV>(tCgV_vec);
            auto tCgVSB_vec = make_tensor_like<TSB>(tCgVSB_page(_, _0{}, _, _));
            copy(tCgV_page(_, i, _), tCgV_copy);
            copy(tCgVSB_page(_, i, _, _), tCgVSB_vec);
#if 0
            // FIXME: slow and incorrect! incorrectness is cause by recast in tensor_convert
            // tensor_convert(tCgV_copy, tCgVSB_vec(_, _, _0{}), tCgVSB_vec(_, _, _1{}), tCgV_vec);
#else
            const auto tCgV_copy2 = recast<array<float_e4m3_t, 2>>(tCgV_copy);
            const auto scale2 = recast<half2>(tCgVSB_vec(_, _, _0{}));
            const auto bias2 = recast<half2>(tCgVSB_vec(_, _, _1{}));
            const auto tCgV_vec2 = recast<half2>(tCgV_vec);
            CUTE_UNROLL
            for (int v = 0; v < size(tCgV_copy2); v++) {
              tCgV_vec2(v) = __hfma2(scale2(v), fast_type_convert<half2>(tCgV_copy2(v)), bias2(v));
            }
#endif
          }
          copy(tCgV_vec, tCrA(_, i, _));
        });

        auto logical_page_id = chunk.start_logical_page_id(0) + lid_in_chunk;
        bool is_full_page = (logical_page_id >= ceil_div(chunk.start_tok_idx(0), PageSize)) &&
                            (logical_page_id < chunk.end_tok_idx(num_tokens) / PageSize);
        if (!is_full_page) {
          enforce_uniform();
          filter_inf_nan(tCrA);
        }

        auto tCsS_page = tCsS(_, _, _, lid_in_chunk);
        CUTE_UNROLL
        for (int j = 0; j < size<2>(tCrC); j++) {
          auto tCsS_vec = make_tensor_like<float>(tCsS_page(_, _, _));
          copy(tCsS_page(_, j, _), tCsS_vec);
          transform(tCsS_vec, tCrB(_, j, _), [](auto v) { return type_convert<half>(v); });
        }

        gemm(tiled_mma2, tCrC, tCrA, tCrB, tCrC);
      }

      __syncthreads();  // sync to reuse logits as chunk_out

      if (warp_id() == 0) {
        copy(tCrC, tCsO);
      }
      __syncthreads();
      CUTE_UNROLL
      for (int warp = 1; warp < NumWarps; warp++) {
        if (warp_id() == warp) {
          transform(tCsO, tCrC, tCsO, [](auto a, auto b) { return a + b; });
        }
        __syncthreads();
      }

      CUTE_UNROLL
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

  using BaseTask::ScaleBiasNumChunks;
};

}  // namespace onnxruntime::contrib::paged

#ifdef DUMMY_TASK_DPRINTF1
#undef DUMMY_TASK_DPRINTF1
#undef TASK_DPRINTF1
#endif
