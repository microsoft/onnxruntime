/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

#include <cmath>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "contrib_ops/cuda/bert/flash_attention/block_info.h"
#include "contrib_ops/cuda/bert/flash_attention/kernel_traits.h"
#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include "contrib_ops/cuda/bert/flash_attention/softmax.h"

namespace onnxruntime {
namespace flash {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE auto
make_tiled_copy_A_warpcontiguousM(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA const& tiled_mma) {
  using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
  using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
  constexpr int AtomShape_M = decltype(cute::size<0>(AtomShape_MNK{}))::value;
  constexpr int kNWarps = decltype(cute::size<0>(TileShape_MNK{}))::value / AtomShape_M;
  constexpr int MMAStride_M = MMA_M * AtomShape_M;
  auto t = make_tile(cute::Layout<cute::Shape<cute::Int<AtomShape_M>, cute::Int<kNWarps>>,
                                  cute::Stride<_1, cute::Int<MMAStride_M>>>{},
                     make_layout(cute::size<2>(TileShape_MNK{})));

  return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutA_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE auto
make_tiled_copy_C_warpcontiguousM(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA const& tiled_mma) {
  using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
  using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
  constexpr int AtomShape_M = decltype(cute::size<0>(AtomShape_MNK{}))::value;
  constexpr int kNWarps = decltype(cute::size<0>(TileShape_MNK{}))::value / AtomShape_M;
  constexpr int MMAStride_M = MMA_M * AtomShape_M;
  auto t = make_tile(cute::Layout<cute::Shape<cute::Int<AtomShape_M>, cute::Int<kNWarps>>,
                                  cute::Stride<_1, cute::Int<MMAStride_M>>>{},
                     // TODO: Shouldn't this be size<1>?
                     make_layout(cute::size<2>(TileShape_MNK{})));
  // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousM "); print(t); printf("\n");  }
  return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_first, bool Check_inf = false, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0& scores, Tensor1& scores_max, Tensor1& scores_sum,
                                         Tensor2& acc_o, float softmax_scale_log2) {
  if (Is_first) {
    flash::template reduce_max</*zero_init=*/true>(scores, scores_max);
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    flash::reduce_sum(scores, scores_sum);
  } else {
    cute::Tensor scores_max_prev = make_fragment_like(scores_max);
    cute::copy(scores_max, scores_max_prev);
    flash::template reduce_max</*zero_init=*/false>(scores, scores_max);
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    cute::Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < cute::size(scores_max); ++mi) {
      float scores_max_cur = !Check_inf
                                 ? scores_max(mi)
                                 : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
      float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
      scores_sum(mi) *= scores_scale;
#pragma unroll
      for (int ni = 0; ni < cute::size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    cute::Tensor scores_sum_cur = make_fragment_like(scores_sum);
    flash::reduce_sum(scores, scores_sum_cur);
#pragma unroll
    for (int mi = 0; mi < cute::size(scores_sum); ++mi) {
      scores_sum(mi) += scores_sum_cur(mi);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename TiledCopy>
inline __device__ void write_softmax_to_gmem(
    cute::Tensor<Engine0, Layout0> const& tOrP, cute::Tensor<Engine1, Layout1>& tPgP, TiledCopy gmem_tiled_copy_P) {
  // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
  cute::Layout l = tOrP.layout();
  cute::Tensor tPrP = make_tensor(tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
  CUTE_STATIC_ASSERT_V(cute::size<2>(tPgP) == _1{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(tPrP) == cute::size<1>(tPgP));
#pragma unroll
  for (int mi = 0; mi < cute::size<1>(tPrP); ++mi) {
    cute::copy(gmem_tiled_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params& params, const int bidb, const int bidh, const int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int MMA_M = kBlockM / decltype(cute::size<0>(typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal) {
    n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    // We exit early and write 0 to gO and gLSE.
    // Otherwise we might read OOB elements from gK and gV.
    if (n_block_max <= 0) {
      const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
      const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
      Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_stride(params.o_row_stride, _1{}));
      Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

      typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
      auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
      Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
      Tensor tOrO = make_tensor<Element>(shape(tOgO));
      clear(tOrO);
      // Construct identity layout for sO
      Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
      // Repeat the partitioning with identity layouts
      Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
      Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
      if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
          tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
      }
      // Clear_OOB_K must be false since we don't want to write zeros to gmem
      flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
      for (int m = 0; m < size<1>(tOgO); ++m) {
        const int row = get<0>(tOcO(0, m, 0));
        if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
          gLSE(row) = INFINITY;
        }
      }
      return;
    }
  }

  // We iterate over the blocks in reverse order. This is because the last block is the only one
  // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
  // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

  const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
  const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

  cute::Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                                cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
  cute::Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
  cute::Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
                                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{},
                                make_stride(params.v_row_stride, _1{}));
  cute::Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.p_ptr) + row_offset_p),
                                cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>{},
                                make_stride(params.seqlen_k_rounded, _1{}));

  cute::Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                                typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  cute::Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : cute::size(sQ)),
                                typename Kernel_traits::SmemLayoutKV{});
  cute::Tensor sV = make_tensor(sK.data() + cute::size(sK), typename Kernel_traits::SmemLayoutKV{});
  cute::Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  cute::Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
  auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

  cute::Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  cute::Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  cute::Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
  cute::Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  cute::Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
  cute::Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  cute::Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  cute::Tensor tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
  cute::Tensor tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
  cute::Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  cute::Tensor acc_o = partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  cute::Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  cute::Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  cute::Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // TODO: this might need to change if we change the mma instruction in SM70
  cute::Tensor scores_max = make_tensor<ElementAccum>(cute::Shape<cute::Int<2 * cute::size<1>(acc_o)>>{});
  cute::Tensor scores_sum = make_fragment_like(scores_max);

  //
  // PREDICATES
  //

  // Construct identity layout for sQ and sK
  cute::Tensor cQ = make_identity_tensor(make_shape(cute::size<0>(sQ), cute::size<1>(sQ)));   // (BLK_M,BLK_K) -> (blk_m,blk_k)
  cute::Tensor cKV = make_identity_tensor(make_shape(cute::size<0>(sK), cute::size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

  // Repeat the partitioning with identity layouts
  cute::Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);     // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  cute::Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  cute::Tensor tQpQ = make_tensor<bool>(make_shape(cute::size<2>(tQsQ)));
  cute::Tensor tKVpKV = make_tensor<bool>(make_shape(cute::size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < cute::size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < cute::size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  // Prologue

  cute::Tensor tQrQ = make_fragment_like(tQgQ);
  // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
  if (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  if (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    cute::Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(cute::size<1>(tSsQ) == cute::size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  int n_block = n_block_max - 1;
  // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                     binfo.actual_seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();

  if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    cute::Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(cute::size<1>(tSsQ) == cute::size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  clear(acc_o);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
  // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
  // We will have at least 1 "masking" iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
  // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps = !Is_causal
                                      ? 1
                                      : (Is_even_MN ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    cute::Tensor acc_s = partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    } else {
      // Clear the smem tiles to account for predicated off loads
      flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    cute::Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    // We don't put the masking before the matmul S = Q K^T because we don't clear sK
    // for rows outside actual_seqlen_k. So those rows could have Inf / NaN, and the matmul
    // can produce Inf / NaN.
    if (!Is_causal) {
      if (!Is_even_MN) {
        flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
      }
    } else {
      // I can't get the stride from idx_row
      flash::apply_mask_causal(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                               // m_block * kBlockM + get<0>(idx_row(0)),
                               m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                               binfo.actual_seqlen_q,
                               kNWarps * 16);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the synchronization
      // isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    masking_step == 0
        ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2)
        : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

    // Convert scores from fp32 to fp16/bf16
    cute::Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    cute::Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    // if (Return_softmax) {
    //   cute::Tensor tOrP_copy = make_fragment_like(tOrP);
    //   copy(tOrP, tOrP_copy);
    //   flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_thr_copy_P);
    //   tPgP.data() = tPgP.data() + (-kBlockN);
    // }

    flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    // This check is at the end of the loop since we always have at least 1 iteration
    if (n_masking_steps > 1 && n_block <= 0) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= 0; --n_block) {
    cute::Tensor acc_s = partition_fragment_C(tiled_mma, cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();
    // Advance gV
    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the synchronization
      // isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    cute::Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

    cute::Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    cute::Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    // if (Return_softmax) {
    //   cute::Tensor tOrP_copy = make_fragment_like(tOrP);
    //   copy(tOrP, tOrP_copy);
    //   flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_thr_copy_P);
    //   tPgP.data() = tPgP.data() + (-kBlockN);
    // }

    flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // Epilogue

  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
  cute::Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
  cute::Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < cute::size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = (sum == 0.f || sum != sum) ? INFINITY : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = inv_sum;
#pragma unroll
    for (int ni = 0; ni < cute::size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // Convert acc_o from fp32 to fp16/bf16
  cute::Tensor rO = flash::convert_type<Element>(acc_o);
  cute::Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);  // auto smem_thr_copy_O = make_tiled_copy_C_warpcontiguousM<MMA_M>(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma).get_thread_slice(tidx);
  cute::Tensor taccOrO = smem_thr_copy_O.retile_S(rO);              // ((Atom,AtomNum), MMA_M, MMA_N)
  cute::Tensor taccOsO = smem_thr_copy_O.partition_D(sO);           // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  cute::Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                                cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{},
                                make_stride(params.o_row_stride, _1{}));
  cute::Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse),
                                  cute::Shape<cute::Int<kBlockM>>{}, cute::Stride<_1>{});

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  cute::Tensor tOsO = gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  cute::Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  cute::Tensor tOrO = make_tensor<Element>(cute::shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  cute::Tensor caccO = make_identity_tensor(cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  cute::Tensor taccOcO = thr_mma.partition_C(caccO);                                                  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(cute::size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  cute::Tensor taccOcO_row = logical_divide(taccOcO, cute::Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(cute::size(lse) == cute::size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < cute::size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  cute::Tensor cO = make_identity_tensor(make_shape(cute::size<0>(sO), cute::size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  cute::Tensor tOpO = make_tensor<bool>(make_shape(cute::size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < cute::size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params& params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  using GmemTiledCopyO = std::conditional_t<
      !Split,
      typename Kernel_traits::GmemTiledCopyOaccum,
      typename Kernel_traits::GmemTiledCopyO>;
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
  // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
  const int n_block_min = n_split_idx * n_blocks_per_split;
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal) {
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
  }
  if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
    // Otherwise we might read OOB elements from gK and gV,
    // or get wrong results when we combine gOaccum from different blocks.
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    clear(tOrOaccum);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgOaccum); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
        gLSEaccum(row) = Split ? -INFINITY : INFINITY;
      }
    }
    return;
  }

  // We iterate over the blocks in reverse order. This is because the last block is the only one
  // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
  // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

  const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
  const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb) + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
  const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb) + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;

  Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                          Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_stride(params.k_row_stride, _1{}));
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
  Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_stride(params.v_row_stride, _1{}));
  // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
  // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
  // This maps to accessing the first 64 rows of knew_ptr.
  Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.knew_ptr) + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.knew_row_stride, _1{}));
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
  Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.vnew_ptr) + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.vnew_row_stride, _1{}));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);        // (KCPY, KCPY_N, KCPY_K)
  Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);        // (VCPY, VCPY_N, VCPY_K)
  Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // TODO: this might need to change if we change the mma instruction in SM70
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  //
  // PREDICATES
  //

  // // Allocate predicate tensors for m and n
  // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
  // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));   // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);     // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  // Prologue

  Tensor tQrQ = make_fragment_like(tQgQ);
  // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);

  int n_block = n_block_max - 1;
  // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
  flash::copy_2_sources</*Is_2_sources=*/Append_KV, Is_even_MN, Is_even_K>(
      gmem_tiled_copy_QKV, tKgK, tKgKnew, tKsK, tKVcKV, tKVpKV,
      binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
  cute::cp_async_fence();

  // flash::cp_async_wait<0>();
  // __syncthreads();
  // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
  // __syncthreads();

  clear(acc_o);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
  // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
  // We will have at least 1 "masking" iteration.

  // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
  // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
  constexpr int n_masking_steps = !Is_causal
                                      ? 1
                                      : (Is_even_MN ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    if constexpr (Append_KV) {
      // if (cute::thread0()) { print(tKgK); }
      // if (cute::thread0()) { print(tKsK); }
      // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("seqlen_k_cache = %d, (nblock + 1) * kBlockN = %d\n", binfo.seqlen_k_cache, (n_block + 1) * kBlockN); }
      if (bidh % params.h_h_k_ratio == 0 && binfo.seqlen_k_cache < (n_block + 1) * kBlockN) {
        flash::copy_w_min_idx<Is_even_K>(
            tKsK, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
      }
      // __syncthreads();
      // if (cute::thread0()) { print(tKgK); }
      // __syncthreads();
    }

    // Advance gV
    if (masking_step > 0) {
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      if (Append_KV) {
        tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
      }
      flash::copy_2_sources</*Is_2_sources=*/Append_KV, /*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV, tVgVnew, tVsV, tKVcKV, tKVpKV, 0, binfo.seqlen_k_cache - n_block * kBlockN);
    } else {
      // Clear the smem tiles to account for predicated off loads
      flash::copy_2_sources</*Is_2_sources=*/Append_KV, Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV, tVgV, tVgVnew, tVsV, tKVcKV, tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    // if (cute::thread0()) { print(scores); }
    // We don't put the masking before the matmul S = Q K^T because we don't clear sK
    // for rows outside actual_seqlen_k. So those rows could have Inf / NaN, and the matmul
    // can produce Inf / NaN.
    if (!Is_causal) {
      if (!Is_even_MN) {
        flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
      }
    } else {
      flash::apply_mask_causal(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                               m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                               binfo.actual_seqlen_q,
                               kNWarps * 16);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
    // __syncthreads();

    // if (tidx == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("n_block = %d, n_block_min = %d\n", n_block, n_block_min); }
    if constexpr (Append_KV) {
      // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("n_split_idx = %d, bidh = %d, params.h_h_k_ratio = %d, seqlen_k_cache = %d, (nblock + 1) * kBlockN = %d\n", n_split_idx, bidh, params.h_h_k_ratio, binfo.seqlen_k_cache, (n_block + 1) * kBlockN); }
      if (bidh % params.h_h_k_ratio == 0 && binfo.seqlen_k_cache < (n_block + 1) * kBlockN) {
        flash::copy_w_min_idx<Is_even_K>(
            tVsV, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
      }
    }

    if (n_block > n_block_min) {
      // Advance gK
      // if (tidx == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("tKgKnew = %p\n", tKgKnew.data()); }
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      if (Append_KV) {
        tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
      }
      // if (tidx == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("tKgKnew = %p, row_idx_switch = %d\n", tKgKnew.data(), binfo.seqlen_k_cache - (n_block - 1) * kBlockN); }
      flash::copy_2_sources</*Is_2_sources=*/Append_KV, /*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKgKnew, tKsK, tKVcKV, tKVpKV, 0,
          binfo.seqlen_k_cache - (n_block - 1) * kBlockN);
      // This cp_async_fence needs to be in the if block, otherwise the synchronization
      // isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // We have key_padding_mask so we'll need to Check_inf
    masking_step == 0
        ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || !Is_even_MN>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2)
        : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

    // Convert scores from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    // if (cute::thread0()) { print(scores); }

    // This check is at the end of the loop since we always have at least 1 iteration
    if (n_masking_steps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();
    if constexpr (Append_KV) {
      // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("n_split_idx = %d, bidh = %d, params.h_h_k_ratio = %d, seqlen_k_cache = %d, (nblock + 1) * kBlockN = %d\n", n_split_idx, bidh, params.h_h_k_ratio, binfo.seqlen_k_cache, (n_block + 1) * kBlockN); }
      if (bidh % params.h_h_k_ratio == 0 && binfo.seqlen_k_cache < (n_block + 1) * kBlockN) {
        flash::copy_w_min_idx<Is_even_K>(
            tKsK, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
      }
    }
    // Advance gV
    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    if (Append_KV) {
      tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
    }
    flash::copy_2_sources</*Is_2_sources=*/Append_KV, /*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV, tVgVnew, tVsV, tKVcKV, tKVpKV, 0, binfo.seqlen_k_cache - n_block * kBlockN);
    cute::cp_async_fence();

    flash::gemm(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);

    flash::cp_async_wait<0>();
    __syncthreads();
    if constexpr (Append_KV) {
      // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("seqlen_k_cache = %d, (nblock + 1) * kBlockN = %d\n", binfo.seqlen_k_cache, (n_block + 1) * kBlockN); }
      if (bidh % params.h_h_k_ratio == 0 && binfo.seqlen_k_cache < (n_block + 1) * kBlockN) {
        flash::copy_w_min_idx<Is_even_K>(
            tVsV, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
      }
    }
    if (n_block > n_block_min) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      if (Append_KV) {
        tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
      }
      flash::copy_2_sources</*Is_2_sources=*/Append_KV, /*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKgKnew, tKsK, tKVcKV, tKVpKV, 0,
          binfo.seqlen_k_cache - (n_block - 1) * kBlockN);
      // This cp_async_fence needs to be in the if block, otherwise the synchronization
      // isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // Epilogue

  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
  Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
  // if (cute::thread0()) { print(acc_o_rowcol); }
  Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = inv_sum;
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }
  // if (cute::thread0()) { print(lse); }
  // if (cute::thread0()) { print(acc_o_rowcol); }

  Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem_)), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  using SmemTiledCopyO = std::conditional_t<
      !Split,
      typename Kernel_traits::SmemCopyAtomO,
      typename Kernel_traits::SmemCopyAtomOaccum>;
  auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
  auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor rO = flash::convert_type<ElementO>(acc_o);
  Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);          // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sOaccum is larger than sQ, so we need to syncthreads here
  // TODO: allocate enough smem for sOaccum
  if constexpr (Split) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

  const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
  const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

  Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                               Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
  Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                 Shape<Int<kBlockM>>{}, Stride<_1>{});
  // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

  GmemTiledCopyO gmem_tiled_copy_Oaccum;
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

  __syncthreads();

  Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
  cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);                                // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
  // __syncthreads();
  // if (cute::thread0()) { print(tOgOaccum); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params& params) {
  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
  // them to have the same number of threads or have to traverse the attention matrix
  // in the same order.
  // In the Philox RNG, we use the offset to store the batch, head, and the lane id
  // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
  // the attention matrix. This way, as long as we have the batch, head, and the location of
  // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

  flash::compute_attn_1rowblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params& params) {
  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
  // The block index for the head.
  const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;
  flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params& params) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;
  constexpr int kMaxSplits = 1 << Log_max_splits;
  constexpr int kBlockM = 16;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
  // static_assert(kMaxSplits <= 8, "kMaxSplits must be <= 8 for now, will extend layer");
  static_assert(kBlockM == 16 || kBlockM == 32, "kBlockM must be 16 or 32");
  static_assert(Kernel_traits::kNThreads == 128, "We assume that each block has 128 threads");

  // Shared memory.
  // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
  __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

  // The thread and block index.
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  const index_t row_offset_lse = bidx * kBlockM;
  Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                 Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                 make_stride(params.b * params.h * params.seqlen_q, _1{}));
  Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
  constexpr int kNLsePerThread = (kMaxSplits * kBlockM + Kernel_traits::kNThreads - 1) / Kernel_traits::kNThreads;

  // Read the LSE values from gmem and store them in shared memory, then tranpose them.
  constexpr int kRowsPerLoadLSE = Kernel_traits::kNThreads / kBlockM;
#pragma unroll
  for (int l = 0; l < kNLsePerThread; ++l) {
    const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
    const int col = tidx % kBlockM;
    ElementAccum lse = (row < params.num_splits && col < params.b * params.h * params.seqlen_q - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
    if (row < kMaxSplits) {
      sLSE[row][col] = lse;
    }
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
  }
  // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
  __syncthreads();
  Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
  constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
  // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
  // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
  // 16 rows, so each time we load we can load 8 rows).
  // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
  // static_assert(kThreadsPerSplit <= 32);
  static_assert(kRowsPerLoadTranspose <= 32);
  static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
#pragma unroll
  for (int l = 0; l < kNLsePerThread; ++l) {
    const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
    const int col = tidx / kRowsPerLoadTranspose;
    lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
  }

  // Compute the logsumexp of the LSE along the split dimension.
  ElementAccum lse_max = lse_accum(0);
#pragma unroll
  for (int l = 1; l < kNLsePerThread; ++l) {
    lse_max = max(lse_max, lse_accum(l));
  }
  MaxOp<float> max_op;
  lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
  lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
  float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
  for (int l = 1; l < kNLsePerThread; ++l) {
    lse_sum += expf(lse_accum(l) - lse_max);
  }
  SumOp<float> sum_op;
  lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
  // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
  // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
  ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
  // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
  if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
    gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
  }
// Store the scales exp(lse - lse_logsum) in shared memory.
#pragma unroll
  for (int l = 0; l < kNLsePerThread; ++l) {
    const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
    const int col = tidx / kRowsPerLoadTranspose;
    if (row < params.num_splits && col < kBlockM) {
      sLSE[row][col] = expf(lse_accum(l) - lse_logsum);
    }
  }
  __syncthreads();

  const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
  Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + row_offset_oaccum),
                               Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               Stride<Int<kHeadDim>, _1>{});
  typename Kernel_traits::GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
  auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
  Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
  Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
  Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
  clear(tOrO);

  // Predicates
  Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  // Repeat the partitioning with identity layouts
  Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
  Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpOaccum); ++k) {
      tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d;
    }
  }
// Load Oaccum in then scale and accumulate to O
#pragma unroll 2
  for (int split = 0; split < params.num_splits; ++split) {
    flash::copy</*Is_even_MN=*/false, Is_even_K>(
        gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOrOaccum); ++m) {
      int row = get<0>(tOcOaccum(0, m, 0));
      ElementAccum lse_scale = sLSE[split][row];
#pragma unroll
      for (int k = 0; k < size<2>(tOrOaccum); ++k) {
#pragma unroll
        for (int i = 0; i < size<0>(tOrOaccum); ++i) {
          tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
        }
      }
      // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); print(tOrO); }
    }
    tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
  }
  // if (cute::thread0()) { print(tOrO); }

  Tensor rO = flash::convert_type<Element>(tOrO);
// Write to gO
#pragma unroll
  for (int m = 0; m < size<1>(rO); ++m) {
    const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
    if (idx < params.b * params.h * params.seqlen_q) {
      const int batch_idx = idx / (params.h * params.seqlen_q);
      const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
      // The index to the rows of Q
      const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
      auto o_ptr = reinterpret_cast<Element*>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
#pragma unroll
      for (int k = 0; k < size<2>(rO); ++k) {
        if (Is_even_K || tOpOaccum(k)) {
          const int col = get<1>(tOcOaccum(0, m, k));
          Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                  Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
          // TODO: Should check if this is using vectorized store, but it seems pretty fast
          copy(rO(_, m, k), gO);
          // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
          // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
        }
      }
    }
  }
}

}  // namespace flash
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
