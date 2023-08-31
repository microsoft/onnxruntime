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
    copy(scores_max, scores_max_prev);
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
    cute::Tensor<Engine0, Layout0> const& tOrP, cute::Tensor<Engine1, Layout1>& tPgP, TiledCopy gmem_thr_copy_P) {
  // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
  cute::Layout l = tOrP.layout();
  cute::Tensor tPrP = make_tensor(tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
  CUTE_STATIC_ASSERT_V(cute::size<2>(tPgP) == _1{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(tPrP) == cute::size<1>(tPgP));
#pragma unroll
  for (int mi = 0; mi < cute::size<1>(tPrP); ++mi) {
    copy(gmem_thr_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
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
}  // namespace flash
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
