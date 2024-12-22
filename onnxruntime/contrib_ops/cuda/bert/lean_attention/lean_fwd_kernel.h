// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <inttypes.h>

#include "contrib_ops/cuda/bert/lean_attention/block_info.h"
#include "contrib_ops/cuda/bert/lean_attention/kernel_traits.h"
#include "contrib_ops/cuda/bert/lean_attention/utils.h"
#include "contrib_ops/cuda/bert/lean_attention/softmax.h"
#include "contrib_ops/cuda/bert/lean_attention/mask.h"

namespace onnxruntime {
namespace lean {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Specialized for Prefill
template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, int kMaxSplits, bool Append_KV, typename Params>
inline __device__ void lean_compute_attn_impl_ver3(const Params& params, const int cta_id, int start_tile_gid, int start_tile_hid, int num_tiles, const int num_tiles_per_head) {
#if defined(DEBUG_LEAN_ATTENTION)
  // Timing
  auto kernel_start = clock64();
  long long int comp1_duration = 0;
  long long int comp2_duration = 0;
  long long int epilogue_duration = 0;
  long long int prologue_duration = 0;
  long long int epil1_duration = 0;
  long long int epil2_duration = 0;
  long long int epil3_duration = 0;

  const int tracing_block = 0;
#endif

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

  using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;
  using GmemTiledCopyOaccum = typename Kernel_traits::GmemTiledCopyOaccum;

  const int num_m_blocks_per_head = (params.seqlen_q + kBlockM - 1) / kBlockM;

  // // This is the solution to the summation series (n+1)(n+2)/2 = start_tile_hid + 1
  // int cur_m_block = Is_causal ? (int)ceilf((sqrtf(9 + (8*start_tile_hid)) - 3) / 2) : start_tile_hid/num_tiles_per_head;
  float block_scale = (float)kBlockM / (float)kBlockN;
  int cur_m_block = Is_causal ? kBlockM > kBlockN ? (int)ceilf((sqrtf(1 + (8 * start_tile_hid + 8) / block_scale) - 3) / 2)
                                                  // : (int)((-1 + sqrt(1 + 8 * block_scale * start_tile_hid)) / 2) * (1 / block_scale) + (int)((start_tile_hid - (1 / block_scale) * ((int)((-1 + sqrt(1 + 8 * block_scale * start_tile_hid)) / 2) * ((int)((-1 + sqrt(1 + 8 * block_scale * start_tile_hid)) / 2) + 1) / 2)) / ((int)((-1 + sqrt(1 + 8 * block_scale * start_tile_hid)) / 2) + 1))
                                                  : static_cast<int>((-1 + sqrt(1 + 8 * start_tile_hid * block_scale)) / (2 * block_scale))
                              : start_tile_hid / num_tiles_per_head;
  int num_tiles_in_block = Is_causal ? (int)ceilf(block_scale * (cur_m_block + 1)) : num_tiles_per_head;
  int cur_bidb = start_tile_gid / (num_tiles_per_head * params.h);
  int cur_bidh = (start_tile_gid - (cur_bidb * num_tiles_per_head * params.h)) / num_tiles_per_head;

  int num_tiles_left = num_tiles;

#if defined(DEBUG_LEAN_ATTENTION)
  if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
    printf("Debugging block = %d\n", tracing_block);
    printf("kBlockM = %d\n", kBlockM);
    printf("kBlockN = %d\n", kBlockN);
    printf("kHeadDim = %d\n", kHeadDim);
    printf("kNWarps = %d\n", kNWarps);
    printf("IsEvenMN = %d\n", Is_even_MN);
    printf("block_scale = %f\n", block_scale);
    printf("seq_len_q -change = %d\n", params.seqlen_q);
    printf("seq_len_k = %d\n", params.seqlen_k);
    printf("q_batch_stride = %ld\n", params.q_batch_stride);
    printf("q_head_stride = %ld\n", params.q_head_stride);
    printf("q_row_stride = %ld\n", params.q_row_stride);
    printf("k_batch_stride = %ld\n", params.k_batch_stride);
    printf("k_head_stride = %ld\n", params.k_head_stride);
    printf("k_row_stride = %ld\n", params.k_row_stride);
    printf("v_row_stride = %ld\n", params.v_row_stride);
    printf("o_row_stride = %ld\n", params.o_row_stride);
    printf("start_m_block = %d\n", cur_m_block);
    printf("start_tile_gid = %d\n", start_tile_gid);
    printf("start_tile_hid = %d\n", start_tile_hid);
    printf("cur_bidb = %d/%d\n", cur_bidb, params.b);
    printf("cur_bidh = %d/%d\n", cur_bidh, params.h);
    printf("num_m_blocks_per_head = %d\n", num_m_blocks_per_head);
    printf("cur_m_block = %d\n", cur_m_block);
    printf("num_tiles_in_block = %d\n", num_tiles_in_block);
    printf("Total tiles = %d\n", num_tiles);
  }
#endif

  // Prologue
  int n_tile_min = kBlockM > kBlockN ? start_tile_hid - (block_scale * cur_m_block * (cur_m_block + 1) / 2)
                                     : start_tile_hid - (int)(((int)floorf(cur_m_block * block_scale) * ((int)floorf(cur_m_block * block_scale) + 1) / 2) / block_scale) - ((cur_m_block % int(1 / block_scale)) * (floorf(cur_m_block * block_scale) + 1));
  int n_tile = n_tile_min + num_tiles_left - 1 >= num_tiles_in_block ? num_tiles_in_block - 1 : n_tile_min + num_tiles_left - 1;

  index_t row_offset_q = cur_bidb * params.q_batch_stride +
                         +cur_m_block * kBlockM * params.q_row_stride + cur_bidh * params.q_head_stride;
  index_t row_offset_k = cur_bidb * params.k_batch_stride +
                         +n_tile * kBlockN * params.k_row_stride + (cur_bidh / params.h_h_k_ratio) * params.k_head_stride;

  Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                          Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_stride(params.k_row_stride, _1{}));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

  // PREDICATES
  //

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

  // // Start from the last block of first head
  // lean::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
  //                                 params.seqlen_q - cur_m_block * kBlockM);

  // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
  // lean::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
  //                                 params.seqlen_k - n_tile * kBlockN);
  // cute::cp_async_fence();

  index_t row_offset_v = cur_bidb * params.v_batch_stride +
                         +n_tile * kBlockN * params.v_row_stride + (cur_bidh / params.h_h_k_ratio) * params.v_head_stride;
  Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_stride(params.v_row_stride, _1{}));
  Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // Tiled Matrix Multiply
  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling - Can be moved
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

#if defined(DEBUG_LEAN_ATTENTION)
  if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
    printf("n_tile_min = %d\n", n_tile_min);
    printf("n_tile = %d\n", n_tile);
    printf("row_offset_q = %" PRId64 "\n", row_offset_q);
    printf("row_offset_k = %" PRId64 "\n", row_offset_k);
    printf("row_offset_v = %" PRId64 "\n", row_offset_v);
  }

  int num_blocks = 0;
#endif

  for (; num_tiles_left > 0;) {
#if defined(DEBUG_LEAN_ATTENTION)
    num_blocks += 1;
    auto prologue_start = clock64();
#endif

    cur_bidb = start_tile_gid / (num_tiles_per_head * params.h);
    cur_bidh = (start_tile_gid - (cur_bidb * num_tiles_per_head * params.h)) / num_tiles_per_head;
    // Scheduling Policy - below

    // Calculate split ID
    int block_start_gid = start_tile_gid - n_tile_min;
    int cta_id_block_start = block_start_gid > params.high_load_tbs * params.max_tiles_per_tb
                                 ? params.high_load_tbs + ((block_start_gid - (params.high_load_tbs * params.max_tiles_per_tb)) / (params.max_tiles_per_tb - 1))
                                 : block_start_gid / params.max_tiles_per_tb;
    int n_split_idx = cta_id - cta_id_block_start;

    // Check host/
    int host_cta = 0;
    int total_splits = 1;
    if (n_tile_min == 0) {
      host_cta = 1;
      int block_end_gid = start_tile_gid + num_tiles_in_block - 1;
      int cta_id_block_end = block_end_gid > params.high_load_tbs * params.max_tiles_per_tb
                                 ? params.high_load_tbs + ((block_end_gid - (params.high_load_tbs * params.max_tiles_per_tb)) / (params.max_tiles_per_tb - 1))
                                 : block_end_gid / params.max_tiles_per_tb;
      total_splits = cta_id_block_end - cta_id + 1;
    }

    int end_cta = 0;
    if (n_tile == num_tiles_in_block - 1) {
      end_cta = 1;
    }

    start_tile_gid += n_tile - n_tile_min + 1;
    start_tile_hid += n_tile - n_tile_min + 1;
    if (start_tile_hid >= num_tiles_per_head) {
      // Next head
      start_tile_hid = 0;
    }
    num_tiles_left -= n_tile - n_tile_min + 1;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, cur_bidb);
    // This is a hack, we really need to handle this outside the kernel
    // But can't figure out a way to get actual seqlen_k in host-side code.
    int max_actual_tiles = (binfo.actual_seqlen_k + kBlockN - 1) / kBlockN;
    int num_actual_tiles_in_block = Is_causal ? std::max(max_actual_tiles, (int)ceilf(block_scale * (cur_m_block + 1))) : max_actual_tiles;
    if (n_tile >= max_actual_tiles) {
      tKgK.data() = tKgK.data() + (-int((n_tile - max_actual_tiles - 1) * kBlockN * params.k_row_stride));
      tVgV.data() = tVgV.data() + (-int((n_tile - max_actual_tiles - 1) * kBlockN * params.v_row_stride));
      n_tile = max_actual_tiles - 1;
    }
    if constexpr (Append_KV) {
      if (end_cta) {
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.

        const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, cur_bidb) + (n_tile * kBlockN) * params.knew_row_stride + (cur_bidh / params.h_h_k_ratio) * params.knew_head_stride;
        const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, cur_bidb) + (n_tile * kBlockN) * params.vnew_row_stride + (cur_bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.knew_ptr) + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.knew_row_stride, _1{}));
#if defined(DEBUG_LEAN_ATTENTION)
        if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
          printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data());
        }
#endif
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.vnew_ptr) + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                   make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

        const int n_block_copy_min = std::max(n_tile_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();

#if defined(DEBUG_LEAN_ATTENTION)
        if (threadIdx.x == 0 && (blockIdx.z == tracing_block || blockIdx.z == tracing_block + 1)) {
          printf("Block %d n_tile_min %d n_tile %d n_block_copy_min %d\n", blockIdx.z, n_tile_min, n_tile, n_block_copy_min);
        }
#endif
        for (int n_block = n_tile; n_block >= n_block_copy_min; n_block--) {
          lean::copy_w_min_idx<Is_even_K>(
              tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
          tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));

          lean::copy_w_min_idx<Is_even_K>(
              tKgKnew, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN);
          tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
          tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
          tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
      }
    }
    lean::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                      binfo.actual_seqlen_q - cur_m_block * kBlockM);
    lean::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                      binfo.actual_seqlen_k - n_tile * kBlockN);
    cute::cp_async_fence();

#if defined(DEBUG_LEAN_ATTENTION)
    if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
      printf("##### CTA : %d\n", blockIdx.z);
      printf("cur_bidb = %d/%d\n", cur_bidb, params.b);
      printf("cur_bidh = %d/%d\n", cur_bidh, params.h);
      printf("cur_m_block = %d\n", cur_m_block);
      printf("seqlen_k_cache = %d\n", binfo.seqlen_k_cache);
      printf("actual_seqlen_q = %d\n", binfo.actual_seqlen_q);
      printf("actual_seqlen_k = %d\n", binfo.actual_seqlen_k);
      printf("num_tiles_in_block = %d\n", num_tiles_in_block);
      printf("n_tile(new) = %d\n", n_tile);
      printf("n_tile_min = %d\n", n_tile_min);
      printf("host_cta = %d\n", host_cta);
      printf("end_cta = %d\n", end_cta);
      printf("n_split_idx = %d\n", n_split_idx);
      printf("total_splits = %d\n", total_splits);
      printf("\n#### For next block:\n");
      printf("start_tile_gid = %d\n", start_tile_gid);
      printf("start_tile_hid = %d\n", start_tile_hid);
      printf("num_tiles_left = %d\n", num_tiles_left);
      printf("\n");
    }
#endif

    // All scheduling policy decisions should be made above this line
    clear(acc_o);

    lean::Softmax<2 * size<1>(acc_o)> softmax;

    lean::Mask<Is_causal, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, 0.0f);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    lean::cp_async_wait<0>();
    __syncthreads();

#if defined(DEBUG_LEAN_ATTENTION)
    prologue_duration += clock64() - prologue_start;
    auto compute_start = clock64();
#endif

    // Clear the smem tiles to account for predicated off loads
    lean::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_tile * kBlockN);
    cute::cp_async_fence();

    lean::gemm(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);

#if defined(DEBUG_LEAN_ATTENTION)
    if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
      printf("Tile 0 - Svalue: acc_s[0] = %f\n", acc_s(0));
    }
#endif

    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, n_tile * kBlockN, cur_m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

    lean::cp_async_wait<0>();
    __syncthreads();

#if defined(DEBUG_LEAN_ATTENTION)
    if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      print(tVsV);
    }
    // __syncthreads();
#endif

    if (n_tile > n_tile_min) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      lean::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the synchronization
      // isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // We have key_padding_mask so we'll need to Check_inf
    softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);

#if defined(DEBUG_LEAN_ATTENTION)
    if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
      printf("Tile 0 - PValue[0] = %f\n", acc_s(0));
    }
#endif

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = lean::convert_type<Element>(acc_s);
    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(rP.data(), lean::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    lean::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

#if defined(DEBUG_LEAN_ATTENTION)
    if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
      printf("Tile 0 - AfterPV[0] = %f\n", acc_o(0));
    }
#endif

    n_tile -= 1;

#if defined(DEBUG_LEAN_ATTENTION)
    comp1_duration += clock64() - compute_start;
    compute_start = clock64();
#endif

    // These are the iterations where we don't need masking on S
    for (; n_tile >= n_tile_min; --n_tile) {
      Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
      clear(acc_s);
      lean::cp_async_wait<0>();
      __syncthreads();

      // Advance gV
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));

      lean::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
      cute::cp_async_fence();

      lean::gemm(
          acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K);
#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("ntile %d Svalue: acc_s[0] = %f\n", n_tile, acc_s(0));
      }
#endif

      lean::cp_async_wait<0>();
      __syncthreads();
      if (n_tile > n_tile_min) {
        // Advance gK
        tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        lean::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
        // This cp_async_fence needs to be in the if block, otherwise the synchronization
        // isn't right and we get race conditions.
        cute::cp_async_fence();
      }

      mask.template apply_mask</*Causal_mask=*/false>(
          acc_s, n_tile * kBlockN, cur_m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);
      softmax.template softmax_rescale_o</*Is_first=*/false, false>(acc_s, acc_o, params.scale_softmax_log2);

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("ntile %d Pvalue: acc_s[0] = %f\n", n_tile, acc_s(0));
      }
#endif
      Tensor rP = lean::convert_type<Element>(acc_s);

      // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
      // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
      Tensor tOrP = make_tensor(rP.data(), lean::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

      lean::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("ntile %d AfterPV[0] = %f\n", n_tile, acc_o(0));
      }
#endif
    }

#if defined(DEBUG_LEAN_ATTENTION)
    // Epilogue
    comp2_duration += clock64() - compute_start;
    auto epilogue_start = clock64();
#endif

    if (host_cta && end_cta) {
#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("acc_o[0] = %f\n", acc_o(0));
      }
#endif

      Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, params.scale_softmax, params.rp_dropout);

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("lse[0] = %f\n", lse(0));
        printf("acc_o[0] = %f\n", acc_o(0));
      }
#endif

      // Convert acc_o from fp32 to fp16/bf16
      Tensor rO = lean::convert_type<Element>(acc_o);

      Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
      // Partition sO to match the accumulator partitioning
      auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
      auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
      Tensor taccOrO = smem_thr_copy_O.retile_S(rO);     // ((Atom,AtomNum), MMA_M, MMA_N)
      Tensor taccOsO = smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

      // sO has the same size as sQ, so we don't need to sync here.
      if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
      }

      cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

      const index_t row_offset_o = cur_bidb * params.o_batch_stride +
                                   cur_m_block * kBlockM * params.o_row_stride + cur_bidh * params.o_head_stride;

      Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_stride(params.o_row_stride, _1{}));

      typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
      auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
      Tensor tOsO = gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
      Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

      __syncthreads();

      Tensor tOrO = make_tensor<Element>(shape(tOgO));
      cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

      // Construct identity layout for sO
      Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
      // Repeat the partitioning with identity layouts
      Tensor tOcO = gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
      Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
      if (!Is_even_K) {
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
          tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
      }
#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("tOpO[0] = %d\n", tOpO(0));
        printf("tOrO[0] = %f\n", tOrO(0));
      }
#endif
      // Clear_OOB_K must be false since we don't want to write zeros to gmem
      lean::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, params.seqlen_q - cur_m_block * kBlockM);
      // epil1_duration += clock64() - epilogue_start;
    } else if (!host_cta) {
      Tensor lse = softmax.template normalize_softmax_lse<false, true>(acc_o, params.scale_softmax);

      Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum*>(smem_)), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
      // Partition sO to match the accumulator partitioning
      using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomOaccum;
      auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
      auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
      Tensor rO = lean::convert_type<ElementAccum>(acc_o);
      Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);          // ((Atom,AtomNum), MMA_M, MMA_N)
      Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

      // sOaccum is larger than sQ, so we need to syncthreads here
      // TODO: allocate enough smem for sOaccum
      __syncthreads();

      cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

      const index_t row_offset_oaccum = (((index_t)(n_split_idx * params.b + cur_bidb) * params.h + cur_bidh) * params.seqlen_q + cur_m_block * kBlockM) * params.d_rounded;
      const index_t row_offset_lseaccum = ((n_split_idx * params.b + cur_bidb) * params.h + cur_bidh) * params.seqlen_q + cur_m_block * kBlockM;

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("n_split_idx = %d\n", n_split_idx);
        // printf("row_offset_o = %" PRId64 "\n", row_offset_o);
        printf("row_offset_oaccum = %" PRId64 "\n", row_offset_oaccum);
        printf("row_offset_lseaccum = %" PRId64 "\n", row_offset_lseaccum);
      }
#endif

      Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + (row_offset_oaccum)),
                                   Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                   make_stride(kHeadDim, _1{}));
      Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                     Shape<Int<kBlockM>>{}, Stride<_1>{});

      GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
      auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
      Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
      Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

      __syncthreads();

      Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
      cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

      Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
      Tensor taccOcO = thr_mma.partition_C(caccO);                                // (MMA,MMA_M,MMA_K)
      static_assert(decltype(size<0>(taccOcO))::value == 4);
      // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
      Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
      CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
      // This partitioning is unequal because only threads 0,4,8,etc write to gLSE
      // and the rest are unused.
      if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
          const int row = get<0>(taccOcO_row(mi));
          if (row < params.seqlen_q - cur_m_block * kBlockM) {
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
      lean::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - cur_m_block * kBlockM);

      __threadfence();

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && (blockIdx.z == tracing_block || blockIdx.z == tracing_block + 1)) {
        printf("Block %d Writing Flag %d\n", blockIdx.z, (cur_bidb * params.h * num_m_blocks_per_head) + (cur_bidh * num_m_blocks_per_head) + cur_m_block);
      }
#endif

      atomicAdd(reinterpret_cast<int32_t*>(params.sync_flag) + (cur_bidb * params.h * num_m_blocks_per_head) + (cur_bidh * num_m_blocks_per_head) + cur_m_block, 1);

#if defined(DEBUG_LEAN_ATTENTION)
      epil2_duration += clock64() - epilogue_start;
#endif
    } else {
      constexpr int kNThreads = Kernel_traits::kNThreads;

      static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
      static_assert(kNThreads == 128, "We assume that each block has 128 threads");

      ////////////////////////////////////////////////////////////////////////////////
#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("Before LSE acc_o[0] = %f\n", acc_o(0));
      }
#endif

      Tensor lse = softmax.template normalize_softmax_lse<false, true>(acc_o, params.scale_softmax);

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("After LSE acc_o[0] = %f\n", acc_o(0));
        printf("lse[0] = %f\n", lse(0));
      }
#endif

      Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum*>(smem_)), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
      // Partition sO to match the accumulator partitioning
      using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomOaccum;
      auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
      auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
      Tensor rO = lean::convert_type<ElementAccum>(acc_o);
      Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);          // ((Atom,AtomNum), MMA_M, MMA_N)
      Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

      // sOaccum is larger than sQ, so we need to syncthreads here
      // TODO: allocate enough smem for sOaccum
      __syncthreads();

      // We move to SMEM and back because we need equal distribution of
      // accum registers. Initially only threads 0,4,8,etc have oaccum values.
      // So, first move them to SMEM.
      cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

      const index_t row_offset_oaccum = ((cur_bidb * params.h + cur_bidh) * (index_t)params.seqlen_q + cur_m_block * kBlockM) * params.d_rounded;
      const index_t row_offset_lseaccum = (cur_bidb * params.h + cur_bidh) * (index_t)params.seqlen_q + cur_m_block * kBlockM;

      Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + (row_offset_oaccum)),
                                   Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                   make_stride(kHeadDim, _1{}));
      Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                     Shape<Int<kBlockM>>{}, Stride<_1>{});

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("Block %d row_offset_oaccum = %" PRId64 "\n", blockIdx.z, row_offset_oaccum);
        printf("Block %d row_offset_lseaccum = %" PRId64 "\n", blockIdx.z, row_offset_lseaccum);
      }
#endif

      // GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
      // auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
      // Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
      // Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

      constexpr int kBlockN = kNThreads / kBlockM;
      using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
      using GmemTiledCopyOaccum = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                                                           GmemLayoutAtomOaccum{},
                                                           Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
      GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
      auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);

      Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);
      Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
      Tensor tOgOaccumReg = gmem_thr_copy_Oaccum.partition_D(gOaccum);
      Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccumReg));

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("First split t0g0accum.data() %p\n", tOgOaccum.data());
      }
#endif

      __syncthreads();

      // Bring the oaccum back from SMEM to registers
      // Now all threads have oaccum values equaly distributed.
      cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

      /////////////////////////////////////////////////////////////////////////////

      // Shared memory.
      // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
      Tensor sLSE = make_tensor(sV.data(), Shape<Int<kMaxSplits>, Int<kBlockM + 1>>{});  // (SMEM_M,SMEM_N)

      Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
      Tensor taccOcO = thr_mma.partition_C(caccO);                                // (MMA,MMA_M,MMA_K)
      static_assert(decltype(size<0>(taccOcO))::value == 4);
      // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
      Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
      CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M

      // This partitioning is unequal because only threads 0,4,8,etc write to gLSE
      // and the rest are unused.
      if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
          const int col = get<0>(taccOcO_row(mi));
          if (col < params.seqlen_q - cur_m_block * kBlockM) {
            sLSE(0, col) = lse(mi);
#if defined(DEBUG_LEAN_ATTENTION)
            if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
              printf("threadIdx.x %d col %d mi%d slSE %f\n", threadIdx.x, col, mi, lse(mi));
            }
#endif
          }
        }
      }

      // Synchronize here to make sure all atomics are visible to all threads.
      // Not exactly sure why we need this, but it seems to be necessary.
      __threadfence();
      while (atomicAdd(reinterpret_cast<int32_t*>(params.sync_flag) +
                           (cur_bidb * params.h * num_m_blocks_per_head) +
                           (cur_bidh * num_m_blocks_per_head) + cur_m_block,
                       0) < (total_splits - 1) * kNThreads) {
        __threadfence();
#if defined(DEBUG_LEAN_ATTENTION)
        if (threadIdx.x % 32 == 0 && blockIdx.z == tracing_block) {
          printf("Waiting Block: %d target-value: %d\n", blockIdx.z, (total_splits - 1) * kNThreads);
        }
#endif
      }

#if defined(DEBUG_LEAN_ATTENTION)
      // Print sync flag value
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        int32_t sync_flag = atomicAdd(reinterpret_cast<int32_t*>(params.sync_flag) +
                                          (cur_bidb * params.h * num_m_blocks_per_head) +
                                          (cur_bidh * num_m_blocks_per_head) + cur_m_block,
                                      0);
        if (threadIdx.x % 32 == 0 && blockIdx.z == tracing_block) {
          printf("Sync flag value: %d\n", sync_flag);
        }
      }
#endif

      Tensor gLSEaccumRead = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                         Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                         make_stride(params.b * params.h * params.seqlen_q, _1{}));
      // Read the LSE values from gmem and store them in shared memory, then tranpose them.
      constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;  // R
      constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;                                // R

#pragma unroll
      for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        // We skip the first row = 0, as we already populated it in shared memory.
        ElementAccum lse = (row > 0 && row < total_splits && col < params.b * params.h * (index_t)params.seqlen_q - row_offset_lseaccum) ? gLSEaccumRead(row, col) : -std::numeric_limits<ElementAccum>::infinity();
        if (row > 0 && row < kMaxSplits) {
          sLSE(row, col) = lse;

#if defined(DEBUG_LEAN_ATTENTION)
          if (threadIdx.x % 32 == 0 && blockIdx.z == tracing_block) {
            printf("ThreadIdx %d l %d row %d col %d lse %f\n", threadIdx.x, l, row, col, lse);
          }
#endif
        }
      }
      __syncthreads();  // For all LSEs to reach shared memory
      Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
      constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("kNLsePerThread %d kRowsPerLoadLSE %d kRowsPerLoadTranspose %d\n", kNLsePerThread, kRowsPerLoadLSE, kRowsPerLoadTranspose);
      }
#endif

      // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
      // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
      // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
      // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
      // static_assert(kThreadsPerSplit <= 32);
      static_assert(kRowsPerLoadTranspose <= 32);
      static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
#pragma unroll
      for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE(row, col) : -std::numeric_limits<ElementAccum>::infinity();

#if defined(DEBUG_LEAN_ATTENTION)
        if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
          printf("ThreadIdx %d l %d row %d col %d lse_accum %f\n", threadIdx.x, l, row, col, lse_accum(l));
        }
#endif
      }

      // Compute the logsumexp of the LSE along the split dimension.
      ElementAccum lse_max = lse_accum(0);
#pragma unroll
      for (int l = 1; l < kNLsePerThread; ++l) {
        lse_max = max(lse_max, lse_accum(l));
      }
      MaxOp<float> max_op;
      lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
      lse_max = lse_max == -std::numeric_limits<ElementAccum>::infinity() ? 0.0f : lse_max;  // In case all local LSEs are -inf
      float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
      for (int l = 1; l < kNLsePerThread; ++l) {
        lse_sum += expf(lse_accum(l) - lse_max);
      }
      SumOp<float> sum_op;
      lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
      // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
      // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
      ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum)
                                    ? std::numeric_limits<ElementAccum>::infinity()
                                    : logf(lse_sum) + lse_max;
// if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) { gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum; }
// Store the scales exp(lse - lse_logsum) in shared memory.
#pragma unroll
      for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < total_splits && col < kBlockM) {
          sLSE(row, col) = expf(lse_accum(l) - lse_logsum);
          ElementAccum lse_scale = sLSE(row, col);
#if defined(DEBUG_LEAN_ATTENTION)
          if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
            printf("ThreadIdx %d l %d row %d col %d lse_accum %f lse_logsum %f sLSE %f\n", threadIdx.x, l, row, col, lse_accum(l), lse_logsum, lse_scale);
          }
#endif
        }
      }

      Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
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

      // Sync here for sLSE stores to go through
      __syncthreads();
// First reduce self Oaccum
#pragma unroll
      for (int m = 0; m < size<1>(tOrOaccum); ++m) {
        int row = get<0>(tOcOaccum(0, m, 0));
        ElementAccum lse_scale = sLSE(0, row);
#pragma unroll
        for (int k = 0; k < size<2>(tOrOaccum); ++k) {
#pragma unroll
          for (int i = 0; i < size<0>(tOrOaccum); ++i) {
            tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
#if defined(DEBUG_LEAN_ATTENTION)
            if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
              printf("ThreadIdx %d Split %d m %d Row %d k %d i %d LSE %f Oaccum %f O %f\n", threadIdx.x, 0, m, row, k, i, lse_scale, tOrOaccum(i, m, k), tOrO(i, m, k));
            }
#endif
          }
        }
      }

      tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * (index_t)params.seqlen_q * params.d_rounded;

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("After First Split t0g0accum.data() %p\n", tOgOaccum.data());
      }
#endif
      // Load Oaccum in then scale and accumulate to O
      // Here m is each row of 0accum along token dimension
      // k is
      for (int split = 1; split < total_splits; ++split) {
        lean::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * (index_t)params.seqlen_q - row_offset_lseaccum);
#pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
          int row = get<0>(tOcOaccum(0, m, 0));
          ElementAccum lse_scale = sLSE(split, row);
#pragma unroll
          for (int k = 0; k < size<2>(tOrOaccum); ++k) {
#pragma unroll
            for (int i = 0; i < size<0>(tOrOaccum); ++i) {
              tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
#if defined(DEBUG_LEAN_ATTENTION)
              if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
                printf("ThreadIdx %d Split %d m %d Row %d k %d i %d LSE %f Oaccum %f O %f\n", threadIdx.x, split, m, row, k, i, lse_scale, tOrOaccum(i, m, k), tOrO(i, m, k));
              }
#endif
            }
          }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * (index_t)params.seqlen_q * params.d_rounded;
      }

      Tensor r1 = lean::convert_type<Element>(tOrO);

// Write to gO
#pragma unroll
      for (int m = 0; m < size<1>(r1); ++m) {
        const int idx = cur_m_block * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.seqlen_q) {
          // The index to the rows of Q
          const int row = idx;
          auto o_ptr = reinterpret_cast<Element*>(params.o_ptr) + cur_bidb * params.o_batch_stride + cur_bidh * params.o_head_stride + row * params.o_row_stride;
#pragma unroll
          for (int k = 0; k < size<2>(r1); ++k) {
            if (Is_even_K || tOpOaccum(k)) {
              const int col = get<1>(tOcOaccum(0, m, k));
              Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                      Shape<Int<decltype(size<0>(r1))::value>>{}, Stride<_1>{});
              copy(r1(_, m, k), gO);
            }
          }
        }
      }
#if defined(DEBUG_LEAN_ATTENTION)
      epil3_duration += clock64() - epilogue_start;
#endif
    }

    if (num_tiles_left) {
      // We can probably do better than this
      // We first decrement the pointers back to starting.
      // We can probably just use q_ptr and k_ptr directly. But can't figure out how to do it.
      // Without disturbing the gQ, gK, gV tensor pointer CUTE objects.
      tQgQ.data() = tQgQ.data() + (-int(row_offset_q));
      tKgK.data() = tKgK.data() + (((num_tiles_in_block - n_tile_min - 1) * kBlockN) * params.k_row_stride - row_offset_k);
      tVgV.data() = tVgV.data() + (((num_tiles_in_block - n_tile_min - 1) * kBlockN) * params.v_row_stride - row_offset_v);
      cur_m_block = cur_m_block + 1 >= num_m_blocks_per_head ? 0 : cur_m_block + 1;
      num_tiles_in_block = Is_causal ? (int)ceilf(block_scale * (cur_m_block + 1)) : num_tiles_per_head;
      n_tile = num_tiles_left - 1 >= num_tiles_in_block ? num_tiles_in_block - 1 : num_tiles_left - 1;
      n_tile_min = 0;
      cur_bidb = start_tile_gid / (num_tiles_per_head * params.h);
      cur_bidh = (start_tile_gid - (cur_bidb * num_tiles_per_head * params.h)) / num_tiles_per_head;

      row_offset_q = cur_bidb * params.q_batch_stride +
                     +cur_m_block * kBlockM * params.q_row_stride + cur_bidh * params.q_head_stride;
      row_offset_k = cur_bidb * params.k_batch_stride +
                     +n_tile * kBlockN * params.k_row_stride + (cur_bidh / params.h_h_k_ratio) * params.k_head_stride;
      row_offset_v = cur_bidb * params.v_batch_stride +
                     +n_tile * kBlockN * params.v_row_stride + (cur_bidh / params.h_h_k_ratio) * params.v_head_stride;

      tQgQ.data() = tQgQ.data() + row_offset_q;
      tKgK.data() = tKgK.data() + row_offset_k;
      tVgV.data() = tVgV.data() + row_offset_v;

#if defined(DEBUG_LEAN_ATTENTION)
      if (threadIdx.x == 0 && blockIdx.z == tracing_block) {
        printf("#### Ready for next block:\n");
        printf("next_block %d\n", cur_m_block);
        printf("n_tile %d\n", n_tile);
        printf("row_offset_q = %" PRId64 "\n", row_offset_q);
        printf("row_offset_k = %" PRId64 "\n", row_offset_k);
        printf("row_offset_v = %" PRId64 "\n", row_offset_v);
      }
#endif
    }

#if defined(DEBUG_LEAN_ATTENTION)
    epilogue_duration += clock64() - epilogue_start;
#endif
  }

#if defined(DEBUG_LEAN_ATTENTION)
  if (threadIdx.x == 0) {
    uint smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    printf("%d %d %d %d %lld %lld %lld %lld %lld %lld %lld %lld\n",
           blockIdx.z, num_blocks, smid, cta_id, clock64() - kernel_start, prologue_duration, comp1_duration,
           comp2_duration, epilogue_duration, epil1_duration, epil2_duration, epil3_duration);
  }
#endif
}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, int kMaxSplits, bool Append_KV, typename Params>
inline __device__ void lean_compute_attn(const Params& params) {
  // const int cta_id = blockIdx.z < 54 ? 4*blockIdx.z : blockIdx.z < 108 ? 4*(blockIdx.z % 54) + 2 : blockIdx.z < 162 ? 4*(blockIdx.z % 108) + 1 : 4*(blockIdx.z % 162) + 3;
  const int cta_id = blockIdx.z;
  int start_tile_gid = cta_id < params.high_load_tbs ? params.max_tiles_per_tb * cta_id : (params.max_tiles_per_tb - 1) * cta_id + params.high_load_tbs;
  int start_tile_hid = start_tile_gid % params.tiles_per_head;
  int num_tiles = cta_id < params.high_load_tbs ? params.max_tiles_per_tb : params.max_tiles_per_tb - 1;

  lean::lean_compute_attn_impl_ver3<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, kMaxSplits, Append_KV>(params, cta_id, start_tile_gid, start_tile_hid, num_tiles, params.tiles_per_head);
}

}  // namespace lean
}  // namespace onnxruntime
