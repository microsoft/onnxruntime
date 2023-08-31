/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cuda.h>
#include <vector>

namespace onnxruntime {
namespace flash {

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = uint32_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  int h_h_k_ratio;  // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  // The O matrix (output).
  void* __restrict__ o_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the P matrix.
  void* __restrict__ p_ptr;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;

  int* __restrict__ blockmask;

  bool is_bf16 = false;
  bool is_causal;

  const cudaDeviceProp* dprops;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Headdim>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace flash
}  // namespace onnxruntime