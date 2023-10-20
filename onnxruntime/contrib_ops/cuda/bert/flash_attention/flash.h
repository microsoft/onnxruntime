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
  void* __restrict__ q_ptr = nullptr;
  void* __restrict__ k_ptr = nullptr;
  void* __restrict__ v_ptr = nullptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride = 0;
  index_t k_batch_stride = 0;
  index_t v_batch_stride = 0;
  index_t q_row_stride = 0;
  index_t k_row_stride = 0;
  index_t v_row_stride = 0;
  index_t q_head_stride = 0;
  index_t k_head_stride = 0;
  index_t v_head_stride = 0;

  // The number of heads.
  int h = 0;
  int h_k = 0;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  int h_h_k_ratio = 0;  // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  // The O matrix (output).
  void* __restrict__ o_ptr = nullptr;
  void* __restrict__ oaccum_ptr = nullptr;

  // The stride between rows of O.
  index_t o_batch_stride = 0;
  index_t o_row_stride = 0;
  index_t o_head_stride = 0;

  // The pointer to the P matrix.
  void* __restrict__ p_ptr = nullptr;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr = nullptr;
  void* __restrict__ softmax_lseaccum_ptr = nullptr;

  // The dimensions.
  int b = 0;
  int seqlen_q = 0;
  int seqlen_k = 0;
  int seqlen_knew = 0;
  int d = 0;
  int seqlen_q_rounded = 0;
  int seqlen_k_rounded = 0;
  int d_rounded = 0;

  // The scaling factors for the kernel.
  float scale_softmax = 0.0;
  float scale_softmax_log2 = 0.0;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q = nullptr;
  int* __restrict__ cu_seqlens_k = nullptr;

  int* __restrict__ blockmask = nullptr;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr = nullptr;
  void* __restrict__ vnew_ptr = nullptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride = 0;
  index_t vnew_batch_stride = 0;
  index_t knew_row_stride = 0;
  index_t vnew_row_stride = 0;
  index_t knew_head_stride = 0;
  index_t vnew_head_stride = 0;

  bool is_bf16 = false;
  bool is_causal = false;

  // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
  // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
  bool is_seqlens_k_cumulative = true;
  int num_splits = 0;  // For split-KV version

  const cudaDeviceProp* dprops = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Headdim>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);
template <typename T, int Headdim>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace flash
}  // namespace onnxruntime
