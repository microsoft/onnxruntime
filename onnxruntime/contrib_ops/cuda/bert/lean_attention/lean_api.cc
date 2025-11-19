/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Modifications: support lean attention.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_LEAN_ATTENTION

#include "contrib_ops/cuda/bert/lean_attention/lean_api.h"
#include <cutlass/numeric_types.h>

#include "contrib_ops/cuda/bert/lean_attention/flash.h"
#include "contrib_ops/cuda/bert/lean_attention/static_switch.h"

namespace onnxruntime {
namespace lean {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      size_t batch_size,
                      size_t seqlen_q,
                      size_t seqlen_k,
                      size_t seqlen_q_rounded,
                      size_t seqlen_k_rounded,
                      size_t num_heads,
                      size_t num_heads_k,
                      size_t head_size,
                      size_t head_size_rounded,
                      // device pointers
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* seqused_k,
                      void* p_d,
                      void* softmax_lse_d,
                      float softmax_scale,
                      bool is_causal,
                      bool is_bf16,
                      bool kv_bsnh = true,
                      int window_size_left = -1,
                      int window_size_right = -1) {
  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;

  params.is_bf16 = is_bf16;

  // All stride are in elements, not bytes.
  if (kv_bsnh) {
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = num_heads_k * head_size;
    params.v_row_stride = num_heads_k * head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = head_size;
    params.v_head_stride = head_size;
    params.o_row_stride = num_heads * head_size;
    params.o_head_stride = head_size;
  } else {
    params.q_row_stride = num_heads * head_size;
    params.k_row_stride = head_size;
    params.v_row_stride = head_size;
    params.q_head_stride = head_size;
    params.k_head_stride = seqlen_k * head_size;
    params.v_head_stride = seqlen_k * head_size;
    params.o_row_stride = num_heads * head_size;
    params.o_head_stride = head_size;
  }

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
    params.k_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.v_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.o_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
  } else {
    params.q_batch_stride = 0;
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.o_batch_stride = 0;
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int*>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)  // Ignore conversion from 'size_t' to 'int', possible loss of data
#pragma warning(disable : 4244)  // Ignore conversion from 'double' to 'float', possible loss of data
#endif
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // In our API, causal/unidirectional determines if we only look at prior tokens. However, the flash API separates
  // local and causal, meaning when we have local window size
  params.is_causal = is_causal;
  if (is_causal && (window_size_left >= 0 || window_size_right != 0)) {
    params.is_causal = false;
  }
  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.is_seqlens_k_cumulative = true;
}

size_t get_softmax_lse_size(size_t seqlen, size_t batch_size, size_t num_heads) {
  size_t bytes = sizeof(float) * batch_size * num_heads * seqlen;
  return bytes;
}

size_t get_softmax_lse_accum_size(size_t num_splits, size_t batch_size, size_t num_heads, size_t seqlen_q) {
  size_t bytes = sizeof(float) * num_splits * batch_size * seqlen_q * num_heads;
  return bytes;
}

size_t get_out_accum_size(size_t num_splits, size_t batch_size, size_t num_heads,
                          size_t seqlen_q, size_t head_size_rounded) {
  size_t bytes = sizeof(float) * num_splits * batch_size * seqlen_q * num_heads * head_size_rounded;
  return bytes;
}

size_t get_sync_flag_size(size_t num_m_blocks, size_t batch_size, size_t num_heads) {
  size_t bytes = sizeof(int) * batch_size * num_heads * num_m_blocks;
  return bytes;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      run_mha_fwd_lean_dispatch<elem_type, kHeadDim>(params, stream);
    });
  });
}

std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t> get_num_splits_and_buffer_sizes(size_t batch_size, size_t max_seqlen_q, size_t max_seqlen_k,
                                                                                                           size_t num_heads, size_t num_heads_k, size_t head_size, size_t num_SMs, bool is_causal) {
  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int block_m = head_size <= 64 ? 64 : (head_size <= 128 ? 64 : 64);
  const int num_m_blocks = (max_seqlen_q + block_m - 1) / block_m;
  const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
  if (max_seqlen_q == 1) {
    is_causal = false;
  }

  max_seqlen_q = max_seqlen_q * num_heads / num_heads_k;

#if defined(DEBUG_LEAN_ATTENTION)
  printf("block_n: %d\n", block_n);
  printf("block_m: %d\n", block_m);
  printf("num_m_blocks: %d\n", num_m_blocks);
  printf("num_n_blocks: %d\n", num_n_blocks);
  printf("max_seqlen_q: %lu\n", max_seqlen_q);
  printf("max_seqlen_k: %lu\n", max_seqlen_k);
  printf("is_causal: %d\n", is_causal);
  printf("num_heads: %lu\n", num_heads);
  printf("num_heads_k: %lu\n", num_heads_k);
#endif

  size_t tiles_per_head = 0;
  if (is_causal) {
    // Prefill - Causal
    for (int i = 0; i < num_m_blocks; i++) {
      tiles_per_head += (((i + 1) * block_m) + block_n - 1) / block_n;
    }
  } else {
    // Decode or Not Causal
    // Tiles per head is the number of blocks in the first block
    tiles_per_head = num_m_blocks * num_n_blocks;
  }
  size_t total_tiles = tiles_per_head * batch_size * num_heads_k;

  // StreamK Lean has as many threadblocks as SMs
  // This should be a function of tile size and number of scratchpad space

  // We want at least two tiles per CTA to be efficient
  // And then 2 CTAs per SM
  size_t lean_griddimz = num_SMs * 2;
  if (total_tiles <= 2 * 2 * num_SMs) {
    lean_griddimz = std::min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks);
    // params.lean_griddimz = num_m_blocks * batch_size * num_heads;
  } else {
    // Max split of 64 per block is allowed, so we conservatively set it to 32
    // to account for ceil
    lean_griddimz = std::min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks);
  }
  size_t max_tiles_per_tb = (total_tiles + lean_griddimz - 1) / lean_griddimz;
  // Find max number of splits
  size_t num_splits = 0;
  if (total_tiles % lean_griddimz == 0) {
    num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) / (max_tiles_per_tb));
  } else {
    num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 3) / (max_tiles_per_tb - 1));
  }
  size_t high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz);

#if defined(DEBUG_LEAN_ATTENTION)
  printf("Causal: %d params.tiles_per_head : %lu\n", is_causal, tiles_per_head);
  printf("num_splits = %lu\n", num_splits);
  printf("total_tiles = %lu\n", total_tiles);
  printf("lean_griddimz = %lu\n", lean_griddimz);
  printf("max_tiles_per_tb = %lu\n", max_tiles_per_tb);
  printf("high_load_tbs = %lu\n", high_load_tbs);
#endif

  if (num_splits > 1) {
    size_t softmax_lse_accum_bytes = get_softmax_lse_accum_size(num_splits, batch_size, num_heads_k, max_seqlen_q);
    auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
    const size_t head_size_rounded = round_multiple(head_size, 32);
    size_t out_accum_bytes = get_out_accum_size(num_splits, batch_size, num_heads_k, max_seqlen_q, head_size_rounded);
    size_t sync_flag_bytes = get_sync_flag_size(num_m_blocks, batch_size, num_heads_k);
    return {num_splits, softmax_lse_accum_bytes, out_accum_bytes, sync_flag_bytes, lean_griddimz, max_tiles_per_tb, high_load_tbs, tiles_per_head};
  } else {
    return {0, 0, 0, 0, lean_griddimz, max_tiles_per_tb, high_load_tbs, tiles_per_head};
  }
}

bool is_supported(const cudaDeviceProp& dprops, size_t head_size, size_t num_heads, size_t num_heads_k) {
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  return (is_sm8x || is_sm90) && (head_size == 64 || head_size == 128) && (num_heads % num_heads_k == 0);
}

// This API is used when past key and value are present... since cached, these are assumed to have sequence length
// of max_sequence_length, so seqlen_k == max_sequence_length. The actual past sequence length is held in seqlens_k_.
Status mha_fwd_kvcache(const cudaDeviceProp& dprops,
                       cudaStream_t stream,
                       void* q,            // batch_size x seqlen_q x num_heads x head_size
                       void* kcache,       // batch_size x seqlen_k_max x num_heads_k x head_size or batch_size x num_heads_k x seqlen_k_max x head_size
                       void* vcache,       // batch_size x seqlen_k_max x num_heads_k x head_size or batch_size x num_heads_k x seqlen_k_max x head_size
                       void* k_new,        // (optional) batch_size x seqlen_k_new x num_heads_k x head_size
                       void* v_new,        // (optional) batch_size x seqlen_k_new x num_heads_k x head_size
                       void* out,          // batch_size x seqlen_q x num_heads x head_size
                       void* softmax_lse,  // batch_size x num_heads x seqlen_q
                       void* seqlens_k_,   // batch_size
                       void* rotary_cos,   // seqlen_ro x (rotary_dim / 2)
                       void* rotary_sin,   // seqlen_ro x (rotary_dim / 2)
                       int* block_table,   // batch_size x max_num_blocks_per_seq
                       int batch_size,
                       int num_heads,
                       int num_heads_k,
                       int head_size,
                       int seqlen_q,
                       int seqlen_k,
                       int seqlen_k_new,
                       int rotary_dim,
                       const float softmax_scale,
                       bool is_causal,
                       bool is_bf16,
                       bool past_bsnh,  // otherwise bnsh
                       int num_splits,
                       int grid_dimz,
                       int max_tiles_per_tb,
                       int high_load_tbs,
                       int tiles_per_head,
                       void* softmax_lse_accum,  // num_splits x batch_size x seqlen_q x num_heads
                       void* out_accum,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
                       int* sync_flag,
                       int local_window_size,
                       bool is_rotary_interleaved,
                       bool is_packed_qkv,
                       int max_num_blocks_per_seq,
                       int page_block_size) {
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
  const bool paged_KV = block_table != nullptr;

#if defined(DEBUG_LEAN_ATTENTION)
  printf(
      "batch_size: %d num_heads %d num_heads_k %d head_size %d seqlen_q %d seqlen_k %d seqlen_k_new %d "
      "softmax_scale %f is_causal %d is_bf16 %d past_bsnh %d num_splits %d grid_dimz %d max_tiles_per_tb %d "
      "high_load_tbs %d tiles_per_head %d local_window_size %d is_rotary_interleaved %d is_packed_qkv %d "
      "max_num_blocks_per_seq %d page_block_size %d\n",
      batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k, seqlen_k_new,
      softmax_scale, is_causal, is_bf16, past_bsnh, num_splits, grid_dimz, max_tiles_per_tb,
      high_load_tbs, tiles_per_head, local_window_size, is_rotary_interleaved, is_packed_qkv,
      max_num_blocks_per_seq, page_block_size);
#endif

  // Lean attention treats decode as non-causal
  if (seqlen_q == 1) {
    is_causal = false;
  }

  const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0;
  if (seqlenq_ngroups_swapped) {
    const int ngroups = num_heads / num_heads_k;
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  // In kv-cache case, seqlen_k_max as kv sequence length
  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size,
                   seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k,
                   head_size, head_size_rounded,
                   q, kcache, vcache, out,
                   /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr,
                   /*seqused_k=*/nullptr,
                   /*p_ptr=*/nullptr,
                   softmax_lse,
                   softmax_scale,
                   is_causal,
                   is_bf16,
                   past_bsnh,
                   local_window_size,
                   is_causal ? 0 : -1);
  params.dprops = &dprops;

  if (k_new != nullptr && v_new != nullptr) {
    params.seqlen_knew = seqlen_k_new;
    params.knew_ptr = k_new;
    params.vnew_ptr = v_new;
    // All stride are in elements, not bytes.
    params.q_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
    params.k_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.v_batch_stride = seqlen_k * num_heads_k * head_size;  // stride(0)
    params.o_batch_stride = seqlen_q * num_heads * head_size;    // stride(0)
    if (is_packed_qkv) {
      params.q_batch_stride = (seqlen_q * num_heads * head_size) + (2 * seqlen_k_new * num_heads_k * head_size);
      params.q_row_stride = (num_heads * head_size) + (2 * num_heads_k * head_size);
      params.knew_batch_stride = (seqlen_q * num_heads * head_size) + (2 * seqlen_k_new * num_heads_k * head_size);
      params.vnew_batch_stride = (seqlen_q * num_heads * head_size) + (2 * seqlen_k_new * num_heads_k * head_size);
      params.knew_row_stride = (num_heads * head_size) + (2 * num_heads_k * head_size);
      params.vnew_row_stride = (num_heads * head_size) + (2 * num_heads_k * head_size);
    } else {
      params.knew_batch_stride = seqlen_k_new * num_heads_k * head_size;
      params.vnew_batch_stride = seqlen_k_new * num_heads_k * head_size;
      params.knew_row_stride = num_heads_k * head_size;
      params.vnew_row_stride = num_heads_k * head_size;
    }
    params.knew_head_stride = head_size;
    params.vnew_head_stride = head_size;
  } else {
    params.seqlen_knew = 0;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;
  }

  if (seqlenq_ngroups_swapped) {
    if (is_packed_qkv) {
      params.q_batch_stride = (seqlen_q * num_heads_k * head_size) + (2 * seqlen_k_new * num_heads_k * head_size);
    } else {
      params.q_batch_stride = seqlen_q * num_heads_k * head_size;
    }
    params.q_row_stride = head_size;
    params.q_head_stride = seqlen_q * head_size;
    params.o_row_stride = head_size;
    params.o_head_stride = seqlen_q * head_size;
    params.o_batch_stride = seqlen_q * num_heads_k * head_size;
  }

  params.is_seqlens_k_cumulative = seqlens_k_ == nullptr;
  if (seqlens_k_ != nullptr) {
    params.cu_seqlens_k = static_cast<int*>(seqlens_k_);
  }

  if (rotary_cos != nullptr) {
    params.rotary_cos_ptr = rotary_cos;
    params.rotary_sin_ptr = rotary_sin;
    params.is_rotary_interleaved = is_rotary_interleaved;
    params.rotary_dim = rotary_dim;
  }

  params.num_splits = num_splits;
  params.lean_griddimz = grid_dimz;
  params.max_tiles_per_tb = max_tiles_per_tb;
  params.high_load_tbs = high_load_tbs;
  params.tiles_per_head = tiles_per_head;
  if (params.num_splits > 1 && softmax_lse_accum != nullptr && out_accum != nullptr) {
    params.softmax_lseaccum_ptr = softmax_lse_accum;
    params.oaccum_ptr = out_accum;
    params.sync_flag = sync_flag;
  } else {
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
  }

  params.alibi_slopes_ptr = nullptr;
  if (paged_KV) {
    params.block_table = block_table;  // TODO(aciddelgado): cast to int pointer
    params.block_table_batch_stride = max_num_blocks_per_seq;
    // params.num_blocks = num_blocks;
    params.page_block_size = page_block_size;
    params.k_batch_stride = page_block_size * num_heads_k * head_size;
    params.v_batch_stride = page_block_size * num_heads_k * head_size;
  } else {
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    // params.num_blocks = 0;
    params.page_block_size = 1;
  }

  // Only split kernel supports appending to KV cache
  run_mha_fwd(params, stream);

  return Status::OK();
}

}  // namespace lean
}  // namespace onnxruntime

#endif  // USE_LEAN_ATTENTION
