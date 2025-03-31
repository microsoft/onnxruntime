// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if USE_LEAN_ATTENTION

#include "core/providers/cuda/cuda_common.h"
#include <tuple>

namespace onnxruntime {
namespace lean {

Status mha_fwd_kvcache(const cudaDeviceProp& dprops,
                       cudaStream_t stream,
                       void* q,            // batch_size x seqlen_q x num_heads x head_size
                       void* kcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x x head_size
                       void* vcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x x head_size
                       void* k,            // batch_size x seqlen_k_new x num_heads_k x head_size
                       void* v,            // batch_size x seqlen_k_new x num_heads_k x head_size
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
                       int num_splits = 0,
                       int grid_dimz = 0,
                       int max_tiles_per_tb = 0,
                       int high_load_tbs = 0,
                       int tiles_per_head = 0,
                       void* softmax_lse_accum = nullptr,  // num_splits x batch_size x seqlen_q x num_heads
                       void* out_accum = nullptr,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
                       int* sync_flag = nullptr,
                       int local_window_size = -1,
                       bool is_rotary_interleaved = false,
                       bool is_packed_qkv = false,
                       int max_num_blocks_per_seq = 0,
                       int page_block_size = 1);

size_t get_softmax_lse_size(size_t max_seqlen_q, size_t batch_size, size_t num_heads);

std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>
get_num_splits_and_buffer_sizes(size_t batch_size, size_t seqlen_q, size_t seqlen_k, size_t num_heads,
                                size_t num_heads_k, size_t head_size, size_t num_SMs, bool is_causal);

bool is_supported(const cudaDeviceProp& dprops, size_t head_size, size_t num_heads, size_t num_heads_k);

}  // namespace lean
}  // namespace onnxruntime

#endif  //  USE_LEAN_ATTENTION
