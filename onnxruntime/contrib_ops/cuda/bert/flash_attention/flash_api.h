/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#if USE_FLASH_ATTENTION

#include "core/providers/cuda/cuda_common.h"
#include <tuple>

namespace onnxruntime {
namespace flash {

Status mha_fwd(const cudaDeviceProp& dprops,
               cudaStream_t stream,
               void* q,            // batch_size x seqlen_q x num_heads x head_size
               void* k,            // batch_size x seqlen_k x num_heads_k x head_size
               void* v,            // batch_size x seqlen_k x num_heads_k x head_size
               void* out,          // batch_size x seqlen_q x num_heads x head_size
               void* softmax_lse,  // batch_size x num_heads x seqlen_q
               int batch_size,
               int num_heads,
               int num_heads_k,
               int head_size,
               int seqlen_q,
               int seqlen_k,
               float softmax_scale,
               bool is_causal,
               int num_splits = 0,
               void* softmax_lse_accum = nullptr,  // num_splits x batch_size x seqlen_q x num_heads
               void* out_accum = nullptr,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
               bool kv_bsnh = true);

Status mha_varlen_fwd(const cudaDeviceProp& dprops,
                      cudaStream_t stream,
                      void* q,            // half (total_q, num_heads, head_size)
                      void* k,            // half (total_k, num_heads, head_size)
                      void* v,            // half (total_k, num_heads, v_head_size)
                      void* out,          // half (total_q, num_heads, v_head_size)
                      int* cu_seqlens_q,  // int (batch_size + 1)
                      int* cu_seqlens_k,  // int (batch_size + 1)
                      void* softmax_lse,  // float (batch_size, num_heads, max_seqlen_q)
                      int batch_size,
                      int num_heads,
                      int num_heads_k,
                      int head_size,
                      int max_seqlen_q,
                      int max_seqlen_k,
                      float softmax_scale,
                      bool is_causal);

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
                       int batch_size,
                       int num_heads,
                       int num_heads_k,
                       int head_size,
                       int seqlen_q,
                       int seqlen_k,
                       int seqlen_k_new,
                       const float softmax_scale,
                       bool is_causal,
                       bool past_bsnh,  // otherwise bnsh
                       int num_splits = 0,
                       void* softmax_lse_accum = nullptr,  // num_splits x batch_size x seqlen_q x num_heads
                       void* out_accum = nullptr           // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
);

size_t get_softmax_lse_size(int max_seqlen_q, int batch_size, int num_heads);

std::tuple<int, int, int> get_num_splits_and_buffer_sizes(int batch_size, int seqlen_q, int seqlen_k, int num_heads,
                                                          int head_size, int num_SMs);

bool is_supported(const cudaDeviceProp& dprops, int head_size, int num_heads, int num_heads_k);

}  // namespace flash
}  // namespace onnxruntime

#endif  //  USE_FLASH_ATTENTION
