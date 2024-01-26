/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#if USE_TENSORRT_LLM_FMHA

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/kvCacheUtils.h"
#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/tmaDescriptor.h"
#include <limits.h>
#include <stdint.h>

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/multiHeadAttentionCommon.h"
//namespace onnxruntime {
namespace tensorrt_llm
{
namespace kernels
{

enum class ContextFMHAType
{
    DISABLED,
    ENABLED,
    // FP32 accumulation (FP16 I/O)
    ENABLED_WITH_FP32_ACC
};

enum class ContextAttentionMaskType
{
    PADDING,
    CAUSAL,
    SLIDING_WINDOW_CAUSAL
};

struct AlibiParams
{
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;

    AlibiParams(int h, float scale_after_alibi)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    float scale_after_alibi{};
    // Could be simplified to `int rank` derive the others as `num_heads * rank, s * rank` at
    // runtime, but this makes assumptions about the layout downstream
    // (e.g. downstream may only split across the head dimension, so s would be the full sequence)
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};

struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    const void* qkv_ptr;
    // The mask to implement drop-out.
    const void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use trick to avoid I2F/F2I in the INT8 kernel.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lengths
    const int* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    AlibiParams alibi_params{};

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int* o_scratch_ptr;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    int h_kv;

    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;

    // is input/output padded
    bool is_s_padded = false;

    // tma descriptors
    cudaTmaDesc tma_desc_q;
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        use_int8_scale_max = false;

        h_kv = 0;
        sliding_window_size = INT_MAX;
        is_s_padded = false;

        has_alibi = false;
        alibi_params = AlibiParams{};
    }
};

struct Fused_multihead_attention_paged_kv_params_v2
{
    // The Q matrices.
    const void* q_ptr;
    // Paged KV Cache buffer.
    KVBlockArray paged_kv_cache;
    // The O matrix (output).
    void* o_ptr;
    // The packed mask for random mask.
    const void* packed_mask_ptr;

    // The stride between rows of the Q matrices.
    int64_t q_stride_in_bytes;
    // The stride between rows of the paged KV matrices.
    int64_t kv_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // true: for int8, instead of doing max reduce, use max value encoded in scale factor
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    AlibiParams alibi_params;

    // array of length b+1 holding prefix sum of actual kv sequence lengths.
    const int* cu_seqlens;
    // Chunked attention (only handles one tile of Q).
    const int* cu_q_seqlens;

    // q with shape [B, S, H, D] in const cache.
    cudaTmaDesc tma_desc_q;
    // Tma descriptors for paged kv cache.
    // paged kv has [B, 2, Num_blocks] buffers,
    //  and each points to [Tokens_per_block, H, D] contiguous memory.
    cudaTmaDesc* tma_desc_paged_kv;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    int h_kv = 0;
    int h_q_per_kv = 0;

    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;

    // is input/output padded
    bool is_s_padded = false;

    void clear()
    {
        q_ptr = nullptr;
        o_ptr = nullptr;
        packed_mask_ptr = nullptr;

        q_stride_in_bytes = 0;
        kv_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        cu_q_seqlens = nullptr;
        use_int8_scale_max = false;

        h_kv = 0;
        h_q_per_kv = 0;
        sliding_window_size = INT_MAX;
        is_s_padded = false;

        has_alibi = false;
        alibi_params = AlibiParams{};
    }
};

// flags to control kernel choice
struct Launch_params
{
    // seq_length to select the kernel
    int kernel_s = 0;
    // kv_seq_length to set launch strategies.
    int kernel_kv_s = 0;
    // flags to control small batch kernel choice
    // true: never unroll
    bool ignore_b1opt = false;
    // true: always unroll
    bool force_unroll = false;
    // use fp32 accumulation
    bool force_fp32_acc = false;
    // the C/32 format
    bool interleaved = false;
    // by default TMA is not used.
    bool use_tma = false;
    // host seqlens to set tma descriptors
    int* seqlens = nullptr;
    // number of paged kv blocks for context sequence.
    int blocks_per_context_sequence = 0;
    // device ptrs on the host for paged kv cache.
    const int64_t* paged_kv_block_ptrs = nullptr;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool granular_tiling = false;
    // mask type: padding, causal, sliding_window_causal
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    // use specialized kernels without alibi support.
    bool useKernelWithoutAlibi = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;

    void set_default_kernel_selection_params()
    {
        kernel_s = 0;
        kernel_kv_s = 0;
        force_unroll = false;
        use_tma = false;
        flash_attention = false;
        warp_specialization = false;
        granular_tiling = false;
        attention_mask_type = (attention_mask_type == ContextAttentionMaskType::PADDING)
            ? ContextAttentionMaskType::PADDING
            : ContextAttentionMaskType::CAUSAL;
        useKernelWithoutAlibi = false;
    }
};

} // namespace kernels
} // namespace tensorrt_llm
//}
#endif
