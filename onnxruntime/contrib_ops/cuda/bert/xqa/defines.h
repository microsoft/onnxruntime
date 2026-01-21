/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "mha_stdheaders.cuh"

#define STATIC_NB_K_HEADS 0
#if STATIC_NB_K_HEADS
#define NB_K_HEADS 2
#endif

// allowed values are multiples of 16 in range [16, 256]
#ifndef HEAD_ELEMS
#define HEAD_ELEMS 128
#endif

// nbQHeads / nbKHeads for MQA/GQA
#ifndef HEAD_GRP_SIZE
#define HEAD_GRP_SIZE 8
#endif

#define IS_MLA (HEAD_GRP_SIZE == 128 && HEAD_ELEMS == 576)

#if IS_MLA
#define INPUT_ELEM __nv_fp8_e4m3
#define INPUT_ELEM2 __nv_fp8x2_e4m3
#define HEAD_ELEMS_V 512
#else
// 1 means fp16 and 0 means bf16 input/output
#ifndef INPUT_FP16
#define INPUT_FP16 1
#endif

// Don't modify
#if INPUT_FP16
#define INPUT_ELEM half
#define INPUT_ELEM2 half2
#else
#define INPUT_ELEM __nv_bfloat16
#define INPUT_ELEM2 __nv_bfloat162
#endif
#endif

// For beam search. Allowed values: 1, 4
#ifndef BEAM_WIDTH
#define BEAM_WIDTH 1
#endif

#ifndef SPEC_DEC
#define SPEC_DEC 0
#endif

// M_TILESIZE: Number of query tokens processed per CTA in M dimension.
// For decoding (S=1), this equals group_size. Allowed values: 8, 16, 32
#ifndef M_TILESIZE
#define M_TILESIZE 32
#endif

#if SPEC_DEC
using MaskType = uint32_t;
#endif

// Enables SWAP AB optimization for speculative decoding when using a small, fixed Q_SEQ_LEN.
// NOTE: Requires a uniform input sequence length for the entire batch.
#ifdef SPEC_Q_SEQ_LEN
static_assert(SPEC_DEC, "SPEC_Q_SEQ_LEN should only be used when SPEC_DEC is enabled.");
#endif

// 0: half/bf16 based on INPUT_FP16; 1: int8_t; 2: __nv_fp8_e4m3
#ifndef CACHE_ELEM_ENUM
#define CACHE_ELEM_ENUM 0
#endif

// don't modify
#define USE_KV_CACHE true

// don't modify
#ifndef ALLOW_MULTI_BLOCK_MODE
#define ALLOW_MULTI_BLOCK_MODE true
#endif

// For paged KV cache. Allowed values: 0, 16, 32, 64, 128
// 0 means contiguous KV cache (non-paged).
#ifndef TOKENS_PER_PAGE
#define TOKENS_PER_PAGE 0
#endif

// don't modify
#ifndef USE_PAGED_KV_CACHE
#define USE_PAGED_KV_CACHE (TOKENS_PER_PAGE > 0)
#endif

// Paged KV Cache Format
// 0 - XQA Original
// 1 - separate K and V cache pools, each with layout (batch, seq_len, head, head_elem) for VLLM/SGLang
#ifdef USE_PAGED_KV_CACHE
#ifndef PAGED_KV_CACHE_LAYOUT
#define PAGED_KV_CACHE_LAYOUT 0
#endif
#endif

// don't modify
#define USE_BEAM_SEARCH (BEAM_WIDTH > 1)

#if CACHE_ELEM_ENUM == 0
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll")
#else
#define PRAGMA_UNROLL_FP16_ONLY _Pragma("unroll(1)")
#endif

// good for short sequence length but bad for long sequence length. Only for mha.cu.
#ifndef SHORT_SEQ_OPT
#define SHORT_SEQ_OPT 1
#endif

#ifndef SLIDING_WINDOW
#define SLIDING_WINDOW 0
#endif

#ifndef SKIP_SOFTMAX_ATTN
#define SKIP_SOFTMAX_ATTN 0
#endif

#ifndef SKIP_SOFTMAX_ATTN_BLOCK_STATS
#define SKIP_SOFTMAX_ATTN_BLOCK_STATS 0
#endif

#ifndef SKIP_SOFTMAX_ATTN_FIX_THRESHOLD_GREATER_THAN_ONE
#define SKIP_SOFTMAX_ATTN_FIX_THRESHOLD_GREATER_THAN_ONE 1
#endif

// 0 - no PDL
// 1 - naive PDL
// 2 - aggressive PDL (implemented only in mha_sm90.cu for now)
#ifndef ENABLE_PDL
#define ENABLE_PDL 2
#endif

#ifndef USE_INPUT_KV
#define USE_INPUT_KV 0
#endif

#if USE_INPUT_KV
// 0 - no RoPE
// 1 - NEOX style
// 2 - GPTJ style
#ifndef ROPE_STYLE
#define ROPE_STYLE 0
#endif

#if SPEC_DEC
#error "SPEC_DEC is not supported for USE_INPUT_KV"
#endif
#endif

// Output element type:
//   0 - input element type
//   1 - KV cache element type
#ifndef LOW_PREC_OUTPUT
#define LOW_PREC_OUTPUT 0
#endif

#if LOW_PREC_OUTPUT
static_assert(CACHE_ELEM_ENUM != 0);
#endif

// true should be better if warpTile.x * cacheElemSize < 128. otherwise use false.
#define GRP_LOAD_V (CACHE_ELEM_ENUM != 0) || (HEAD_ELEMS == 256 && USE_PAGED_KV_CACHE && BEAM_WIDTH > 1)

// use custom barrier for NVRTC to avoid pulling in many headers
#ifndef USE_CUSTOM_BARRIER
#define USE_CUSTOM_BARRIER 1
#endif

#ifndef OPTIMIZE_FOR_LATENCY
#define OPTIMIZE_FOR_LATENCY 1
#endif

#ifndef IS_SPEC_DEC_TREE
#define IS_SPEC_DEC_TREE 1  // by default SPEC_DEC expect tree-based draft token structure
#endif

#define DBG_BATCH_SIZE 2
#define DBG_SEQ_LEN 256 * 4 + 3
#define DBG_NB_CTAS_PER_SEQ 8

#include <cuda_fp16.h>
#include <cuda_fp8.h>
template <int32_t elemTypeEnum>
using ElemType = mha::conditional_t<elemTypeEnum == 0, INPUT_ELEM,
                                    mha::conditional_t<elemTypeEnum == 1, int8_t, mha::conditional_t<elemTypeEnum == 2, __nv_fp8_e4m3, void>>>;

// we used an optimization where exp(x-rowMax) is computed as:
/*  bias = rowMax * log2e  // shared for the whole row
    exp(x-rowMax) = exp2f(x * log2e - bias)
*/
// But this optimization is not numerically stable when (x * log2e - bias) is computed with FMA and x is too large. For
// this reason, don't set safeInitRowMax with a huge absolute value.
#define SAFE_INIT_ROW_MAX (-1e+5F)
