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

#ifndef MHA_H_COMMON
#define MHA_H_COMMON
#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif
#include "defines.h"
#include "utils.h"
#if SPEC_DEC
#include "specDec.h"
#endif

using CacheElem = ElemType<CACHE_ELEM_ENUM>;
// These depend on HEAD_ELEMS which is Global
constexpr uint32_t validElemsPerHead = HEAD_ELEMS;
constexpr bool isMLA = IS_MLA;
static_assert((isMLA || validElemsPerHead <= 256) && (sizeof(CacheElem) * validElemsPerHead) % 16 == 0);
constexpr uint32_t headElems = validElemsPerHead <= 64 ? 64 : (validElemsPerHead <= 128 ? 128 : (isMLA ? 576 : 256));
static_assert(headElems == 64 || headElems == 128 || headElems == 256 || headElems == 576, "not implemented");
constexpr uint32_t beamWidth = BEAM_WIDTH;

#if SPEC_DEC
__device__ constexpr uint32_t rowsPerBlock = M_TILESIZE;
#endif

inline constexpr bool useSpecDec = SPEC_DEC;

using InputElem = INPUT_ELEM;
using InputElem2 = INPUT_ELEM2;
#if !(SPEC_DEC)
constexpr uint32_t inputSeqLen = 1;  // speculative decoding if > 1
#endif

constexpr bool useKVCache = USE_KV_CACHE;

using SeqLenDataType = uint32_t;
#endif

// Dependent definitions
#ifndef MHA_H_DEPENDENT
#define MHA_H_DEPENDENT
// This depends on HEAD_GRP_SIZE macro which changes per namespace
constexpr uint32_t headGrpSize = HEAD_GRP_SIZE;
#endif

// Common Part 2
#ifndef MHA_H_COMMON_2
#define MHA_H_COMMON_2

constexpr bool usePagedKVCache = USE_PAGED_KV_CACHE;
constexpr uint32_t tokensPerPage = TOKENS_PER_PAGE;

using IOHead = Vec<InputElem, validElemsPerHead>;
using InputHead = IOHead;
using GMemCacheHead = Vec<CacheElem, validElemsPerHead>;

constexpr uint32_t validElemsPerKHead = validElemsPerHead;
constexpr bool lowPrecOutput = LOW_PREC_OUTPUT;

#if IS_MLA
constexpr uint32_t validElemsPerVHead = 512;
static_assert(lowPrecOutput == false);
using OutputHead = Vec<__nv_bfloat16, validElemsPerVHead>;
#else
constexpr uint32_t validElemsPerVHead = validElemsPerHead;
using OutputHead = mha::conditional_t<lowPrecOutput, GMemCacheHead, InputHead>;
#endif
using OutputElem = OutputHead::Elem;

using PaddedInputHead = Vec<InputElem, headElems>;
using PaddedCacheHead = Vec<CacheElem, headElems>;

// impl detail, may be moved to mha.cu/mha_sm90.cu
constexpr bool isHeadPadded = (validElemsPerHead != headElems);

constexpr bool useInputKV = USE_INPUT_KV;

using GMemKVCacheHead = mha::conditional_t<useInputKV, GMemCacheHead, GMemCacheHead const>;

using KVCachePageIndex = int32_t;  // shape: KVCacheHead[nbKHeads][tokensPerPage]. Page index in the global pool of pages

constexpr bool allowSlidingWindow = SLIDING_WINDOW;

struct BeamSearchParams {
  uint32_t const* __restrict__ indices;  // shape: [batchSize][beamWidth][capacity]
  uint32_t capacity;
  uint32_t const* __restrict__ ctxLenList;  // shape: [batchSize][beamWidth]. Should be [batchSize] but we have to
                                            // match trt-llm API.
};

// function declarations removed to prevent ambiguity with namespaces
// they are defined in mha_impl.cuh included in xqa_loader.cu

#if STATIC_NB_K_HEADS
constexpr uint32_t nbKHeads = NB_K_HEADS;

constexpr uint32_t nbVHeads = nbKHeads;
constexpr uint32_t nbQHeads = nbKHeads * headGrpSize;
constexpr uint32_t nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;
#endif
constexpr uint32_t cacheElemSize = sizeof(CacheElem);
constexpr uint32_t inputElemSize = sizeof(InputElem);
constexpr uint32_t outputElemSize = sizeof(OutputElem);

constexpr uint32_t ioHeadBytes = sizeof(IOHead);
constexpr uint32_t gmemCacheHeadBytes = sizeof(GMemCacheHead);

constexpr uint32_t paddedInputHeadBytes = sizeof(PaddedInputHead);
constexpr uint32_t paddedCacheHeadBytes = sizeof(PaddedCacheHead);

constexpr uint32_t allowMultiBlockMode = ALLOW_MULTI_BLOCK_MODE;

enum class XQAKernelType : int32_t {
  kAMPERE_WARP_SPECIALIZED = 0,
  kHOPPER_WARP_SPECIALIZED = 1,
  kSM120_MLA = 2
};

#ifdef GENERATE_CUBIN
#define CUBIN_EXPORT extern "C"
#else
#define CUBIN_EXPORT static
#endif

#endif
