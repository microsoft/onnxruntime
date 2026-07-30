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
#include "ldgsts.cuh"
#include "mha.h"
#include "utils.cuh"

// for beam search
template <typename Head, uint32_t tokensPerPage, uint32_t nbPages>
struct IndexedHeadPtrImpl {
  static_assert(tokensPerPage != 0 && nbPages != 0);
  const uint32_t* indices;  // values are in range [0, beamWidth)
  Head* pool;
  const Vec<KVCachePageIndex, nbPages>* pageIndices;
  uint32_t nbKHeads;
  uint32_t offset;  // applied onto pool + pointers

  __device__ inline Head& operator[](uint32_t i) const {
    return *(*this + i);
  }

  __device__ inline Head* operator+(uint32_t i) const {
    assert(indices[i] < beamWidth);
    assert(nbPages == 1 || offset % tokensPerPage == 0);
    const auto pageIdx = pageIndices[indices[i]][nbPages == 1 ? 0U : i / tokensPerPage];
    return pool + (tokensPerPage * nbKHeads * pageIdx + offset + i % tokensPerPage);
  }
};

template <typename Head>
struct IndexedHeadPtrImpl<Head, 0, 0> {
  const uint32_t* indices;  // values are in range [0, beamWidth)
  Head* pointer;
  uint32_t offset;
  uint32_t beamStride;

  __device__ inline Head& operator[](uint32_t i) const {
    return *(*this + i);
  }

  __device__ inline Head* operator+(uint32_t i) const {
    assert(indices[i] < beamWidth);
    return pointer + (beamStride * indices[i] + offset + i);
  }
};

template <typename Head, uint32_t tokensPerPage, uint32_t nbPages = 0>
using IndexedHeadPtr = IndexedHeadPtrImpl<Head, tokensPerPage, nbPages>;

// for beamWidth = 1
template <typename Head, uint32_t tokensPerPage, uint32_t nbPages>
struct HeadPtr {
  static_assert(tokensPerPage != 0 && nbPages != 0);
  Head* pool;
  Vec<KVCachePageIndex, nbPages> pageIndices;
  uint32_t nbKHeads;
  uint32_t offset;  // offset inside the first page.

  __device__ inline Head& operator[](uint32_t i) const {
    return *(*this + i);
  }

  __device__ inline Head* operator+(uint32_t i) const {
#if PAGED_KV_CACHE_LAYOUT == 1 && USE_PAGED_KV_CACHE
    const auto pageIdx = pageIndices[nbPages == 1 ? 0U : i / tokensPerPage];
    return (pageIdx & (1U << 31))
               ? nullptr
               : pool + (tokensPerPage * nbKHeads * pageIdx + offset + (i % tokensPerPage) * nbKHeads);
#else
    assert(nbPages == 1 || offset % tokensPerPage == 0);
    const auto pageIdx = pageIndices[nbPages == 1 ? 0U : i / tokensPerPage];
    return (pageIdx & (1U << 31)) ? nullptr
                                  : pool + (tokensPerPage * nbKHeads * pageIdx + offset + i % tokensPerPage);
#endif
  }
};

template <typename Head>
struct HeadPtr<Head, 0, 0> : TinyPtr<Head> {
};

// template <typename Head>
// #if BEAM_WIDTH == 1
// using SrcHeadPtr = TinyPtr<Head const>;
// #else
// using SrcHeadPtr = IndexedHeadPtr<Head>;
// #endif

// @fixme: give evict first hint for last part.
template <typename Head, uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle, bool isFull,
          uint32_t dstNbHeads, typename SrcHeadPtr, typename LocalHeadIdxMap = uint32_t (*)(uint32_t)>
__device__ inline void copyPartialHeadsAsync(
    const Warp& warp, Array2D<LdGrain, dstNbHeads, exactDiv(exactDiv(sizeof(Head), nbPartsPerHead), grainBytes)>& dst,
    uint32_t dstHeadOffset, const SrcHeadPtr& src, uint32_t idxPart, uint32_t nbAvailHeads = maxNbCopiedHeads,
    LocalHeadIdxMap&& localHeadIdxMap = [](uint32_t x) { return x; }) {
  static_assert(maxNbCopiedHeads <= dstNbHeads);
  assert(idxPart < nbPartsPerHead);
  assert(dstHeadOffset + maxNbCopiedHeads <= dstNbHeads);
  assert(sizeof(Head) * (src.offset + maxNbCopiedHeads) <= (1ULL << 32));
  assert(!isFull || nbAvailHeads >= maxNbCopiedHeads);
  constexpr uint32_t headBytes = sizeof(Head);
  constexpr uint32_t partBytes = exactDiv(headBytes, nbPartsPerHead);
  constexpr uint32_t warpLdBytes = partBytes * maxNbCopiedHeads;
  constexpr uint32_t thrdLdBytes = exactDiv(warpLdBytes, warp_size);
  assertIsPowerOf2<thrdLdBytes>();
  static_assert(thrdLdBytes >= grainBytes);
  // a segment is responsible for loading one partial head collaboratively
  constexpr uint32_t thrdsPerSeg = exactDiv(partBytes, grainBytes);
  static_assert(thrdsPerSeg > 0 && thrdsPerSeg <= warp_size);
  assertIsPowerOf2<thrdsPerSeg>();
  assert(__shfl_sync(0xFU << (laneId() / 4 * 4), src.offset, 0, 4) == src.offset);
  const auto warpLane = laneId();
  const uint32_t segIdx = warpLane / thrdsPerSeg;
  const uint32_t segLane = warpLane % thrdsPerSeg;
  constexpr uint32_t partsPerWarpInst = exactDiv(grainBytes * warp_size, partBytes);
#pragma unroll
  for (uint32_t i = 0; i < thrdLdBytes / grainBytes; i++) {
    const uint32_t idxHeadLocal = partsPerWarpInst * i + segIdx;
    assert(idxHeadLocal < maxNbCopiedHeads);
    const bool isHeadInBound = isFull || (idxHeadLocal < nbAvailHeads);
    constexpr uint32_t grainsPerPart = exactDiv(partBytes, grainBytes);
    using SrcHead = mha::decay_t<decltype(src[0])>;
    constexpr uint32_t nbValidGrains = exactDiv(sizeof(SrcHead), grainBytes);
    const uint32_t idxGrainInsideHead = grainsPerPart * idxPart + segLane;
    const bool isGrainInBound = (!isHeadPadded || idxGrainInsideHead < nbValidGrains);
    const SrcHead* const pSrcHead = src + localHeadIdxMap(idxHeadLocal);
    const bool isValidPage = (pSrcHead != nullptr);
    const LdGrain* const pSrc = reinterpret_cast<const LdGrain*>(pSrcHead) + idxGrainInsideHead;
    LdGrain* const pDst = &dst.template at<swizzle>(dstHeadOffset + idxHeadLocal, segLane);
    assert(!hasBankConflict(pDst));
    ldgsts::copyAsync<grainBytes>(pDst, pSrc, isValidPage && isHeadInBound && isGrainInBound ? grainBytes : 0u);
  }
}

template <typename Head, uint32_t maxNbCopiedHeads, uint32_t nbWarps, bool swizzle, bool isFull, uint32_t dstNbHeads,
          typename SrcHeadPtr, typename LocalHeadIdxMap = uint32_t (*)(uint32_t)>
__device__ inline void copyHeadsAsync(
    uint32_t idxWarp, Array2D<LdGrain, dstNbHeads, exactDiv(sizeof(Head), grainBytes)>& dst, const SrcHeadPtr& src,
    uint32_t nbAvailHeads = maxNbCopiedHeads, LocalHeadIdxMap&& localHeadIdxMap = [](uint32_t x) { return x; }) {
  assert(idxWarp < nbWarps);
  const Warp& warp = this_warp();
  constexpr uint32_t maxNbHeadsPerWarp = exactDiv(maxNbCopiedHeads, nbWarps);
  const uint32_t dstHeadOffset = maxNbHeadsPerWarp * idxWarp;
  const uint32_t warpNbAvailHeads = (dstHeadOffset < nbAvailHeads ? nbAvailHeads - dstHeadOffset : 0);
  constexpr uint32_t idxPart = 0;
  copyPartialHeadsAsync<Head, maxNbHeadsPerWarp, 1, swizzle, isFull, dstNbHeads>(warp, dst, dstHeadOffset, src,
                                                                                 idxPart, warpNbAvailHeads, [&](uint32_t x) { return localHeadIdxMap(dstHeadOffset + x); });
}

template <bool isAsync, uint32_t maxTotalNbGrains, uint32_t nbWarps, bool isFull = true>
__device__ inline void copyGrains(
    uint32_t idxWarp, LdGrain* dst, const LdGrain* src, uint32_t totalNbGrains = maxTotalNbGrains) {
  assert((isFull && totalNbGrains == maxTotalNbGrains) || (!isFull && totalNbGrains <= maxTotalNbGrains));
  constexpr uint32_t nbThrds = warp_size * nbWarps;
  const uint32_t tid = warp_size * idxWarp + laneId();
// copy output to scratch
#pragma unroll
  for (uint32_t i = 0; i < divUp(maxTotalNbGrains, nbThrds); i++) {
    const uint32_t idx = nbThrds * i + tid;
    if (!(isFull && maxTotalNbGrains % nbThrds == 0) && idx >= totalNbGrains) {
      break;
    }
    if constexpr (isAsync) {
      ldgsts::copyAsync<grainBytes>(&dst[idx], &src[idx], grainBytes);
    } else {
      dst[idx] = src[idx];
    }
  }
}

// with ldmatrix, what we load for fp8 cache is T0:{e0,e1,e2,e3}; T1:{e4, e5, e6, e7}; T2:{e8,e9,e10,e11}; T3:{e12, e13,
// e14, e15}; When casted to fp16, it will be T0:{e0, e1}; T1{e4, e5};...  | T0:{e2, e3}; T1{e6, e7}; ... We need to
// reorder Q to match that order. isFwd=false to revert the reorder.
template <uint32_t nbWarps, bool swizzled, bool isFwd, uint32_t cols, uint32_t rows>
__device__ inline void reorder16bQHeadsToMatch8bKCache(uint32_t idxWarp, Array2D<LdGrain, rows, cols>& qHeads) {
  assert(idxWarp < nbWarps);
  constexpr uint32_t nbWarpIters = exactDiv(exactDiv(cols, 2) * rows, warp_size);  // warps * iters
  constexpr uint32_t nbWorkingWarps = mha::min(nbWarps, nbWarpIters);
  if (idxWarp >= nbWorkingWarps) {
    return;
  }
  static_assert(cols % 2 == 0);
  const uint32_t tid = warp_size * idxWarp + laneId();
  constexpr uint32_t iterCols = exactDiv(warp_size * nbWorkingWarps, rows) * 2;
  static_assert(cols % iterCols == 0, "fix this by reducing nbWorkingWarps, or use divUp and add runtime check");
  constexpr uint32_t nbIters = exactDiv(cols, iterCols);
  static_assert(nbIters == exactDiv(nbWarpIters, nbWorkingWarps));
  const uint32_t r = tid % rows;
  const uint32_t cInit = tid / rows * 2;
#pragma unroll
  for (uint32_t n = 0; n < nbIters; n++) {
    const uint32_t c = cInit + iterCols * n;
    const LdGrain src[2] = {
        qHeads.template at<swizzled>(r, c),
        qHeads.template at<swizzled>(r, c + 1),
    };
    const auto& s = reinterpret_cast<const Vec<uint32_t, LdGrain::size * 2>&>(src);
    if constexpr (isFwd) {
      qHeads.template at<swizzled>(r, c) = LdGrain{s[0], s[2], s[4], s[6]};
      qHeads.template at<swizzled>(r, c + 1) = LdGrain{s[1], s[3], s[5], s[7]};
    } else {
      qHeads.template at<swizzled>(r, c) = LdGrain{s[0], s[4], s[1], s[5]};
      qHeads.template at<swizzled>(r, c + 1) = LdGrain{s[2], s[6], s[3], s[7]};
    }
  }
}

template <bool usePagedKVCache>
struct KVCacheList;

template <>
struct KVCacheList<true> {
#if PAGED_KV_CACHE_LAYOUT == 1
  GMemCacheHead* kCacheVLLM;
  GMemCacheHead* vCacheVLLM;
#else
  GMemKVCacheHead* pool;
#endif
  const KVCachePageIndex* kvCachePageList;  // shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
  const SeqLenDataType* seqLenList;         // shape: [batchSize][beamWidth] (for compatibility)
  uint32_t maxNbPagesPerSeq;
  // Added to the value read from seqLenList. ORT callers pass the *past* sequence length (the
  // length before the token being decoded) plus extraSeqLen == 1, matching the contiguous
  // KVCacheList<false> specialization below and keeping getCacheSeqLen() uniform over both.
  uint32_t extraSeqLen;
};

template <>
struct KVCacheList<false> {
  GMemKVCacheHead* kData;            // shape: KVCacheHead[batchSize][beamWidth][nbKHeads][capacity]
  GMemKVCacheHead* vData;            // shape: KVCacheHead[batchSize][beamWidth][nbKHeads][capacity]
  const SeqLenDataType* seqLenList;  // shape: [batchSize][beamWidth] (for compatibility)
  uint32_t capacity;
  bool isBSNH;
  uint32_t extraSeqLen;
};

__device__ inline uint32_t getSeqLen(const uint32_t* seqLenList, uint32_t idxReq, uint32_t extraSeqLen) {
  uint64_t cachePolicy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0;\n" : "=l"(cachePolicy));
  uint32_t len;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;\n"
      : "=r"(len)
      : "l"(&seqLenList[idxReq * beamWidth]), "l"(cachePolicy));
  for (uint32_t i = 0; i < beamWidth; i++) {
    assert(len == seqLenList[idxReq * beamWidth + i]);
  }
  return len + extraSeqLen;
}

template <bool isPaged>
__device__ inline uint32_t getCacheSeqLen(const KVCacheList<isPaged>& cacheList, uint32_t idxReq) {
  return getSeqLen(cacheList.seqLenList, idxReq, cacheList.extraSeqLen);
}

__device__ inline uint32_t getCtxCacheSeqLen(const BeamSearchParams& beamSearchParams, uint32_t idxReq) {
  return getSeqLen(beamSearchParams.ctxLenList, idxReq, 0);
}

template <uint32_t nbLoadedPages>
__device__ inline Vec<KVCachePageIndex, nbLoadedPages> getPage(const KVCacheList<true>& cacheList, bool isK,
                                                               uint32_t idxReq, uint32_t idxBeam, uint32_t idxPageBeg, uint32_t nbPages) {
  const auto maxNbPagesPerSeq = cacheList.maxNbPagesPerSeq;
  Vec<KVCachePageIndex, nbLoadedPages> ret;
#pragma unroll
  for (uint32_t i = 0; i < nbLoadedPages; i++) {
    const uint32_t idxPage = idxPageBeg + i;
#if PAGED_KV_CACHE_LAYOUT == 1 && USE_PAGED_KV_CACHE
    ret[i] = (idxPage < nbPages ? cacheList.kvCachePageList[maxNbPagesPerSeq * idxReq + idxPage] : kBAD_PAGE_INDEX);
#else
    ret[i] = (idxPage < nbPages ? cacheList.kvCachePageList[beamWidth * 2 * maxNbPagesPerSeq * idxReq + 2 * maxNbPagesPerSeq * idxBeam + maxNbPagesPerSeq * (isK ? 0U : 1U) + idxPage]
                                : kBAD_PAGE_INDEX);
#endif
  }
  return ret;
}

template <uint32_t nbWarps, uint32_t nbLoadedPages>
__device__ inline void loadPagesForBeamSearchAsync(uint32_t idxWarp,
                                                   const Vec<Vec<KVCachePageIndex, nbLoadedPages>, beamWidth>& dst, KVCacheList<true>& cacheList, bool isK,
                                                   uint32_t idxReq, uint32_t idxPageBeg, uint32_t nbPages) {
  assert(idxWarp < nbWarps);
  const auto maxNbPagesPerSeq = cacheList.maxNbPagesPerSeq;
  static_assert(beamWidth < warp_size);
  const auto tid = warp_size * idxWarp + laneId();
  const auto idxBeam = tid / nbLoadedPages;
  const auto idxLoadedPage = tid % nbLoadedPages;
  static_assert(warp_size * nbWarps >= beamWidth * nbLoadedPages);
  if (idxBeam < beamWidth) {
    constexpr uint32_t nbBytes = sizeof(KVCachePageIndex);
    const uint32_t idxPage = idxPageBeg + idxLoadedPage;
    ldgsts::copyAsync<nbBytes>(&dst[idxBeam][idxLoadedPage],
                               &cacheList.kvCachePageList[beamWidth * 2 * maxNbPagesPerSeq * idxReq + 2 * maxNbPagesPerSeq * idxBeam + (isK ? 0U : maxNbPagesPerSeq) + idxPage],
                               idxPage < nbPages ? nbBytes : 0U);
  }
}

template <uint32_t nbWarps, uint32_t length, bool isFullTile = false>
__device__ inline void loadIndicesForBeamSearchAsync(uint32_t idxWarp, Vec<uint32_t, length>& dst,
                                                     const BeamSearchParams& params, uint32_t idxReq, uint32_t idxBeam, uint32_t uniformSeqOffset, uint32_t seqLen) {
  constexpr uint32_t nbThreads = warp_size * nbWarps;
  // constexpr uint32_t indicesPerInst = mha::min(exactDiv(grainBytes, sizeof(uint32_t)), divUp(length, nbThreads));
  // // @fixme: std::bit_ceil on length
  constexpr uint32_t indicesPerInst = 1U;  // to handle unaligned case.
  constexpr uint32_t bytesPerInst = sizeof(uint32_t) * indicesPerInst;
  assertIsPowerOf2<indicesPerInst>();
  const uint32_t capacity = params.capacity;
  const uint32_t srcOffset = (idxReq * beamWidth + idxBeam) * capacity + uniformSeqOffset;
  const uint32_t tid = warp_size * idxWarp + laneId();
  constexpr uint32_t indicesPerIter = indicesPerInst * nbThreads;
#pragma unroll
  for (uint32_t i = 0; i < length / indicesPerIter; i++) {
    const uint32_t idx = indicesPerIter * i + indicesPerInst * tid;
    ldgsts::copyAsync<bytesPerInst>(&dst[idx], &params.indices[srcOffset + idx],
                                    (isFullTile || uniformSeqOffset + idx < seqLen) ? bytesPerInst : 0);
  }
  if constexpr (length % indicesPerIter != 0) {
    const uint32_t idx = indicesPerIter * (length / indicesPerIter) + indicesPerInst * tid;
    if (idx < length) {
      ldgsts::copyAsync<bytesPerInst>(&dst[idx], &params.indices[srcOffset + idx],
                                      (isFullTile || uniformSeqOffset + idx < seqLen) ? bytesPerInst : 0);
    }
  }
}

__device__ inline InputElem2 float2ToInputElem2(float2 src) {
  InputElem2 dst;
  if constexpr (mha::is_same_v<InputElem2, half2>) {
    reinterpret_cast<half2&>(dst) = __float22half2_rn(src);
    return dst;
  } else if constexpr (mha::is_same_v<InputElem2, nv_bfloat162>) {
    reinterpret_cast<nv_bfloat162&>(dst) = __float22bfloat162_rn(src);
    return dst;
  } else if constexpr (mha::is_same_v<InputElem2, __nv_fp8x2_e4m3>) {
    reinterpret_cast<__nv_fp8x2_e4m3&>(dst) = __nv_fp8x2_e4m3{src};
    return dst;
  } else {
    trap();
  }
}

template <bool real>
using TokenOrNone = RealTypeOrNone<real, CtaBarrier::arrival_token>;

template <bool real>
__device__ inline TokenOrNone<real> arrive(CtaBarrier* pBarrier) {
  if constexpr (real) {
    return pBarrier->arrive();
  } else {
    assert(pBarrier == nullptr);
    return None{};
  }
}

template <bool real>
__device__ inline void wait(CtaBarrier* pBarrier, TokenOrNone<real>&& token) {
  if constexpr (real) {
    pBarrier->wait(mha::move(token));
  } else {
    assert(pBarrier == nullptr);
    __syncwarp();
  }
}

template <bool real>
__device__ inline bool test_wait(CtaBarrier* pBarrier, TokenOrNone<real>&& token) {
  if constexpr (real) {
    uint32_t complete;
    asm volatile(
        "{\n"
        ".reg .pred       complete;\n"
        "mbarrier.test_wait.acquire.cta.shared::cta.b64 complete, [%1], %2;\n"
        "selp.b32 %0, 1, 0, complete;\n}\n"
        : "=r"(complete)
        : "l"(__cvta_generic_to_shared(pBarrier)), "l"(token));
    return bool(complete);
  } else {
    return false;
  }
}

template <bool real>
using ParityOrNone = RealTypeOrNone<real, bool>;

template <bool real>
__device__ inline void wait_parity(CtaBarrier* pBarrier, ParityOrNone<real> parity) {
  assert(real == (pBarrier != nullptr));
  if constexpr (real) {
    pBarrier->wait_parity(parity);
  } else {
    __syncwarp();
  }
}

template <bool real>
__device__ inline bool test_wait_parity(CtaBarrier* pBarrier, ParityOrNone<real> parity) {
  assert(real == (pBarrier != nullptr));
  if constexpr (real) {
#if USE_CUSTOM_BARRIER
    return pBarrier->test_wait_parity(parity);
#else
    return pBarrier->try_wait_parity_for(parity, cuda::std::chrono::nanoseconds(0));
#endif
  } else {
    return false;
  }
}

template <bool real = true>
__device__ inline ParityOrNone<real>& flip(ParityOrNone<real>& flip) {
  if constexpr (real) {
    flip = !flip;
  }
  return flip;
}

template <bool real = true>
__device__ inline ParityOrNone<real> getAndFlip(ParityOrNone<real>& flag) {
  const ParityOrNone<real> ret = flag;
  if constexpr (real) {
    flag = !flag;
  }
  return ret;
}
