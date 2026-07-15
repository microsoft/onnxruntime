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

#include "cuda_hint.cuh"
#include "defines.h"
#if !(IS_MLA)
#include "ldgsts.cuh"
#include "mha.h"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#ifndef GENERATE_CUBIN
#include "hostUtils.h"
#include <cuda_runtime.h>
#ifndef NDEBUG
#include <cstdio>
#endif
#endif

// There are 4 ways to pass ctaRowMax backward from gemm1 warps to gemm0 warps:
//  1. Protect with xFwdBarriers+xBwdBarriers. This way, ctaRowMax is available to gemm0 warps together with x tiles and
//  warpRowMax/warpRowSum. But ctaRowMax is required before warp tile online softmax, while the other buffers is needed
//  only after online softmax. So xBwdBarriers wait will need to be moved before online softmax.
//  2. Similar to approach 1, but we add an additional register copy of ctaRowMax in gemm0 warps. It's loaded from smem
//  ctaRowMax after warp tile online softmax, so the current warp tile can't use it. But we can pass it to next
//  iteration so softmax of next tile can use it. The update will be delayed by 1 more iteration and we need one or two
//  more registers. Alternatively, put the extra copy in shared memory, so we have double buffer for ctaRowMax.
//  3. Protected with dedicated backward barriers (xFwdBarriers + ctaRowmaxBwdBarriers). Then we don't have drawbacks of
//  1 or 2, but we need extra smem barriers and extra arrive/wait instructions.
//  4. No protection, just use volatile read/write. This approach gives most timely update and has lowest cost, but the
//  result is non-deterministic up to an small numeric error.
// #define CTA_ROW_MAX_BACKWARD_METHOD 4
// 1 is 8% slower than 4. 2/3 are 10% slower than 4.
#define CTA_ROW_MAX_BACKWARD_METHOD 1

static_assert(inputElemSize >= cacheElemSize);

constexpr uint32_t cacheElemsPerGrain = exactDiv(grainBytes, cacheElemSize);
constexpr uint32_t inputElemsPerGrain = exactDiv(grainBytes, inputElemSize);
constexpr bool enableMicroFastPath = false;

// x: horizontal stacking for cta horizontal tile size
// y: vertical stacking for cta vertical tile size
// z: must be 2 for warp specialization.
constexpr uint3 ctaShapeInWarps = {4, 1, 2};

static_assert(ctaShapeInWarps.z == 2);  // for warp specialization
constexpr uint32_t nbWarpsPerCta = ctaShapeInWarps.x * ctaShapeInWarps.y * ctaShapeInWarps.z;
constexpr uint32_t ctaSize = warp_size * nbWarpsPerCta;

#if SPEC_DEC
// Use 32 row size
constexpr uint32_t nbValidRows = rowsPerBlock;
static_assert(nbValidRows <= 32u);
#else
constexpr uint32_t nbValidRows = headGrpSize * beamWidth;
#endif
constexpr uint2 warpTile = {64, roundUp(nbValidRows, 16U)};
static_assert(nbValidRows <= warpTile.y);

constexpr uint32_t gemm1WarpsPerGrp = exactDiv(headElems, warpTile.x);
constexpr uint32_t gemm1NbWarpGrps = exactDiv(ctaShapeInWarps.x, gemm1WarpsPerGrp);  // warp groups split along seqLen dim.

constexpr uint2 ctaTile = {warpTile.x * ctaShapeInWarps.x,  // if .x is greater than headSize, then gemm1 uses split-K
                           warpTile.y* ctaShapeInWarps.y};

constexpr uint32_t cvtExpansion = exactDiv(inputElemSize, cacheElemSize);

#ifndef __CUDA_ARCH__
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#else
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 900
constexpr uint32_t preferedKHeadPartBytes = 128;
__constant__ constexpr uint32_t cacheVTileSeqLen = 64;
#else
// Safe default for older or unknown architectures
constexpr uint32_t preferedKHeadPartBytes = 64;
__constant__ constexpr uint32_t cacheVTileSeqLen = 32;
#endif
#endif
constexpr uint32_t kHeadPartBytes = mha::min(preferedKHeadPartBytes, paddedCacheHeadBytes);
// constexpr uint32_t cacheElemsPerKHeadPart = exactDiv(kHeadPartBytes, cacheElemSize);

constexpr bool persistentQ = paddedInputHeadBytes * ctaTile.y <= (16u << 10);
static_assert(persistentQ);
constexpr uint32_t qHeadPartBytes = persistentQ ? paddedInputHeadBytes : kHeadPartBytes;
[[maybe_unused]] constexpr uint32_t qHeadPartElems = exactDiv(qHeadPartBytes, inputElemSize);

constexpr uint32_t nbPartsPerCacheKHead = exactDiv(paddedCacheHeadBytes, kHeadPartBytes);
[[maybe_unused]] constexpr uint32_t nbPartsPerInputKHead = exactDiv(paddedInputHeadBytes, kHeadPartBytes);
constexpr uint32_t nbPartsPerInputQHead = exactDiv(paddedInputHeadBytes, qHeadPartBytes);

// false - each warp load V tiles independent of each other; true - all warps in a warp group load V tiles together.
// @fixme: when true, and nbVBuffers is only 2, we need to sync all warps in a group after finishing using a buffer and
// before refill it with prefetch data. We may need at least 3.
constexpr bool grpLoadV = GRP_LOAD_V;

// number of shared memory buffers for latency hiding
constexpr uint32_t nbQBuffers = mha::min(nbPartsPerInputQHead, 2u);  // for latency hiding
constexpr uint32_t nbKBuffers = 2;                                   // for latency hiding
constexpr uint32_t nbVBuffers = 2;                                   // @fixme: H100 SXM need more in-flight requests. may need to increase this.
constexpr uint32_t nbXBuffers = 1;

__device__ inline uint3 getWarpIdx(Warp const& warp = this_warp()) {
  return uint3{ctaShapeInWarps.x == 1 ? 0 : makeWarpUniform(warp, threadIdx.x / warp_size),
               ctaShapeInWarps.y == 1 ? 0 : makeWarpUniform(warp, threadIdx.y),
               ctaShapeInWarps.z == 1 ? 0 : makeWarpUniform(warp, threadIdx.z)};
}

__device__ inline uint32_t gemm1WarpGrpIdx(uint32_t warpIdxX) {
  return gemm1NbWarpGrps == 1 ? 0 : warpIdxX / gemm1WarpsPerGrp;
}

__device__ inline uint32_t gemm1WarpIdxInGrp(uint32_t warpIdxX) {
  return gemm1WarpsPerGrp == 1 ? 0 : (gemm1NbWarpGrps == 1 ? warpIdxX : warpIdxX % gemm1WarpsPerGrp);
}

constexpr uint32_t instM = 16;
[[maybe_unused]] constexpr uint32_t instN = 8;
// constexpr uint32_t instK = 16;

using QuadRegRowMax = QuadRegRowMaxT<warpTile.y>;            // data is replicated across 4 threads in a MMA quad.
using ThrdRegRowMax = ThrdRegRowMaxT<warpTile.y>;            // unlike QuadRegRowMax, not replicated.
using UniformRescaleMask = UniformRescaleMaskT<warpTile.y>;  // uniform and stored in UR

__device__ inline bool any(UniformRescaleMask const& x) {
  uint32_t val = 0U;
#pragma unroll
  for (uint32_t i = 0; i < x.size; i++) {
    uint32_t word = x[i];
    constexpr uint32_t wordBits = 32;
    if (warpTile.y % wordBits != 0 && i + 1 == x.size) {
      constexpr uint32_t validBits = warpTile.y % wordBits;
      word &= ((1U << validBits) - 1);
    }
    val |= word;
  }
  return val != 0;
}

#ifndef NDEBUG
__device__ inline void printRowMax(ThrdRegRowMax const& src) {
  for (uint32_t i = 0; i < warp_size * src.size; i++) {
    if (laneId() == i % warp_size) {
      printf("%f%s", src[i / warp_size], i == 31 ? "\n" : " ");
    }
    __syncwarp();
  }
}

__device__ inline void printRowMax(QuadRegRowMax const& src) {
  for (uint32_t i = 0; i < src.size / 4; i++) {
    for (uint32_t j = 0; j < 8; j++) {
      if (laneId() == 4 * j) {
        for (uint32_t k = 0; k < 4; k++) {
          printf("%f%s", src[i * 4 + k], i == 31 ? "\n" : " ");
        }
      }
      __syncwarp();
    }
  }
}
#endif

struct alignas(16) SMemWarpRowMax {
  __device__ inline float const& operator[](uint32_t idxRow) const {
    assert(idxRow < ThrdRegRowMax::size * warp_size);
    uint32_t const idxInstM8 = idxRow / quadPerWarp;
    return data[ThrdRegRowMax::size == 1 ? 0 : idxInstM8 / 4][idxRow % quadPerWarp][idxInstM8 % 4];
  }

  __device__ inline float& operator[](uint32_t idxRow) {
    return const_cast<float&>(static_cast<SMemWarpRowMax const&>(*this)[idxRow]);
  }

  // When data is register, data is replicate across 4 threads in a quad.
  template <bool asVolatile>
  __device__ inline QuadRegRowMax const loadToRegForQuad(Warp const& warp) const {
    uint32_t const idxQuad = laneId() / 4;
    QuadRegRowMax result;
#pragma unroll
    for (uint32_t i = 0; i < divUp(warpTile.y, quadPerWarp * 4); i++) {
      auto const& src = data[i][idxQuad];
      auto& dst = reinterpret_cast<float (&)[4]>(result[4 * i]);
      if constexpr (asVolatile) {
        asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                     : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
                     : "l"(__cvta_generic_to_shared(&src)));
      } else {
        reinterpret_cast<float4&>(dst) = reinterpret_cast<float4 const&>(src);
      }
    }
    return result;
  }

  template <bool asVolatile>
  __device__ inline ThrdRegRowMax const loadToReg(Warp const& warp) const {
    ThrdRegRowMax result;
#pragma unroll
    for (uint32_t i = 0; i < result.size; i++) {
      auto const& src = this->operator[](warp_size * i + laneId());
      float& dst = result[i];
      if constexpr (asVolatile) {
        dst = static_cast<float const volatile&>(src);
        // asm volatile("ld.volatile.shared.f32 %0, [%1];\n"
        //     : "=f"(dst) : "l"(__cvta_generic_to_shared(&src)));
      } else {
        dst = src;
      }
    }
    return result;
  }

  template <bool asVolatile>
  __device__ inline void storeFromReg(Warp const& warp, QuadRegRowMax const& regData) {
    for (uint32_t i = 0; i < regData.size; i++) {
      assert(regData[i] == __shfl_sync(0xFU << (laneId() / 4 * 4), regData[i], 0, 4));
    }
    if (laneId() % 4 != 0) {
      return;
    }
    uint32_t const idxQuad = laneId() / 4;
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = data[i][idxQuad];
      auto const& src = reinterpret_cast<float const(&)[4]>(regData[4 * i]);
      if constexpr (asVolatile) {
        asm volatile(
            "st.volatile.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(__cvta_generic_to_shared(&dst)),
            "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
      } else {
        reinterpret_cast<float4&>(dst) = reinterpret_cast<float4 const&>(src);
      }
    }
  }

  template <bool asVolatile>
  __device__ inline void storeFromReg(Warp const& warp, ThrdRegRowMax const& regData) {
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = this->operator[](warp_size * i + laneId());
      assert(!hasBankConflict(&dst));
      float const src = regData[i];
      if constexpr (asVolatile) {
        static_cast<float volatile&>(dst) = src;
      } else {
        dst = src;
      }
    }
  }

  __device__ inline void atomicMaxUpdate(Warp const& warp, ThrdRegRowMax const& regData) {
#pragma unroll
    for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
      auto& dst = this->operator[](warp_size * i + laneId());
      assert(!hasBankConflict(&dst));
      float const src = regData[i];
      atomicMax(&dst, src);
    }
  }

  float data[ThrdRegRowMax::size][quadPerWarp][4];
};

// cacheVTileSeqLen may be smaller than x cols, so we need multiple v tiles per X tile.
constexpr uint32_t nbCacheVTilesPerXTile = exactDiv(warpTile.x, cacheVTileSeqLen);

[[maybe_unused]] constexpr uint32_t nbWarpGrpsPerXTile = mha::min(nbCacheVTilesPerXTile, gemm1NbWarpGrps);

#if USE_PAGED_KV_CACHE
constexpr uint32_t nbPagesPerWarpTile = (warpTile.x <= tokensPerPage ? 1U : exactDiv(warpTile.x, tokensPerPage));
using KCachePageIndices = Vec<KVCachePageIndex, nbPagesPerWarpTile>;
constexpr uint32_t nbPagesPerVTile = (cacheVTileSeqLen <= tokensPerPage ? 1 : exactDiv(cacheVTileSeqLen, tokensPerPage));
using VCachePageIndices = Vec<KVCachePageIndex, nbPagesPerVTile>;
#endif

static_assert(ctaShapeInWarps.y == 1);

struct alignas(128) SharedMem {
  using QSmemBuffer = Array2D<LdGrain, warpTile.y, exactDiv(qHeadPartBytes, grainBytes)>;
  using KSmemBuffer = Array2D<LdGrain, warpTile.x, exactDiv(kHeadPartBytes, grainBytes)>;
  using XSmemBuffer = Array2D<LdGrain, warpTile.y, exactDiv(inputElemSize* warpTile.x, grainBytes)>;
  using VSmemBuffer = Array2D<LdGrain, cacheVTileSeqLen, exactDiv(grpLoadV ? headElems : warpTile.x, cacheElemsPerGrain)>;

  QSmemBuffer q[ctaShapeInWarps.y][nbQBuffers];
  KSmemBuffer k[ctaShapeInWarps.x][nbKBuffers];
  XSmemBuffer x[ctaShapeInWarps.y][ctaShapeInWarps.x];
  static_assert(nbXBuffers == 1);
  VSmemBuffer v[gemm1NbWarpGrps][grpLoadV ? 1 : gemm1WarpsPerGrp][nbVBuffers];

  SMemWarpRowMax warpRowMax[ctaShapeInWarps.y][ctaShapeInWarps.x];  // the max used when computing this->x
  SMemWarpRowMax warpRowSum[ctaShapeInWarps.y][ctaShapeInWarps.x];  // the row sum of gemm0 output

#if CTA_ROW_MAX_BACKWARD_METHOD == 1 || CTA_ROW_MAX_BACKWARD_METHOD == 2 || CTA_ROW_MAX_BACKWARD_METHOD == 3
  // protected with xFwdBarriers+xBwdBarriers for CTA_ROW_MAX_BACKWARD_METHOD 1 or 2, and with
  // xFwdBarriers+ctaRowMaxBwdBarriers for 3. Cannot reuse warpRowMax because a gemm1 warp is not sure whether other
  // gemm1 warps have finished using it, unless we want to pay extra sync.
  SMemWarpRowMax ctaRowMax[ctaShapeInWarps.y][ctaShapeInWarps.x];
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
  SMemWarpRowMax ctaRowMax[ctaShapeInWarps.y];  // just a hint, no strict protection required if you don't care about
                                                // non-deterministic output (up to a small numeric error)
#endif

#if BEAM_WIDTH > 1
  Vec<uint32_t, warpTile.x> gemm0CacheIndir[ctaShapeInWarps.x];
  Vec<uint32_t, cacheVTileSeqLen> gemm1CacheIndir[grpLoadV ? gemm1NbWarpGrps : ctaShapeInWarps.x];
#if USE_PAGED_KV_CACHE
  Vec<KCachePageIndices, beamWidth> kCachePages[ctaShapeInWarps.x];
  Vec<VCachePageIndices, beamWidth> vCachePages[grpLoadV ? gemm1NbWarpGrps : ctaShapeInWarps.x];
#endif
#endif

  using Barrier = CtaBarrier;

  Barrier qBarrier[ctaShapeInWarps.y];
  // Beside X buffers, also protects warpRowMax and warpRowSum. For CTA_ROW_MAX_BACKWARD_METHOD==1 or 2, also
  // ctaRowMax.
  CtaBarrierPair xBarriers[ctaShapeInWarps.y][ctaShapeInWarps.x];
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
  Barrier ctaRowMaxBwdBarriers[ctaShapeInWarps.y]
                              [ctaShapeInWarps.x];  // xFwdBarriers+ctaRowMaxBwdBarriers protects ctaRowMax
#endif

#if GRP_LOAD_V
  static constexpr uint32_t nbOtherBarriers = nbVBuffers * gemm1NbWarpGrps + gemm1NbWarpGrps;
  Barrier otherBarriers[nbOtherBarriers];
#endif
  __device__ inline Barrier* vBarrier(uint32_t warpGrpIdx, uint32_t idxBuf) {
#if GRP_LOAD_V
    return &reinterpret_cast<Barrier(&)[gemm1NbWarpGrps][nbVBuffers]>(otherBarriers)[warpGrpIdx][idxBuf];
#else
    return nullptr;
#endif
  }

  __device__ inline Barrier* warpGrpBar(uint32_t warpGrpIdx) {
#if GRP_LOAD_V
    return &otherBarriers[nbVBuffers * gemm1NbWarpGrps + warpGrpIdx];
#else
    return nullptr;
#endif
  }
};

CUBIN_EXPORT __device__ constexpr uint32_t smemSize = sizeof(SharedMem);
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static_assert(smemSize < kMAX_SMEM_SIZE);
#endif

#if 0
template <bool swizzled, uint32_t rows, uint32_t cols>
__device__ inline void smemRotateInplace(Warp const& Warp, Array2D<LdGrain, rows, cols>& data, uint32_t idxPart, uint32_t idxToken) {
    static_assert(inputSeqLen == 1);
    constexpr uint32_t rowElems = inputElemsPerGrain * cols;
    constexpr uint32_t nbParts = exactDiv(headElems, idxPart);
    static_assert(nbParts % 2 == 0);
    bool const isFirstHalf = (idxPart < nbParts / 2);
    static_assert(mha::is_same_v<InputElem, half>, "not implemented");
    if constexpr (cols <= warp_size) {
        static_assert(warp_size % cols == 0);
        constexpr uint32_t thrdGrpSize = LdGrain::size * cols;
        uint32_t const idxThrdGrp = laneId() / thrdGrpSize;
        uint32_t const thrdGrpLane = laneId() % thrdGrpSize;
        constexpr uint32_t nbThrdGrps = warp_size / thrdGrpSize;
        static_assert(warp_size % thrdGrpSize == 0);
        constexpr uint32_t nbElemsPerWord = exactDiv(sizeof(LdGrain::Elem), inputElemSize);
        Vec<float, nbElemsPerWord> cosAngles;
        Vec<float, nbElemsPerWord> sinAngles;
#pragma unroll
        for (uint32_t i = 0; i < angles.size; i++) {
            uint32_t const n = rowElems * (idxPart % (nbParts / 2)) + angles.size * thrdGrpLane + i;
            float const angle = powf(1E-4f, n * (2.f / headElems)) * idxToken;
            sincosf(angle, &sinAngles[i], &cosAngles[i]);
        }

        constexpr uint32_t nbIters = exactDiv(rows, nbThrdGrps);
#pragma unroll
        for (uint32_t i = 0; i < nbIters; i++) {
            auto const word = data.template at<swizzled>(nbThrdGrps * i + idxThrdGrp, thrdGrpLane / LdGrain::size)[thrdGrpLane % LdGrain::size];
            float2 const val = __half22float2(reinterpret_cast<InputElem2 const&>(word));
            Vec<float, nbElemsPerWord> result;
#pragma unroll
            for (uint32_t j = 0; j < nbElemsPerWord; j++) {
                if (isFirstHalf) {
                    result[j] = cosAngles[j] * ;
                }
            }
        }
    }
    else {
        static_assert(cols <= warp_size, "not implemented");
    }
}
#endif

using WarpAcc = WarpAccT<warpTile.y, warpTile.x>;

#if SPEC_DEC
#define MMAS_N_PER_MASK 2

__device__ inline void applyMaskFromInput(Warp const& warp, WarpAcc& acc, MaskType const* mask, uint32_t rowOffset,
                                          uint32_t nbValidCols, uint32_t qSeqLen, uint32_t actualQSeqLen, uint32_t headGrpSize
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                                          ,
                                          int32_t tok0WinBeg, uint32_t seqIter, uint32_t const cacheSeqLen, uint32_t const warpTileTokenBeg
#endif
) {
  uint32_t const idxInQuad = laneId() % 4;
  uint32_t const idxQuad = laneId() / 4;
  // Packed mask is aligned with 32 bits (2 uint16_t).
  uint32_t const nbPackedMasksPerRow = divUp(qSeqLen, 32u) * 2u;
  uint16_t const* uint16Mask = reinterpret_cast<uint16_t const*>(mask);
  constexpr uint64_t fullMask = ~uint64_t{0};
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
  Range const tileRange = {warpTileTokenBeg, warpTileTokenBeg + warpTile.x};
  Range const maxMaskOutRange = {0, mha::max(0, tok0WinBeg) + (nbValidRows / MMAS_N_PER_MASK - 1)};
  bool const ctaNeedBegMask = tileRange.beg < maxMaskOutRange.end;
  assert(ctaNeedBegMask == overlap(tileRange, maxMaskOutRange));
  int32_t const tok0NbMaskOut = int32_t(tok0WinBeg) - int32_t(warpTileTokenBeg);
  uint32_t const nbSeqItersWithoutSpecDecMask = (cacheSeqLen - actualQSeqLen) / ctaTile.x;
  bool const ctaNeedSpecDecMask = (seqIter >= nbSeqItersWithoutSpecDecMask);
#else
  constexpr bool ctaNeedBegMask = false;
  bool const ctaNeedSpecDecMask = true;
  int32_t const tok0NbMaskOut = -2147483648;
#endif
  bool const needMask = ctaNeedBegMask || ctaNeedSpecDecMask;

  if (!needMask) {
    return;
  }
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      uint32_t const idxQTokInCta = (rowOffset + instM * m + idxQuad + i * 8) / headGrpSize;
      uint32_t const tokenRow = min(idxQTokInCta, actualQSeqLen - 1);
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
      int32_t const begNbMaskOut = tok0NbMaskOut + int32_t(idxQTokInCta);
      uint64_t const begMask = (begNbMaskOut > 0 ? fullMask << begNbMaskOut : fullMask);
#else
      uint64_t const begMask = fullMask;
#endif

#pragma unroll
      for (uint32_t mask_n = 0; mask_n < acc.cols / MMAS_N_PER_MASK; mask_n++) {
        uint32_t const firstCol = instN * mask_n * MMAS_N_PER_MASK + InstAcc::cols * idxInQuad;
        uint32_t const lastCol = firstCol + instN * (MMAS_N_PER_MASK - 1) + InstAcc::cols - 1;
        uint32_t const maskPos0 = firstCol + actualQSeqLen < nbValidCols
                                      ? 0u
                                      : min(firstCol + actualQSeqLen - nbValidCols, actualQSeqLen - 1);
        uint32_t const maskPos1 = lastCol + actualQSeqLen < nbValidCols
                                      ? 0u
                                      : min(lastCol + actualQSeqLen - nbValidCols, actualQSeqLen - 1);
        uint32_t const maskPosStart = (maskPos0 / 16) * 16;
        uint32_t packedMask = ~uint32_t{0};
        if (ctaNeedSpecDecMask) {
          reinterpret_cast<uint16_t*>(&packedMask)[0] = uint16Mask[tokenRow * nbPackedMasksPerRow + (maskPos0 / 16)];
          reinterpret_cast<uint16_t*>(&packedMask)[1] = uint16Mask[tokenRow * nbPackedMasksPerRow + (maskPos1 / 16)];
        }
#pragma unroll
        for (uint32_t nj = 0; nj < MMAS_N_PER_MASK; nj++) {
#pragma unroll
          for (uint32_t j = 0; j < InstAcc::cols; j++) {
            uint32_t const n = (mask_n * MMAS_N_PER_MASK + nj);
            uint32_t const col = instN * n + InstAcc::cols * idxInQuad + j;
            // bool const maskFlag = col + qSeqLen < nbValidCols ? true : mask[tokenRow * qSeqLen + (col +
            // qSeqLen - nbValidCols)];
            bool const maskFlag = col + actualQSeqLen < nbValidCols
                                      ? true
                                      : packedMask & (1u << ((col + actualQSeqLen - nbValidCols) - maskPosStart));

            bool const begMaskFlag = ctaNeedBegMask ? (begMask & (1ULL << col)) : true;

            acc(m, n)(i, j) = maskFlag && begMaskFlag && col < nbValidCols ? acc(m, n)(i, j) : safeInitRowMax;
          }
        }
      }
    }
  }
}
#endif

__device__ inline QuadRegRowMax warpTileOnlineSoftmax(Warp const& warp, QuadRegRowMax const& rowMaxHint, WarpAcc& acc) {
  QuadRegRowMax rowMax = rowMaxHint;
// compute per-thread row max
#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++) {
          rowMax[m * InstAcc::rows + i] = fmaxf(rowMax[m * InstAcc::rows + i], acc(m, n)(i, j));
        }
      }
    }
  }
// compute warp row max
#pragma unroll
  for (uint32_t xorMask = 2; xorMask != 0; xorMask /= 2) {
#pragma unroll
    for (uint32_t i = 0; i < rowMax.size; i++) {
      rowMax[i] = fmaxf(rowMax[i], __shfl_xor_sync(~0U, rowMax[i], xorMask));
    }
  }
// update acc and rowMax
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      float const maxVal = rowMax[m * InstAcc::rows + i];
      float const bias = maxVal * log2e;
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          float& elem = acc(m, n)(i, j);
          assert(maxVal >= elem);
          elem = exp2f(elem * log2e - bias);
        }
      }
    }
  }
  return rowMax;
}

using GemmOutRegTile = Array2D<InputElem2, WarpAcc::rows * InstAcc::rows, WarpAcc::cols * exactDiv(InstAcc::cols, 2)>;

__device__ inline GemmOutRegTile toFp16(WarpAcc const& acc) {
  GemmOutRegTile dst;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j += 2) {
#if INPUT_FP16
          dst(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2) = __floats2half2_rn(acc(m, n)(i, j), acc(m, n)(i, j + 1));
#else
          dst(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2) = __floats2bfloat162_rn(acc(m, n)(i, j), acc(m, n)(i, j + 1));
#endif
        }
      }
    }
  }
  return dst;
}

__device__ inline WarpAcc toWarpAcc(GemmOutRegTile const& outTile) {
  WarpAcc acc;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j += 2) {
#if INPUT_FP16
          float2 const fp32Vals = __half22float2(outTile(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2));
#else
          float2 const fp32Vals = __bfloat1622float2(outTile(m * InstAcc::rows + i, (n * InstAcc::cols + j) / 2));
#endif
          acc(m, n)(i, j) = fp32Vals.x;
          acc(m, n)(i, j + 1) = fp32Vals.y;
        }
      }
    }
  }
  return acc;
}

__device__ inline QuadRegRowMax computeRowSum(Warp const& warp, GemmOutRegTile const& src) {
  Vec<InstAcc, exactDiv(GemmOutRegTile::rows, InstAcc::rows)> acc{};
#if INPUT_FP16
  InputElem2 const b[2][1] = {__floats2half2_rn(1, 1), __floats2half2_rn(1, 1)};
#else
  InputElem2 const b[2][1] = {__floats2bfloat162_rn(1, 1), __floats2bfloat162_rn(1, 1)};
#endif
#pragma unroll
  for (uint32_t n = 0; n < exactDiv(GemmOutRegTile::cols, 2); n++) {
#pragma unroll
    for (uint32_t m = 0; m < exactDiv(GemmOutRegTile::rows, 2); m++) {
      InputElem2 const a[2 /*kEx*/][2 /*mEx*/] = {src(m * 2, n * 2), src(m * 2 + 1, n * 2), src(m * 2, n * 2 + 1), src(m * 2 + 1, n * 2 + 1)};
      mma<InputElem>(acc[m].data, reinterpret_cast<uint32_t const(&)[2][2]>(a),
                     reinterpret_cast<uint32_t const(&)[2][1]>(b));
    }
  }
  QuadRegRowMax rowSum;
#pragma unroll
  for (uint32_t i = 0; i < acc.size; i++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::rows; j++) {
      rowSum[i * InstAcc::rows + j] = acc[i](j, 0);
#pragma unroll
      for (uint32_t k = 0; k < InstAcc::cols; k++) {
        assert(acc[i](j, k) == acc[i](j, 0));
      }
    }
    rowSum[i * 2] = acc[i](0, 0);
    rowSum[i * 2 + 1] = acc[i](1, 0);
  }
// Sometimes there are errors in sum and they mismatch inside a quad. Force broadcast from lane 0 of each quad to
// eliminate mismatch. This has no visible impact on final result and can be removed.
#pragma unroll
  for (uint32_t i = 0; i < QuadRegRowMax::size; i++) {
    auto const lane0Val = __shfl_sync(0xFU << (laneId() / 4 * 4), rowSum[i], 0, 4);
    // Disable the assert, sometimes it triggers because of different orders of accumulation.
    // assert(fabs(rowSum[i] - lane0Val) < 1E-4f);
    rowSum[i] = lane0Val;
  }
  return rowSum;
}

__device__ inline void storeOrderedGemmOutTile(Warp const& warp, SharedMem::XSmemBuffer& dst, GemmOutRegTile const& src) {
  static_assert(sizeof(dst) == sizeof(src) * warp_size);
  uint32_t const lane = laneId();
#if __CUDA_ARCH__ >= 900
  constexpr uint2 storeUnits = {4, 1};  // in 8x8 b16 matrices.
  static_assert(storeUnits.x * storeUnits.y == 4);
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8 * storeUnits.y); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8 * storeUnits.x); n++) {
      uint32_t const idxRowLocal = lane % 8;
      uint32_t const flatIdxMatLocal = lane / 8;
      uint2 const idxMatLocal = {flatIdxMatLocal % storeUnits.x, flatIdxMatLocal / storeUnits.x};
      LdGrain* const p = &dst.template at<true>(
          8 * (storeUnits.y * m + idxMatLocal.y) + idxRowLocal, storeUnits.x * n + idxMatLocal.x);

      LdGrain data;
#pragma unroll
      for (uint32_t i = 0; i < storeUnits.y; i++) {
#pragma unroll
        for (uint32_t j = 0; j < storeUnits.x; j++) {
          data[i * storeUnits.x + j] = reinterpret_cast<uint32_t const&>(src(m * storeUnits.y + i, n * storeUnits.x + j));
        }
      }
      stmatrix_4x<false>(warp, p, data);
    }
  }
#else
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8); n++) {
      uint32_t const idxRowLocal = laneId() / 4;
      uint32_t const idxWordLocal = laneId() % 4;
      dst.template at<true>(8 * m + idxRowLocal, n)[idxWordLocal] = reinterpret_cast<uint32_t const&>(src(m, n));
    }
  }
#endif
}

// Reorder to compensate the reorder caused by V cache load+conversion.
__device__ inline void reorderAndStoreGemmOutTile(
    Warp const& warp, SharedMem::XSmemBuffer& dst, GemmOutRegTile const& src) {
  static_assert(sizeof(dst) == sizeof(src) * warp_size);
  uint32_t const lane = laneId();
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(dst.rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(dst.cols * grainBytes / inputElemSize, 8 * 2); n++) {
      uint32_t const idxRowLocal = laneId() / 4;
      uint32_t const idxSegLocal = laneId() % 4;
      Vec<InputElem2, cvtExpansion> seg;
#pragma unroll
      for (uint32_t e = 0; e < cvtExpansion; e++) {
        seg[e] = src(m, n * cvtExpansion + e);
      }
      // reorder
      // Ideally compiler should be able to fuse this into toFp16() and just reorder input registers of F2FP
      // instructions.
      Vec<InputElem, cvtExpansion * 2> reorderedSeg;
#pragma unroll
      for (uint32_t e = 0; e < cvtExpansion; e++) {
        reorderedSeg[e] = seg[e].x;
        reorderedSeg[cvtExpansion + e] = seg[e].y;
      }
      static_assert(cvtExpansion <= LdGrain::size);
      constexpr uint32_t nbSegPerGrain = exactDiv(grainBytes, sizeof(seg));
      reinterpret_cast<Vec<uint32_t, cvtExpansion>&>(dst.template at<true>(8 * m + idxRowLocal,
                                                                           n * cvtExpansion + idxSegLocal / nbSegPerGrain)[idxSegLocal % nbSegPerGrain * cvtExpansion]) = reinterpret_cast<Vec<uint32_t, cvtExpansion>&>(reorderedSeg);
    }
  }
}

__device__ inline void storeGemmOutTile(
    Warp const& warp, SharedMem::XSmemBuffer& dst, GemmOutRegTile const& src, bool reorder) {
  if (reorder) {
    reorderAndStoreGemmOutTile(warp, dst, src);
  } else {
    storeOrderedGemmOutTile(warp, dst, src);
  }
}

__device__ inline GemmOutRegTile loadGemmOutTile(Warp const& warp, SharedMem::XSmemBuffer const& src) {
  uint32_t const lane = laneId();
  GemmOutRegTile dst;
  static_assert(sizeof(src) == sizeof(dst) * warp_size);
#if __CUDA_ARCH__ >= 900
  constexpr uint2 storeUnits = {4, 1};  // in 8x8 b16 matrices.
  static_assert(storeUnits.x * storeUnits.y == 4);
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(SharedMem::XSmemBuffer::rows, 8 * storeUnits.y); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(SharedMem::XSmemBuffer::cols * grainBytes / inputElemSize, 8 * storeUnits.x);
         n++) {
      uint32_t const idxRowLocal = lane % 8;
      uint32_t const flatIdxMatLocal = lane / 8;
      uint2 const idxMatLocal = {flatIdxMatLocal % storeUnits.x, flatIdxMatLocal / storeUnits.x};
      LdGrain const* const p = &src.template at<true>(
          8 * (storeUnits.y * m + idxMatLocal.y) + idxRowLocal, storeUnits.x * n + idxMatLocal.x);

      LdGrain data = ldmatrix_4x<false>(warp, p);
#pragma unroll
      for (uint32_t i = 0; i < storeUnits.y; i++) {
#pragma unroll
        for (uint32_t j = 0; j < storeUnits.x; j++) {
          reinterpret_cast<uint32_t&>(dst(m * storeUnits.y + i, n * storeUnits.x + j)) = data[i * storeUnits.x + j];
        }
      }
    }
  }
#else
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(SharedMem::XSmemBuffer::rows, 8); m++) {
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(SharedMem::XSmemBuffer::cols * grainBytes / inputElemSize, 8); n++) {
      uint32_t const idxRowLocal = laneId() / 4;
      uint32_t const idxWordLocal = laneId() % 4;
      reinterpret_cast<uint32_t&>(dst(m, n)) = src.template at<true>(8 * m + idxRowLocal, n)[idxWordLocal];
    }
  }
#endif
  return dst;
}
// only the first nbValidRows rows are copied, to allow padding.
__device__ inline void copyOutputToGlobalMem(Warp const& warp, OutputHead* dst, uint32_t nbQHeads,
#if SPEC_DEC
                                             uint32_t headGrpSize, uint32_t idxHeadGrpOffset, uint32_t nbValidHeadTokens,
#else
                                             uint32_t idxHeadGrp,
#endif
                                             uint2 dstOffset, SharedMem::XSmemBuffer const& src) {
  static_assert(sizeof(PaddedInputHead) == grainBytes * SharedMem::XSmemBuffer::cols * gemm1WarpsPerGrp);
#if SPEC_DEC
  static_assert(warpTile.y <= SharedMem::XSmemBuffer::rows);
#else
  static_assert(nbValidRows <= SharedMem::XSmemBuffer::rows);
#endif
  constexpr uint32_t nbIters = divUp(nbValidRows * SharedMem::XSmemBuffer::cols, warp_size);
#pragma unroll
  for (uint32_t i = 0; i < nbIters; i++) {
    uint32_t const flatIdx = warp_size * i + laneId();
    uint32_t const r = flatIdx / SharedMem::XSmemBuffer::cols;
    uint32_t const c = flatIdx % SharedMem::XSmemBuffer::cols;
    assert(r < SharedMem::XSmemBuffer::rows);
    LdGrain const data = src.template at<true>(r, c);

    uint32_t const m = dstOffset.y + r;
    uint32_t const n = exactDiv(dstOffset.x, grainBytes / inputElemSize) + c;
#if SPEC_DEC
    if (r >= nbValidHeadTokens) {
#else
    if (nbValidRows * SharedMem::XSmemBuffer::cols % warp_size != 0 && m >= nbValidRows) {
#endif
      break;
    }
    assert(m < nbValidRows);
#if SPEC_DEC
    uint32_t const idxBeam = 0;
    uint32_t const idxInGrp = m;
    uint32_t const tokenIdx = idxInGrp / headGrpSize;
    uint32_t const headIdx = idxInGrp % headGrpSize;
    assert(idxBeam < beamWidth);
    uint32_t const idxHead = idxHeadGrpOffset + tokenIdx * nbQHeads + headIdx;
    assert(idxHead < nbValidHeadTokens * nbQHeads);
#else
    uint32_t const idxBeam = m / headGrpSize;
    uint32_t const idxInGrp = m % headGrpSize;
    assert(idxBeam < beamWidth);
    uint32_t const idxHead = headGrpSize * idxHeadGrp + idxInGrp;
    assert(idxHead < nbQHeads);
#endif
    assert(n < paddedInputHeadBytes / grainBytes);
    if (!isHeadPadded || n < ioHeadBytes / grainBytes) {
      auto const outVec = convert<OutputHead::Elem>(reinterpret_cast<Vec<InputElem, inputElemsPerGrain> const&>(data));
      reinterpret_cast<Vec<mha::decay_t<decltype(outVec)>, exactDiv(ioHeadBytes, grainBytes)>&>(
          dst[nbQHeads * idxBeam + idxHead])[n] = outVec;
    }
  }
}

// MMA instruction expansion in GEMM k-dim and m/n-dim, with b16 8x8 as baseline
template <uint32_t kEx_, uint32_t mnEx_>
struct InstInMat {
  static constexpr uint32_t kEx = kEx_;
  static constexpr uint32_t mnEx = mnEx_;
  uint32_t data[kEx][mnEx];
};

template <uint32_t kEx, uint32_t mnEx, bool transOuter>
using InstInMatWTrans = InstInMat<transOuter ? mnEx : kEx, transOuter ? kEx : mnEx>;

//@fixme: for B-mat, use InstInMat<2, 1>[2] instead.

// kEx is for srcCol and mnEx is for srcRow, before transpose.
// rowBeg/colBeg are in src indices
// note that grainBytes-byte swizzling per 128-byte or per row(>=128byte) is applied when loading to avoid bank
// conflict. transOuter: transpose InstInMat with 8x8 b16 matrices as elements unchanged. transInner: transpose the
// elements, i.e. the 8x8 b16 matrices. transOuter=true and transInner=false is for B matrix of 16816. It actually loads
// two 8x16 B matrices for two instructions. transOuter=false and transInner=false is for A matrix of 16816.
template <uint32_t kEx, uint32_t mnEx, bool transOuter, bool transInner, uint32_t srcRows, uint32_t srcCols>
__device__ inline InstInMatWTrans<kEx, mnEx, transOuter> loadInstInMat(
    Warp const& warp, Array2D<LdGrain, srcRows, srcCols> const& src, uint32_t rowOffset, uint32_t colOffset) {
  static_assert(kEx * mnEx == 4, "implemented only for ldmatrix.x4 for now");
  using Dst = InstInMatWTrans<kEx, mnEx, transOuter>;
  assert(rowOffset % (8 * mnEx) == 0 && colOffset % kEx == 0);
  uint32_t const idx = laneId() / 8;
  uint32_t const idxKEx = idx / Dst::mnEx;
  uint32_t const idxMNEx = idx % Dst::mnEx;
  uint32_t const srcIdxKEx = (transOuter ? idxMNEx : idxKEx);
  uint32_t const srcIdxMNEx = (transOuter ? idxKEx : idxMNEx);

  LdGrain const* const ptr = &src.template at<true>(rowOffset + 8 * srcIdxMNEx + laneId() % 8, colOffset + srcIdxKEx);

  Vec<uint32_t, 4> const data = ldmatrix_4x<transInner>(warp, ptr);
  static_assert(sizeof(Dst) == sizeof(data));
  Dst dst;
#pragma unroll
  for (int i = 0; i < data.size; i++) {
    (&dst.data[0][0])[i] = data[i];
  }
  return dst;
}

template <typename T, uint32_t rows, uint32_t cols, bool transpose>
using Array2DWTrans = Array2D<T, transpose ? cols : rows, transpose ? rows : cols>;

// src rows/cols are in src indices
// dst rows/cols are in InstInMatWTrans
// row is contiguous and gemm-K dim.
// kEx combines with dstCols and mnEx combines with dstRows.
template <uint32_t kEx, uint32_t mnEx, uint32_t dstRows, uint32_t dstCols, bool transArr2D, bool transInstInMatOuter,
          bool transInstInMatInner, uint32_t srcRows, uint32_t srcCols /*in LdGrain*/>
__device__ inline Array2DWTrans<InstInMatWTrans<kEx, mnEx, transInstInMatOuter>, dstRows, dstCols, transArr2D>
loadMatrix(Warp const& warp, Array2D<LdGrain, srcRows, srcCols> const& src, uint32_t rowBeg, uint32_t colBeg) {
  assert(rowBeg % (8 * mnEx * dstRows) == 0 && colBeg % (kEx * dstCols) == 0);
  Array2DWTrans<InstInMatWTrans<kEx, mnEx, transInstInMatOuter>, dstRows, dstCols, transArr2D> dst;
#pragma unroll
  for (uint32_t i = 0; i < dstRows; i++) {
#pragma unroll
    for (uint32_t j = 0; j < dstCols; j++) {
      (transArr2D ? dst(j, i) : dst(i, j)) = loadInstInMat<kEx, mnEx, transInstInMatOuter, transInstInMatInner>(
          warp, src, rowBeg + (mnEx * 8) * i, colBeg + kEx * j);
    }
  }
  return dst;
}

// acc is used as both input and output
// qColBeg is in the unit of LdGrain
// using KElemType = int8_t;
template <typename KElemType>
__device__ inline void smemQKPartGemm(
    Warp const& warp, WarpAcc& acc, SharedMem::QSmemBuffer const& q, uint32_t qColBeg, SharedMem::KSmemBuffer const& k) {
  assert(qColBeg % (SharedMem::KSmemBuffer::cols) == 0);
  constexpr uint32_t kEx = 2;
  constexpr uint32_t mnEx = 2;
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert((mha::is_same_v<KElemType, half> || mha::is_same_v<KElemType, __nv_bfloat16> || mha::is_same_v<KElemType, int8_t> || mha::is_same_v<KElemType, __nv_fp8_e4m3>),
                "not implemented");
  constexpr uint32_t nbInstInMatPerSliceInGemmKDim = 1;
  constexpr uint32_t kElemSize = sizeof(KElemType);
  constexpr uint32_t elemsPerKHeadPart = exactDiv(kHeadPartBytes, kElemSize);
  constexpr uint32_t gemmKSplit = exactDiv(elemsPerKHeadPart, 8 * kEx * nbInstInMatPerSliceInGemmKDim);

  // @fixme: check if compiler mixes LDS+HMMA and does prefetch properly. We are not doing prefetch explicitly. But we
  // do fully unroll and expect compiler to do that for us.
  constexpr uint32_t nbUnroll = cacheElemSize == 2 ? gemmKSplit : 2;
#pragma unroll(nbUnroll)
  for (uint32_t s = 0; s < gemmKSplit; s++) {
    // load q
    constexpr uint32_t qSliceRows = exactDiv(warpTile.y, 8 * mnEx);  // in InstInMat
    constexpr uint32_t qSliceCols = nbInstInMatPerSliceInGemmKDim;
    Array2D<InstInMat<kEx, mnEx>, qSliceRows, qSliceCols> const qSlice = loadMatrix<kEx, mnEx, qSliceRows, qSliceCols, false, false, false>(
        warp, q, 0, qColBeg + kEx * qSliceCols * s);
    // load k
    constexpr uint32_t cvtExp = exactDiv(inputElemSize, kElemSize);
    constexpr uint32_t mnExK = mnEx * cvtExp;
    constexpr uint32_t kExK = exactDiv(kEx, cvtExp);
    constexpr uint32_t kSliceRows = exactDiv(warpTile.x, 8 * mnExK);  // in InstInMat
    constexpr uint32_t kSliceCols = nbInstInMatPerSliceInGemmKDim;
    Array2D<InstInMat<mnExK, kExK>, kSliceRows, kSliceCols> const kSliceOrig = loadMatrix<kExK, mnExK, kSliceRows, kSliceCols, false, true, false>(warp, k, 0, kExK * kSliceCols * s);
    auto const kSlice = [&]() -> Array2D<InstInMat<mnExK, kEx>, kSliceRows, kSliceCols> {
      if constexpr (mha::is_same_v<InputElem, KElemType>) {
        return kSliceOrig;
      } else if constexpr ((mha::is_same_v<KElemType, int8_t> || mha::is_same_v<KElemType, __nv_fp8_e4m3>)) {
        Array2D<InstInMat<mnExK, kEx>, kSliceRows, kSliceCols> ret;
#pragma unroll
        for (uint32_t m = 0; m < kSliceRows; m++) {
#pragma unroll
          for (uint32_t n = 0; n < kSliceCols; n++) {
#pragma unroll
            for (uint32_t i = 0; i < mnExK; i++) {
#pragma unroll
              for (uint32_t j = 0; j < kExK; j++) {
                auto const data = convertKCacheWordToF16<InputElem, KElemType>(kSliceOrig(m, n).data[i][j]);
                ret(m, n).data[i][j * cvtExp] = data[0];
                ret(m, n).data[i][j * cvtExp + 1] = data[1];
              }
            }
          }
        }
        return ret;
      } else {
        assert(!"not implemented");
        trap();
      }
    }();
// compute
#pragma unroll
    for (uint32_t i = 0; i < qSliceRows; i++) {
#pragma unroll
      for (uint32_t j = 0; j < kSliceRows; j++) {
        InstInMat<kEx, mnEx> const matrixA = qSlice(i, 0);
        InstInMat<mnExK, kEx> const matrixB = kSlice(j, 0);
#pragma unroll
        for (uint32_t n = 0; n < mnExK; n++) {
          uint32_t const b[2][1] = {matrixB.data[n][0], matrixB.data[n][1]};
          mma<InputElem>(acc(i, j * mnExK + n).data, matrixA.data, b);
        }
      }
    }
  }
}

// acc is used as both input and output
// v needs transpose
template <typename VElemType>
__device__ inline void smemXVPartGemm(Warp const& warp, WarpAcc& acc, bool skipXRowRescale,
                                      UniformRescaleMask xRowNeedRescaleMask, ThrdRegRowMax xRowScales, SharedMem::XSmemBuffer const& x,
                                      uint32_t idxVTilePerXTile, SharedMem::VSmemBuffer const& vt, uint32_t idxNSplit) {
  static_assert(mha::is_same_v<InputElem, half> || mha::is_same_v<InputElem, __nv_bfloat16>, "not implemented");
  static_assert((mha::is_same_v<VElemType, half> || mha::is_same_v<VElemType, __nv_bfloat16> || mha::is_same_v<VElemType, int8_t> || mha::is_same_v<VElemType, __nv_fp8_e4m3>),
                "not implemented");
  constexpr uint32_t kEx = 2;
  constexpr uint32_t mnEx = 2;
  constexpr uint32_t nbInstInMatPerSliceInGemmKDim = 1;
  static_assert(SharedMem::XSmemBuffer::rows == 8 * InstAcc::rows * WarpAcc::rows);
  static_assert(
      grpLoadV || sizeof(SharedMem::VSmemBuffer::Elem) / cacheElemSize * SharedMem::VSmemBuffer::cols == warpTile.x);
  static_assert(
      !grpLoadV || sizeof(SharedMem::VSmemBuffer::Elem) / cacheElemSize * SharedMem::VSmemBuffer::cols == headElems);
  if (grpLoadV) {
    assert(idxNSplit < gemm1WarpsPerGrp);
  } else {
    assert(idxNSplit == 0);
  }
  constexpr uint32_t gemmKSplit = exactDiv(SharedMem::VSmemBuffer::rows, 8 * kEx * nbInstInMatPerSliceInGemmKDim);

  Vec<InputElem2, QuadRegRowMax::size> xRowScalesQuad;
  if (!enableMicroFastPath || !skipXRowRescale) {
    assertWarpConverged();
#if INPUT_FP16
    Vec<InputElem2, ThrdRegRowMax::size> const xRowScalesF16 = __float2half2_rn(xRowScales);
#else
    Vec<InputElem2, ThrdRegRowMax::size> const xRowScalesF16 = __float2bfloat162_rn(xRowScales);
#endif
    static_assert(sizeof(xRowScalesF16) == sizeof(ThrdRegRowMax));
    reinterpret_cast<QuadRegRowMax&>(xRowScalesQuad) = replicateForQuad(warp, reinterpret_cast<ThrdRegRowMax const&>(xRowScalesF16));
  }

// @fixme: check if compiler mixes LDS+HMMA and does prefetch properly. We are not doing prefetch explicitly. But we do
// fully unroll and expect compiler to do that for us.
#pragma unroll
  for (uint32_t s = 0; s < gemmKSplit; s++) {
    // load x
    constexpr uint32_t xSliceRows = exactDiv(warpTile.y, 8 * mnEx);  // in InstInMat
    constexpr uint32_t xSliceCols = nbInstInMatPerSliceInGemmKDim;
    uint32_t const colBeg = SharedMem::XSmemBuffer::cols / nbCacheVTilesPerXTile * idxVTilePerXTile + exactDiv(inputElemSize * 8 * kEx * nbInstInMatPerSliceInGemmKDim, grainBytes) * s;
    Array2D<InstInMat<kEx, mnEx>, xSliceRows, xSliceCols> xSlice = loadMatrix<kEx, mnEx, xSliceRows, xSliceCols, false, false, false>(warp, x, 0u, colBeg);
    if (!enableMicroFastPath || !skipXRowRescale) {
#pragma unroll
      for (uint32_t m = 0; m < xSliceRows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < mnEx; i++) {
          uint32_t const r = m * mnEx + i;
#pragma unroll
          for (uint32_t n = 0; n < xSliceCols; n++) {
#pragma unroll
            for (uint32_t j = 0; j < kEx; j++) {
              InputElem2& elem = reinterpret_cast<InputElem2&>(xSlice(m, n).data[j][i]);
              elem = skipXRowRescale ? elem : elem * xRowScalesQuad[r];
            }
          }
        }
      }
    }
    // load v slice. rows and cols here are before transpose
    constexpr uint32_t mnExV = mnEx * cvtExpansion;
    constexpr uint32_t vSliceCols = exactDiv(warpTile.x, 8 * mnExV);  // in InstInMat
    constexpr uint32_t vSliceRows = nbInstInMatPerSliceInGemmKDim;
    uint32_t const rowBeg = 8 * kEx * nbInstInMatPerSliceInGemmKDim * s;
    Array2D<InstInMat<mnEx, kEx>, vSliceCols, vSliceRows> const vSliceOrig = loadMatrix<mnEx, kEx, vSliceRows, vSliceCols, true, false, true>(
        warp, vt, rowBeg, mnEx * vSliceCols * idxNSplit);
    Array2D<InstInMat<mnExV, kEx>, vSliceCols, vSliceRows> const vSlice = [&]() {
      if constexpr (mha::is_same_v<InputElem, VElemType>) {
        return vSliceOrig;
      } else if constexpr ((mha::is_same_v<VElemType, int8_t> || mha::is_same_v<VElemType, __nv_fp8_e4m3>)) {
        Array2D<InstInMat<mnExV, kEx>, vSliceCols, vSliceRows> ret;
#pragma unroll
        for (uint32_t m = 0; m < ret.rows; m++) {
#pragma unroll
          for (uint32_t n = 0; n < ret.cols; n++) {
            auto const& src = vSliceOrig(m, n);
            auto& dst = ret(m, n);
#pragma unroll
            for (uint32_t i = 0; i < mnEx; i++) {
#pragma unroll
              for (uint32_t j = 0; j < kEx; j++) {
                auto const data = convertVCacheWordToF16<InputElem, VElemType>(src.data[i][j]);
#pragma unroll
                for (uint32_t e = 0; e < cvtExpansion; e++) {
                  dst.data[i * cvtExpansion + e][j] = data[e];
                }
              }
            }
          }
        }
        return ret;
      } else {
        assert(!"not implemented");
        trap();
      }
    }();
// compute
#pragma unroll
    for (uint32_t i = 0; i < xSliceRows; i++) {
#pragma unroll
      for (uint32_t j = 0; j < vSliceCols; j++) {
        auto const& vInMat = vSlice(j, 0);
#pragma unroll
        for (uint32_t n = 0; n < mnExV; n++) {
          mma<InputElem>(acc(i, j * mnExV + n).data, xSlice(i, 0).data,
                         reinterpret_cast<uint32_t const(&)[2][1]>(vInMat.data[n]));
        }
      }
    }
  }
}

__device__ inline void pickAccRowsForBeamSearch(Warp const& warp, WarpAcc& dst, WarpAcc const& src, bool isCtxTile,
                                                uint32_t idxBeam, void (*func)(float& d, float s)) {
  uint32_t const idxQuad = laneId() / 4;
  constexpr uint32_t nbQuads = warp_size / 4;
#pragma unroll
  for (uint32_t m = 0; m < WarpAcc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < WarpAcc::cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          uint32_t const idxRow = instM * m + nbQuads * i + idxQuad;
          if (isCtxTile || (idxRow >= headGrpSize * idxBeam && idxRow < headGrpSize * idxBeam + headGrpSize)) {
            func(dst(m, n)(i, j), src(m, n)(i, j));
          }
        }
      }
    }
  }
}

__device__ inline void rescaleAcc(
    Warp const& warp, WarpAcc& acc, UniformRescaleMask const& rescaleMask, ThrdRegRowMax const& rowScales) {
  static_assert(WarpAcc::rows * InstAcc::rows * 8 <= ThrdRegRowMax::size * warp_size);
// QuadRegRowMax const quadRowScales = replicateForQuad(warp, rowScales);
#pragma unroll
  for (uint32_t m = 0; m < WarpAcc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
      uint32_t const r = m * InstAcc::rows + i;  // in 8-row unit.
      bool const skip = enableMicroFastPath && ((rescaleMask[r / 4] & (0xFFU << 8 * r)) == 0);
      if (skip) {  // @fixme: do we need this?
        continue;
      }
      // float const scale = quadRowScales[r]; // @fixme: see if this is faster than the line below.
      float const scale = replicateValForQuad(warp, rowScales, r);
#pragma unroll
      for (uint32_t n = 0; n < WarpAcc::cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          acc(m, n)(i, j) *= scale;
        }
      }
    }
  }
}

__device__ inline void rescaleAcc(Warp const& warp, WarpAcc& acc, float scale) {
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < InstAcc::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < InstAcc::cols; j++) {
          acc(m, n)(i, j) *= scale;
        }
      }
    }
  }
}

template <bool useFp32Acc, uint32_t nbWarps, uint32_t nbTiles, uint32_t rows, uint32_t cols>
__device__ inline void smemFp16ArraySum(
    uint32_t idxWarp, Array2D<LdGrain, rows, cols>& dst, Array2D<LdGrain, rows, cols> const tiles[nbTiles]) {
  constexpr uint32_t nbThrds = warp_size * nbWarps;
  uint32_t const tid = warp_size * idxWarp + laneId();
  constexpr uint32_t nbGrains = SharedMem::XSmemBuffer::rows * SharedMem::XSmemBuffer::cols;
  constexpr uint32_t nbGrainsPerThrd = exactDiv(nbGrains, nbThrds);
  using AccType = mha::conditional_t<useFp32Acc, float2, InputElem2>;

#pragma unroll
  for (uint32_t i = 0; i < nbGrainsPerThrd; i++) {
    Vec<AccType, LdGrain::size> result;
    result.fill(AccType{0, 0});
    uint32_t const idx = nbThrds * i + tid;
#pragma unroll
    for (uint32_t j = 0; j < nbTiles; j++) {
      auto const data = reinterpret_cast<Vec<InputElem2, LdGrain::size> const(&)[nbGrains]>(tiles[j])[idx];
      if constexpr (useFp32Acc) {
#if INPUT_FP16
        result = addFloat2(result, __half22float2(data));
#else
        result = addFloat2(result, __bfloat1622float2(data));
#endif
      } else {
        result = __hadd2_rn(result, data);
      }
    }
    auto& dstGrain = reinterpret_cast<Vec<InputElem2, LdGrain::size>(&)[nbGrains]>(dst)[idx];
    if constexpr (useFp32Acc) {
#if INPUT_FP16
      dstGrain = __float22half2_rn(result);
#else
      PRAGMA_UNROLL_FP16_ONLY
      for (uint32_t k = 0; k < LdGrain::size; ++k) {
        dstGrain[k] = __floats2bfloat162_rn(result[k].x, result[k].y);
      }
#endif
    } else {
      dstGrain = result;
    }
  }
}

template <uint32_t nbBuffers>
__device__ inline ThrdRegRowMax mergeRowMax(
    Warp const& warp, TinyPtr<SMemWarpRowMax> const rowMaxBuffers, uint32_t nbSubSeqPerSeq) {
  ThrdRegRowMax regBuffers[nbBuffers];
  auto load = [&](uint32_t n) {
    assert(n < nbSubSeqPerSeq);
    regBuffers[n % nbBuffers] = rowMaxBuffers[n].loadToReg<false>(warp);
  };
#pragma unroll
  for (uint32_t i = 0; i < nbBuffers; i++) {
    if (i >= nbSubSeqPerSeq) {
      break;
    }
    load(i);
  }
  ThrdRegRowMax mergedRowMax = regBuffers[0];
  for (uint32_t n = 0; n < divUp(nbSubSeqPerSeq, nbBuffers); n++) {
#pragma unroll
    for (uint32_t i = 0; i < nbBuffers; i++) {
      uint32_t const idx = nbBuffers * n + i;
      if (idx >= nbSubSeqPerSeq) {
        break;
      }
      mergedRowMax = fmaxf(mergedRowMax, regBuffers[i]);
      uint32_t const idxNext = idx + nbBuffers;
      if (idxNext < nbSubSeqPerSeq) {
        load(idxNext);
      }
    }
  }
  return mergedRowMax;
}

__device__ inline void addAttentionSinks(
    ThrdRegRowMax& globalRowSum, ThrdRegRowMax const globalRowMax, float const* attentionSinks) {
  for (uint32_t i = 0; i < globalRowSum.size; i++) {
    uint32_t srcOffset = warp_size * i + laneId();
    if (srcOffset < headGrpSize) {
      globalRowSum[i] += expf(attentionSinks[srcOffset] - globalRowMax[i]);
    }
  }
}

#ifdef NDEBUG
__device__ __forceinline__
#else
CUBIN_EXPORT __global__
#endif
    void
    kernel_mha_impl(
#if SPEC_DEC
        uint32_t const qSeqLen, uint32_t const nbKHeads, uint32_t const headGrpSize,
        SeqLenDataType const* __restrict__ qCuSeqLens,  // [nbReq + 1]
#else
        uint32_t const nbKHeads,
#endif
#if SLIDING_WINDOW
        uint32_t slidingWinSize,
#endif
        float qScale,
        OutputHead* __restrict__ const output,  // [nbReq][beamWidth][nbQHeads]
#if LOW_PREC_OUTPUT
        float const* rcpOutScale,
#endif
        // NOTE: the input is actually Q buffer when integrated to TRT-LLM.
        IOHead const* __restrict__ const q,  // [nbReq][beamWidth][nbQHeads],
#if SPEC_DEC
        MaskType const* __restrict__ mask,  // [qSeqLen, divUp(qSeqLen, 32)].
#endif
        float const* attentionSinks,  // [headGrpSize]
#ifdef NDEBUG
        KVCacheList<usePagedKVCache> const& cacheList,
#if BEAM_WIDTH > 1
        BeamSearchParams const& beamSearchParams,
#endif
#else
        KVCacheList<usePagedKVCache> const cacheList,
#if BEAM_WIDTH > 1
        BeamSearchParams const beamSearchParams,
#endif
#endif
        uint32_t const batchSize,
        float const* __restrict__ kvCacheScale,  // Device memory scalar. Same scale for K and V cache. Used only for
                                                 // int8/fp8 KV cache.
        uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr) {
  assert(allowMultiBlockMode || gridDim.x == 1);
  bool const isMultiBlock = allowMultiBlockMode && (gridDim.x != 1);
  uint32_t const nbSubSeqPerSeq = allowMultiBlockMode ? gridDim.x : 1;
  uint32_t const idxSubSeqInSeq = allowMultiBlockMode ? blockIdx.x : 0;
  assert(!isMultiBlock || (semaphores != nullptr && scratch != nullptr));

  // gridDim: x - K/V sequence-dim split; y - number of K or V heads per token; z - number of requests
  assert(gridDim.z == batchSize && gridDim.y == nbKHeads);
  extern __shared__ char smemByteBuf[];
  SharedMem& smem = *reinterpret_cast<SharedMem*>(&smemByteBuf[0]);

  uint32_t const idxReq = blockIdx.z;
#if SPEC_DEC
  // Variable query sequence length support.
  bool const variableQSeqLen = qCuSeqLens != nullptr;
  uint32_t const actualQSeqLen = variableQSeqLen ? uint32_t(qCuSeqLens[idxReq + 1] - qCuSeqLens[idxReq]) : qSeqLen;
  // Same as idxReq * qSeqLen if all sequences all the same.
  // Take different beams as different requests/sequences currently.
  uint32_t const reqSeqOffset = variableQSeqLen ? uint32_t(qCuSeqLens[idxReq]) : (qSeqLen * idxReq);

  uint32_t const nbVHeads = nbKHeads;
  uint32_t const nbQHeads = nbKHeads * headGrpSize;
  uint32_t const nbQHeadTokens = nbQHeads * actualQSeqLen;
  uint32_t const nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;

  uint32_t const nbTokenBlocksPerGrp = gridDim.y / nbKHeads;
  uint32_t const idxHeadGrp = blockIdx.y / nbTokenBlocksPerGrp;  // inside one request
  uint32_t const idxHeadTokenInGrp = (blockIdx.y % nbTokenBlocksPerGrp) * warpTile.y;
  uint32_t const totalNbHeadTokensInGrp = actualQSeqLen * headGrpSize;
  uint32_t const nbValidHeadTokens = idxHeadTokenInGrp > totalNbHeadTokensInGrp
                                         ? 0u
                                         : mha::min(totalNbHeadTokensInGrp - idxHeadTokenInGrp, rowsPerBlock);
  // Shift the mask ptr by batch_idx.
  mask += reqSeqOffset * divUp(qSeqLen, 32u);
#else
  uint32_t const nbQHeads = nbKHeads * headGrpSize;

  uint32_t const idxHeadGrp = blockIdx.y;  // inside one request
#endif

  auto const ctaThrdId = threadIdx.x + warp_size * ctaShapeInWarps.x * (threadIdx.y + ctaShapeInWarps.y * threadIdx.z);
  assert(blockDim.x == ctaShapeInWarps.x * warp_size && blockDim.y == ctaShapeInWarps.y && blockDim.z == ctaShapeInWarps.z);
  auto const warp = this_warp();
  uint3 const warpIdx = getWarpIdx(warp);  // @fixme: use BoundedVal
  assert(warpIdx.x < ctaShapeInWarps.x && warpIdx.y < ctaShapeInWarps.y && warpIdx.z < ctaShapeInWarps.z);
  uint32_t const flatWarpIdPerRow = warpIdx.z * ctaShapeInWarps.x + warpIdx.x;  // per ctaShapeInWarps.y value
  unused(flatWarpIdPerRow);

  // initialize shared memory
  static_assert(persistentQ && ctaShapeInWarps.y == 1);
  if (ctaThrdId < ctaShapeInWarps.y) {
    init(&smem.qBarrier[ctaThrdId], warp_size * ctaShapeInWarps.x);  // be sure to use .noinc
  }
  constexpr uint32_t cacheVTileSeqStride = cacheVTileSeqLen * gemm1NbWarpGrps;
  constexpr uint32_t nbXTilesPerXIter = cacheVTileSeqStride < warpTile.x ? 1 : exactDiv(cacheVTileSeqStride, warpTile.x);
  constexpr uint32_t nbXItersPerCtaTile = exactDiv(ctaShapeInWarps.x, nbXTilesPerXIter);
  constexpr uint32_t nbVItersPerXIter = exactDiv(warpTile.x * nbXTilesPerXIter, cacheVTileSeqStride);
  constexpr uint32_t nbWarpGrpsPerXTile = mha::min(nbCacheVTilesPerXTile, gemm1NbWarpGrps);
  unused(nbWarpGrpsPerXTile);
  static_assert(warpTile.x >= cacheVTileSeqLen, "not implemented yet");
  static_assert(ctaSize >= uint32_t(sizeof(smem.xBarriers) / sizeof(CtaBarrierPair)));
  if (ctaThrdId < uint32_t(sizeof(smem.xBarriers) / sizeof(CtaBarrierPair))) {
    (&smem.xBarriers[0][0])[ctaThrdId].initialize(warp_size, warp_size * gemm1WarpsPerGrp * nbWarpGrpsPerXTile);
  }
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
  static_assert(ctaSize >= sizeof(smem.ctaRowMaxBwdBarriers) / sizeof(SharedMem::Barrier));
  if (ctaThrdId < sizeof(smem.ctaRowMaxBwdBarriers) / sizeof(SharedMem::Barrier)) {
    init(&smem.ctaRowMaxBwdBarriers[0][0] + ctaThrdId, warp_size);
  }
#endif
#if CTA_ROW_MAX_BACKWARD_METHOD != 0
  static_assert(ctaSize >= sizeof(smem.ctaRowMax) / sizeof(float));
  if (ctaThrdId < sizeof(smem.ctaRowMax) / sizeof(float)) {
    reinterpret_cast<float*>(&smem.ctaRowMax[0])[ctaThrdId] = SAFE_INIT_ROW_MAX;
  }
#endif
#if GRP_LOAD_V
  static_assert(ctaSize >= gemm1NbWarpGrps * nbVBuffers);
  if (ctaThrdId < gemm1NbWarpGrps * nbVBuffers) {
    init(smem.vBarrier(0, 0) + ctaThrdId, warp_size * gemm1WarpsPerGrp);
  }
  if (ctaThrdId < gemm1NbWarpGrps) {
    init(smem.warpGrpBar(ctaThrdId), warp_size * gemm1WarpsPerGrp);
  }
#endif
  __syncthreads();

#if ENABLE_PDL
  preExit();
  acqBulk();
#endif

  constexpr bool qkSwizzle = true;
  // load whole Q heads into shared memory
#if SPEC_DEC
  if (warpIdx.z == 0) {
    // map from idxQHead to idxHead in q input.
    auto const localQHeadTokenIdxMap = [nbQHeads, headGrpSize, reqSeqOffset, idxReq, idxHeadTokenInGrp](uint32_t idxHeadTokenLocal) -> uint32_t {
      assert(idxHeadTokenLocal < warpTile.y);  // may be larger than nbValidRows, then the output does not matter.
      if constexpr (beamWidth == 1) {
        idxHeadTokenLocal += idxHeadTokenInGrp;
        uint32_t const tokenIdx = (idxHeadTokenLocal / headGrpSize);
        uint32_t const headIdx = idxHeadTokenLocal % headGrpSize;
        return tokenIdx * nbQHeads + headIdx;
      }
    };
    static_assert(nbValidRows <= warpTile.y);
    auto const srcBase = q;
    uint32_t const idxHeadTokenBeg = nbQHeads * reqSeqOffset + (idxHeadGrp * headGrpSize);
    TinyPtr<IOHead const> const src{srcBase, idxHeadTokenBeg};

    bool const isFullTile = (nbValidHeadTokens == warpTile.y);
    static_assert(nbQBuffers == 1);
    if (isFullTile) {
      copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, true, warpTile.y>(
          warpIdx.x, smem.q[warpIdx.y][0], src, nbValidHeadTokens, localQHeadTokenIdxMap);
    } else {
      copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, false, warpTile.y>(
          warpIdx.x, smem.q[warpIdx.y][0], src, nbValidHeadTokens, localQHeadTokenIdxMap);
    }

    ldgsts::barArrive(smem.qBarrier[warpIdx.y], true);
  }
#else
  if (warpIdx.z == 0) {
    // map from idxQHead to idxHead in q input.
    auto const localQHeadIdxMap = [nbQHeads, idxReq, idxHeadGrp](uint32_t idxHeadLocal) -> uint32_t {
      assert(idxHeadLocal < warpTile.y);  // may be larger than nbValidRows, then the output does not matter.
      if constexpr (beamWidth == 1) {
        return idxHeadLocal;
      }
      uint32_t const idxBeam = idxHeadLocal / headGrpSize;
      uint32_t const result = idxHeadLocal + idxBeam * (nbQHeads - headGrpSize);
      uint32_t const idxQHeadInGrp = idxHeadLocal % headGrpSize;
      uint32_t const ref = nbQHeads * idxBeam + idxQHeadInGrp;
      assert(result == ref);
      unused(ref);
      return result;
    };
    static_assert(nbValidRows <= warpTile.y);
    auto const srcBase = q;
    // NOTE: read from Q buffer directly.
    uint32_t const idxHeadBeg = nbQHeads * beamWidth * idxReq + headGrpSize * idxHeadGrp;
    TinyPtr<IOHead const> const src{srcBase, idxHeadBeg};

    constexpr bool isFullTile = (nbValidRows == warpTile.y);
    static_assert(nbQBuffers == 1);
    copyHeadsAsync<PaddedInputHead, warpTile.y, ctaShapeInWarps.x, qkSwizzle, isFullTile, warpTile.y>(
        warpIdx.x, smem.q[warpIdx.y][0], src, nbValidRows, localQHeadIdxMap);
    ldgsts::barArrive(smem.qBarrier[warpIdx.y], true);
  }
#endif

  uint32_t const cacheSeqLen = getCacheSeqLen<usePagedKVCache>(cacheList, idxReq);
#if SLIDING_WINDOW && SPEC_DEC && !IS_SPEC_DEC_TREE
  uint32_t const tok0SeqLen = cacheSeqLen - actualQSeqLen + 1 + idxHeadTokenInGrp;  // ctaTokOffset;
  int32_t const tok0WinBeg = int32_t(tok0SeqLen) - int32_t(slidingWinSize);
  uint32_t const nbTotalSkipTokens = mha::max(0, tok0WinBeg);

#elif SLIDING_WINDOW
  bool const rtIsReallySliding = (cacheSeqLen > slidingWinSize);
  assert(!SPEC_DEC || !rtIsReallySliding);
  uint32_t const nbTotalSkipTokens = rtIsReallySliding ? cacheSeqLen - slidingWinSize : 0;
#else
  constexpr bool rtIsReallySliding = false;
  constexpr uint32_t nbTotalSkipTokens = 0;
#endif
  uint32_t const nbSkipLeadingTiles = nbTotalSkipTokens / ctaTile.x;
  uint32_t const tile0NbSkipTokens = nbTotalSkipTokens % ctaTile.x;
  unused(tile0NbSkipTokens);
#if USE_PAGED_KV_CACHE
  uint32_t const nbPages = divUp(cacheSeqLen, tokensPerPage);
  constexpr uint32_t nbPagesPerCtaTile = exactDiv(ctaTile.x, tokensPerPage);
#endif

  uint32_t const nbSeqIters = useKVCache ? divUp(cacheSeqLen, ctaTile.x) : 0;
#if SLIDING_WINDOW && SPEC_DEC && !IS_SPEC_DEC_TREE
  uint32_t const nbSeqItersWithoutMask = nbSkipLeadingTiles;
#elif SPEC_DEC
  uint32_t const nbSeqItersWithoutMask = (cacheSeqLen - actualQSeqLen) / ctaTile.x;
#endif

  uint32_t const seqStrideIters = nbSubSeqPerSeq;
  constexpr bool isKVCacheQuantized = (cacheElemSize < 2);
  uint32_t const seqIterInit = nbSkipLeadingTiles + idxSubSeqInSeq;
#if BEAM_WIDTH > 1
  uint32_t const nbCtxCtaTiles = beamSearchParams.ctxLenList[idxReq * beamWidth] / ctaTile.x;
#endif
  auto isConvergedTile = [&](uint32_t seqIter) {
#if BEAM_WIDTH == 1
    return true;
#else
    return seqIter < nbCtxCtaTiles;
#endif
  };
  if (warpIdx.z == 0) {
    float const qkScale = qScale * (isKVCacheQuantized ? kvCacheScale[0] : 1.f);  // qkScale is applied onto Q*K.T before softmax.
    CircIdx<nbKBuffers> idxCurrSMemKBuf{nbKBuffers - 1};
    auto const getSMemKTile = [&](uint32_t idx) -> SharedMem::KSmemBuffer& { return smem.k[warpIdx.x][idx]; };
#if BEAM_WIDTH > 1
    auto loadCacheIndir = [&](uint32_t seqIter, uint32_t idxBeam) mutable {
      auto& dst = smem.gemm0CacheIndir[warpIdx.x];
      uint32_t const offset = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
      loadIndicesForBeamSearchAsync<1, warpTile.x>(
          0, dst, beamSearchParams, idxReq, idxBeam, offset, cacheSeqLen);
    };
    loadCacheIndir(seqIterInit, 0U);
#endif
#if USE_PAGED_KV_CACHE
#if BEAM_WIDTH == 1
    KCachePageIndices pageIdx = KCachePageIndices::filled(kBAD_PAGE_INDEX);
#endif
    auto loadPages = [&](uint32_t idxPage) mutable {
#if BEAM_WIDTH == 1
      uint32_t const idxBeam = 0;
      pageIdx = getPage<KCachePageIndices::size>(cacheList, true, idxReq, idxBeam, idxPage, nbPages);
#else
      auto& dst = smem.kCachePages[warpIdx.x];
      loadPagesForBeamSearchAsync<1>(0U, dst, cacheList, true, idxReq, idxPage, nbPages);
#endif
    };
    uint32_t idxPageBeg = nbPagesPerCtaTile * seqIterInit + warpIdx.x * warpTile.x / tokensPerPage;
    loadPages(idxPageBeg);
#else
    constexpr uint32_t idxBeamBase = 0U;
    uint32_t const cacheKBaseBatch = cacheList.capacity * nbKHeads * (idxBeamBase + beamWidth * idxReq);
    uint32_t const cacheKSeqBaseOffset = cacheList.isBSNH
                                             ? (cacheKBaseBatch + idxHeadGrp)
                                             : (cacheKBaseBatch + cacheList.capacity * idxHeadGrp);
#endif
    auto loadKTilePart = [&](uint32_t seqIter, uint32_t idxBeam, uint32_t idxPart) mutable {
      assert(idxBeam < beamWidth);
      assert(seqIter % nbSubSeqPerSeq == seqIterInit % nbSubSeqPerSeq);
      auto const idxNextSMemKBuf = idxCurrSMemKBuf.next();
      auto& dst = getSMemKTile(idxNextSMemKBuf);
      uint32_t const dstHeadOffset = 0;
      uint32_t const seqOffset = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
      uint32_t const idxHeadBeg = (seqOffset % tokensPerPage) * nbKHeads + idxHeadGrp;

#else
      uint32_t const idxHeadBeg = tokensPerPage * idxHeadGrp + seqOffset % tokensPerPage;
#endif
#if BEAM_WIDTH == 1
#if PAGED_KV_CACHE_LAYOUT == 1
      HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> const src{
          cacheList.kCacheVLLM, pageIdx, nbKHeads, idxHeadBeg};
#else
      HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> const src{
          cacheList.pool, pageIdx, nbKHeads, idxHeadBeg};
#endif
#else
      IndexedHeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerWarpTile> const src{
          /*indices=*/smem.gemm0CacheIndir[warpIdx.x].data,
#if PAGED_KV_CACHE_LAYOUT == 1
          /*pool=*/cacheList.kCacheVLLM,
#else
          /*pool=*/cacheList.pool,
#endif
          /*pageIndices=*/smem.kCachePages[warpIdx.x].data,
          /*nbKHeads=*/nbKHeads,
          /*offset=*/idxHeadBeg};
#endif
#else
      uint32_t const idxHeadBeg = cacheList.isBSNH
                                      ? (cacheKSeqBaseOffset + seqOffset * nbKHeads)
                                      : (cacheKSeqBaseOffset + seqOffset);
#if BEAM_WIDTH == 1
      TinyPtr<GMemCacheHead const> const src{cacheList.kData, idxHeadBeg};
#else
      IndexedHeadPtr<GMemCacheHead const, 0, 0> const src{/*indices=*/smem.gemm0CacheIndir[warpIdx.x].data,
                                                          /*pointer=*/cacheList.data,
                                                          /*offset=*/idxHeadBeg,
                                                          /*beamStride=*/cacheList.capacity * nbKHeads * 2};
      // trap();
      // assert("not implemented");
#endif
#endif
      // if (threadIdx.x == dbgPrintTid) {
      //     printf("K: seqIter=%u, idxBeam=%u, idxPart=%u: pointers={%p, %p}, indices={", seqIter, idxBeam,
      //     idxPart, src.pointers[0], src.pointers[1]); uint32_t const nbHeadsAvail = mha::min((seqOffset <
      //     cacheSeqLen ? cacheSeqLen - seqOffset : 0U), warpTile.x); for (int i = 0; i < nbHeadsAvail; i++) {
      //         printf("%u, ", src.indices[i]);
      //     }
      //     printf("}\n");
      // }
      bool const isFullTile = (seqIter + 1 < nbSeqIters);
      if (isFullTile) {
        copyPartialHeadsAsync<PaddedCacheHead, warpTile.x, nbPartsPerCacheKHead, qkSwizzle, true>(
            warp, dst, dstHeadOffset, src, idxPart);
      } else {
        uint32_t const nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                               : 0U);  // may also be full but it can be handled correctly anyway
        copyPartialHeadsAsync<PaddedCacheHead, warpTile.x, nbPartsPerCacheKHead, qkSwizzle, false>(
            warp, dst, dstHeadOffset, src, idxPart, nbHeadsAvail);
      }
#if BEAM_WIDTH > 1
      // to make sure all threads has finished usage of cache indir and pages
      __syncwarp();
#endif
      if (idxPart + 1 == nbPartsPerCacheKHead) {
#if USE_PAGED_KV_CACHE
        bool const isForNextSeqIter = isConvergedTile(seqIter) || idxBeam == beamWidth - 1;
        if (isForNextSeqIter) {
          idxPageBeg += nbPagesPerCtaTile * nbSubSeqPerSeq;
          loadPages(idxPageBeg);
        }
#endif
#if BEAM_WIDTH > 1
        uint32_t idxBeamNext, seqIterDelta;
        mha::tie(idxBeamNext, seqIterDelta) = isConvergedTile(seqIter)
                                                  ? mha::tuple<uint32_t, uint32_t>(0U, 1U)
                                                  : carryLE<beamWidth>(idxBeam + 1, 0);  // optimize for context cache
        loadCacheIndir(seqIter + seqStrideIters * seqIterDelta, idxBeamNext);
#endif
      }
    };

#if BEAM_WIDTH > 1
    ldgsts::commitGroup();
    ldgsts::waitGroup<0>();
    __syncwarp();
#endif
    loadKTilePart(seqIterInit, 0, 0);
    ldgsts::commitGroup();  // @fixme: do prefetch for next iter tile if last part
    idxCurrSMemKBuf++;

    auto& xBar = smem.xBarriers[warpIdx.y][warpIdx.x];
    bool xBarConsumedParityNext = false;

    bool qBarParityNext = false;
    auto& qBar = smem.qBarrier[warpIdx.y];
    qBar.wait_parity(qBarParityNext);
    qBarParityNext = !qBarParityNext;
    constexpr bool reorderForKCache = (useKVCache && inputElemSize == 2 && cacheElemSize == 1);
    if constexpr (reorderForKCache) {
      reorder16bQHeadsToMatch8bKCache<ctaShapeInWarps.x, qkSwizzle, true>(warpIdx.x, smem.q[warpIdx.y][0]);
      unused(qBar.arrive());
      qBar.wait_parity(qBarParityNext);
      qBarParityNext = !qBarParityNext;
      assertWarpConverged();
    }
#if CTA_ROW_MAX_BACKWARD_METHOD == 2
    ThrdRegRowMax initRowMax;
    initRowMax.fill(safeInitRowMax);
#endif
    for (uint32_t seqIter = seqIterInit; seqIter < nbSeqIters; seqIter += seqStrideIters) {
#if SHORT_SEQ_OPT
      if (ctaTile.x * seqIter + warpTile.x * warpIdx.x >= cacheSeqLen) {
        break;
      }
#endif
      auto runGemm0 = [&](auto elemK, uint32_t idxBeam) {
        assert(idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth));
        using KElemType = mha::decay_t<decltype(elemK)>;
        constexpr uint32_t elemsPerKHeadPart = exactDiv(kHeadPartBytes, sizeof(KElemType));
        constexpr uint32_t nbPartsPerKHead = exactDiv(headElems, elemsPerKHeadPart);
        // the accumulator
        WarpAcc acc{};
        constexpr uint32_t nbUnroll = (cacheElemSize == 2 ? nbPartsPerKHead : 1);
#pragma unroll(nbUnroll)
        for (uint32_t p = 0; p < nbPartsPerKHead; p++) {
          constexpr bool syncKTileEarly = (beamWidth > 1);  // alternative is to use double buffer for cacheIndir and pages
          if constexpr (syncKTileEarly) {
            // synchronize gemm0CacheIndir for the next loadKTilePart. the last loaded K tile is also
            // sync'ed at the same time.
            ldgsts::waitGroup<0>();
            __syncwarp();
          }
          // prefetch next part into shared memory
          uint32_t idxPartNext, idxBeamNext, nNextBias;
          mha::tie(idxPartNext, idxBeamNext, nNextBias) = isConvergedTile(seqIter)
                                                              ? carryLE<nbPartsPerKHead, 1U>(p + 1, idxBeam, 0U)
                                                              : carryLE<nbPartsPerKHead, beamWidth>(p + 1, idxBeam, 0U);

          loadKTilePart(seqIter + seqStrideIters * nNextBias, idxBeamNext, idxPartNext);
          ldgsts::commitGroup();
          // @fixme: do L2 cache prefetch for next iter tile if last part

          // q is already synchronized
          if constexpr (!syncKTileEarly) {
            // synchronize k
            ldgsts::waitGroup<1>();
          }
          SharedMem::QSmemBuffer const& smemQ = smem.q[warpIdx.y][0];
          constexpr uint32_t qOffsetPerPart = exactDiv(elemsPerKHeadPart, inputElemsPerGrain);
          uint32_t const smemQOffset = qOffsetPerPart * p;
          SharedMem::KSmemBuffer const& smemKPart = getSMemKTile(idxCurrSMemKBuf);
          // #ifndef NDEGBUG
          //                     for (uint32_t i = 0; i < exactDiv(smemKPart.rows * smemKPart.cols,
          //                     warp_size); i++) {
          //                         uint32_t const idx = warp_size * i + laneId();
          //                         uint32_t const r = idx / smemKPart.cols;
          //                         uint32_t const c = idx % smemKPart.cols;

          //                         assert(smemKPart(r, c) == );
          //                     }
          // #endif
          // do computation.
          smemQKPartGemm<KElemType>(warp, acc, smemQ, smemQOffset, smemKPart);
          idxCurrSMemKBuf++;
        }
        return acc;
      };
      WarpAcc acc;
      //@fixme: alternative is to use separate inner loop, which results in larger but maybe faster code.
      for (uint32_t idxBeam = 0; idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth); idxBeam++) {
        WarpAcc tmp;
        if constexpr (mha::is_same_v<CacheElem, InputElem>) {
          tmp = runGemm0(CacheElem{}, idxBeam);
        } else {
          tmp = runGemm0(CacheElem{}, idxBeam);
        }
        pickAccRowsForBeamSearch(
            warp, acc, tmp, isConvergedTile(seqIter), idxBeam, [](float& d, float s) { d = s; });
      }
      // apply qkScale
      rescaleAcc(warp, acc, qkScale);
#if CTA_ROW_MAX_BACKWARD_METHOD == 0
      QuadRegRowMax initRowMaxQuad;
      initRowMaxQuad.fill(safeInitRowMax);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 1
      // load hint
      xBar.consumed.wait_parity(getAndFlip(xBarConsumedParityNext));
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToRegForQuad<false>(warp);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 2
      QuadRegRowMax initRowMaxQuad = replicateForQuad(warp, initRowMax);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 3
      // load hint
      smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].wait_parity(xBarConsumedParityNext);
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToRegForQuad<false>(warp);
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
      // load hint
      QuadRegRowMax initRowMaxQuad = smem.ctaRowMax[warpIdx.y].loadToRegForQuad<true>(warp);
#endif
      // masking
      uint32_t const warpTileTokenBeg = ctaTile.x * seqIter + warpTile.x * warpIdx.x;
#if SPEC_DEC
      if (seqIter >= nbSeqItersWithoutMask) {
        uint32_t const nbValidCols = (warpTileTokenBeg < cacheSeqLen ? cacheSeqLen - warpTileTokenBeg : 0U);
        applyMaskFromInput(warp, acc, mask, idxHeadTokenInGrp, nbValidCols, qSeqLen, actualQSeqLen, headGrpSize
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                           ,
                           tok0WinBeg, seqIter, cacheSeqLen, warpTileTokenBeg
#endif
        );
      }
#else
      bool const isFirstIter = (seqIter == nbSkipLeadingTiles);
      bool const needMaskLeading = (rtIsReallySliding && isFirstIter);
      bool const isLastIter = (seqIter + 1 == nbSeqIters);
      bool const needMaskTrailing = isLastIter && cacheSeqLen % ctaTile.x != 0;
      if (needMaskLeading || needMaskTrailing) {
        uint32_t const validTokenBeg = (!needMaskLeading || nbTotalSkipTokens < warpTileTokenBeg)
                                           ? 0
                                           : nbTotalSkipTokens - warpTileTokenBeg;
        uint32_t const validTokenEnd = (warpTileTokenBeg < cacheSeqLen ? cacheSeqLen - warpTileTokenBeg : 0U);
        if (validTokenBeg > 0 || validTokenEnd < warpTile.x) {
          applyMask(warp, acc, validTokenBeg, validTokenEnd);
        }
      }
#endif

      // find max and update acc into exp(acc-max).
      QuadRegRowMax const regRowMax = warpTileOnlineSoftmax(warp, initRowMaxQuad, acc);

      // store result and max to shared memory.
      GemmOutRegTile const fp16Acc = toFp16(acc);
      QuadRegRowMax const regRowSum = computeRowSum(warp, fp16Acc);
#if CTA_ROW_MAX_BACKWARD_METHOD != 1
      xBar.consumed.wait_parity(getAndFlip(xBarConsumedParityNext));
#if CTA_ROW_MAX_BACKWARD_METHOD == 2
      initRowMax = smem.ctaRowMax[warpIdx.y][warpIdx.x].loadToReg<false>(warp);
#endif
#endif
      storeOrderedGemmOutTile(warp, smem.x[warpIdx.y][warpIdx.x], fp16Acc);
      smem.warpRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, regRowMax);
      smem.warpRowSum[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, regRowSum);
      unused(xBar.produced.arrive());
    }
  } else {
    assert(warpIdx.z == 1);
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
    unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
#endif
    uint32_t const warpIdxInGrp = gemm1WarpIdxInGrp(warpIdx.x);  // @fixme: use BoundedVal
    uint32_t const warpGrpIdx = gemm1WarpGrpIdx(warpIdx.x);      // @fixme: use BoundedVal
    auto* const pWarpGrpBar = smem.warpGrpBar(warpGrpIdx);
    ParityOrNone<grpLoadV> warpGrpBarParityNext{};
#if BEAM_WIDTH > 1
    auto loadCacheIndir = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter, uint32_t idxBeam) mutable {
      uint32_t const seqOffset = ctaTile.x * seqIter + warpTile.x * nbXTilesPerXIter * xIter + cacheVTileSeqStride * vIter + cacheVTileSeqLen * warpGrpIdx;
      auto& dst = smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x];
      loadIndicesForBeamSearchAsync<grpLoadV ? gemm1WarpsPerGrp : 1U, cacheVTileSeqLen>(
          grpLoadV ? warpIdxInGrp : 0U, dst, beamSearchParams, idxReq, idxBeam, seqOffset, cacheSeqLen);
    };
    loadCacheIndir(seqIterInit, 0, 0, 0);
#endif
    unused(smem.xBarriers[warpIdx.y][warpIdx.x].consumed.arrive(gemm1WarpsPerGrp * nbWarpGrpsPerXTile));
    CircIdx<nbVBuffers> idxCurrSMemVBuf{nbVBuffers - 1};
    auto const getSmemVTile = [&](uint32_t idx) -> SharedMem::VSmemBuffer& { return smem.v[warpGrpIdx][grpLoadV ? 0 : warpIdxInGrp][idx]; };
    auto const getSmemVBar = [&](uint32_t idx) -> SharedMem::Barrier* { return smem.vBarrier(warpGrpIdx, idx); };
#if USE_PAGED_KV_CACHE
#if BEAM_WIDTH == 1
    VCachePageIndices pageIdx = VCachePageIndices::filled(kBAD_PAGE_INDEX);
#endif
    auto loadPages = [&](uint32_t idxPageBeg) mutable {
#if BEAM_WIDTH == 1
      uint32_t const idxBeam = 0;
      pageIdx = getPage<VCachePageIndices::size>(cacheList, false, idxReq, idxBeam, idxPageBeg, nbPages);
#else
      auto& dst = smem.vCachePages[grpLoadV ? warpGrpIdx : warpIdx.x];
      loadPagesForBeamSearchAsync<grpLoadV ? gemm1WarpsPerGrp : 1U>(
          grpLoadV ? warpIdxInGrp : 0U, dst, cacheList, false, idxReq, idxPageBeg, nbPages);
#endif
    };
    uint32_t idxPageBeg = nbPagesPerCtaTile * seqIterInit + cacheVTileSeqLen * warpGrpIdx / tokensPerPage;
    loadPages(idxPageBeg);
#else
    uint32_t const idxBeamBase = 0;
    uint32_t const cacheVBaseBatch = cacheList.capacity * nbKHeads * (idxBeamBase + beamWidth * idxReq);
    uint32_t const cacheVSeqBaseOffset = cacheList.isBSNH
                                             ? (cacheVBaseBatch + idxHeadGrp)
                                             : (cacheVBaseBatch + cacheList.capacity * idxHeadGrp);
#endif
    auto nextStep = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter, uint32_t idxBeam) {
      uint32_t vIterNext, isNextBeam;
      mha::tie(vIterNext, isNextBeam) = carryLE<nbVItersPerXIter>(vIter + 1, 0);

      uint32_t idxBeamNext, xIterNext, nNextBias;
      mha::tie(idxBeamNext, xIterNext, nNextBias) = isConvergedTile(seqIter)
                                                        ? carryLE<1, nbXItersPerCtaTile>(idxBeam + isNextBeam, xIter, 0)
                                                        : carryLE<beamWidth, nbXItersPerCtaTile>(idxBeam + isNextBeam, xIter, 0);

      uint32_t const seqIterNext = seqIter + seqStrideIters * nNextBias;
      return mha::tuple<uint32_t, uint32_t, uint32_t, uint32_t>(seqIterNext, xIterNext, vIterNext, idxBeamNext);
    };
    auto loadVTilePart = [&](uint32_t seqIter, uint32_t xIter, uint32_t vIter,
                             uint32_t idxBeam) mutable {  // @fixme: merge three iteration parameters into idxVTileGlb.
      assert(idxBeam < beamWidth);
      assert(seqIter % nbSubSeqPerSeq == seqIterInit % nbSubSeqPerSeq);
      auto const idxNextSMemVBuf = idxCurrSMemVBuf.next();
      auto& dst = getSmemVTile(idxNextSMemVBuf);
      uint32_t const dstHeadOffset = 0;
      constexpr bool vSwizzle = true;

      uint32_t const seqOffset = ctaTile.x * seqIter + warpTile.x * nbXTilesPerXIter * xIter + cacheVTileSeqStride * vIter + cacheVTileSeqLen * warpGrpIdx;
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
      uint32_t const idxHeadBeg = (seqOffset % tokensPerPage) * nbKHeads + idxHeadGrp;

#else
      uint32_t const idxHeadBeg = tokensPerPage * idxHeadGrp + seqOffset % tokensPerPage;
#endif
#if BEAM_WIDTH == 1
#if PAGED_KV_CACHE_LAYOUT == 1
      HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> const src{
          cacheList.vCacheVLLM, pageIdx, nbKHeads, idxHeadBeg};
#else
      HeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> const src{
          cacheList.pool, pageIdx, nbKHeads, idxHeadBeg};
#endif
#else
      IndexedHeadPtr<GMemCacheHead const, tokensPerPage, nbPagesPerVTile> const src{
          /*indices=*/smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x].data,
#if PAGED_KV_CACHE_LAYOUT == 1
          /*pool=*/cacheList.vCacheVLLM,
#else
          /*pool=*/cacheList.pool,
#endif
          /*pageIndices=*/smem.vCachePages[grpLoadV ? warpGrpIdx : warpIdx.x].data,
          /*nbKHeads=*/nbKHeads,
          /*offset=*/idxHeadBeg};
#endif
#else
      uint32_t const idxHeadBeg = cacheList.isBSNH
                                      ? (cacheVSeqBaseOffset + seqOffset * nbKHeads)
                                      : (cacheVSeqBaseOffset + seqOffset);
#if BEAM_WIDTH == 1
      TinyPtr<GMemCacheHead const> const src{cacheList.vData, idxHeadBeg};
#else
      IndexedHeadPtr<GMemCacheHead const, 0, 0> const src{
          /*indices=*/smem.gemm1CacheIndir[grpLoadV ? warpGrpIdx : warpIdx.x].data,
          /*pointer=*/cacheList.data,
          /*offset=*/idxHeadBeg,
          /*beamStride=*/cacheList.capacity * nbKHeads * 2};
#endif
#endif
      // if (threadIdx.x == dbgPrintTid) {
      //     printf("V: seqIter=%u, xIter=%u, idxBeam=%u, vIter=%u: pointers={%p, %p}, indices={", seqIter, xIter,
      //     idxBeam, vIter, src.pointers[0], src.pointers[1]); uint32_t const nbHeadsAvail = mha::min((seqOffset
      //     < cacheSeqLen ? cacheSeqLen - seqOffset : 0U), cacheVTileSeqLen); for (int i = 0; i < nbHeadsAvail;
      //     i++) {
      //         printf("%u, ", src.indices[i]);
      //     }
      //     printf("}\n");
      // }

#if GRP_LOAD_V
      uint32_t const nbHeadsAvail = (seqIter + 1 < nbSeqIters)
                                        ? cacheVTileSeqLen
                                        : (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                                   : 0U);  // may also be full but it can be handled correctly anyway
      copyHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, false>(
          warpIdxInGrp, dst, src, nbHeadsAvail);
#else
      uint32_t const nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                             : 0U);  // may also be full but it can be handled correctly anyway
      unused(nbHeadsAvail);
      bool const isFullTile = (seqIter + 1 < nbSeqIters);
      if (isFullTile) {
        copyPartialHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, true>(
            warp, dst, dstHeadOffset, src, warpIdxInGrp);
      } else {
        uint32_t const nbHeadsAvail = (seqOffset < cacheSeqLen ? cacheSeqLen - seqOffset
                                                               : 0U);  // may also be full but it can be handled correctly anyway
        copyPartialHeadsAsync<PaddedCacheHead, cacheVTileSeqLen, gemm1WarpsPerGrp, vSwizzle, false>(
            warp, dst, dstHeadOffset, src, warpIdxInGrp, mha::min(nbHeadsAvail, cacheVTileSeqLen));
      }
#endif

#if BEAM_WIDTH > 1
      // to make sure all threads has finished usage of cache indir and pages
      unused(arrive<grpLoadV>(pWarpGrpBar));
      wait_parity<grpLoadV>(pWarpGrpBar, getAndFlip<grpLoadV>(warpGrpBarParityNext));
#endif
#if USE_PAGED_KV_CACHE
      constexpr uint32_t xIterSeqStride = cacheVTileSeqStride * nbVItersPerXIter;
      if constexpr (xIterSeqStride <= tokensPerPage) {
        uint32_t const nbXItersPerPage = exactDiv(tokensPerPage, xIterSeqStride);
        assert(nbXItersPerPage <= nbXItersPerCtaTile);
        if (xIter % nbXItersPerPage == nbXItersPerPage - 1 && vIter == nbVItersPerXIter - 1 && (idxBeam == beamWidth - 1 || isConvergedTile(seqIter))) {
          auto const step = 1;  // cacheVTileSeqLen * gemm1NbWarpGrps / tokensPerPage;
          idxPageBeg += (idxPageBeg % nbPagesPerCtaTile == nbPagesPerCtaTile - 1
                             ? nbPagesPerCtaTile * (nbSubSeqPerSeq - 1) + step
                             : step);
          assert(beamWidth == 1 || cacheVTileSeqStride <= tokensPerPage && "todo: need to substrate from idxPageBeg for beam switching");
          loadPages(idxPageBeg);
        }
      } else {
        assert(nbVItersPerXIter == 1);
        if ((idxBeam == beamWidth - 1 || isConvergedTile(seqIter)) && vIter == nbVItersPerXIter - 1) {
          auto const step = exactDiv(xIterSeqStride, tokensPerPage);
          idxPageBeg += (idxPageBeg % nbPagesPerCtaTile + step >= nbPagesPerCtaTile
                             ? nbPagesPerCtaTile * (nbSubSeqPerSeq - 1) + step
                             : step);
          loadPages(idxPageBeg);
        }
      }
#endif
#if BEAM_WIDTH > 1
      uint32_t seqIterNext, xIterNext, vIterNext, idxBeamNext;
      mha::tie(seqIterNext, xIterNext, vIterNext, idxBeamNext) = nextStep(seqIter, xIter, vIter, idxBeam);
      loadCacheIndir(seqIterNext, xIterNext, vIterNext, idxBeamNext);
#endif
    };
    auto commitVTileLoad = [&](uint32_t idxVBar) {
#if GRP_LOAD_V
      auto& bar = *getSmemVBar(idxVBar);
      ldgsts::barArrive(bar, true);
#else
      ldgsts::commitGroup();
#endif
    };
    auto syncVTileLoad = [&](uint32_t idxVBar, ParityOrNone<grpLoadV> parity, bool alreadyComplete) {
#if GRP_LOAD_V
      if (alreadyComplete) {
        return;
      }
      SharedMem::Barrier& bar = *getSmemVBar(idxVBar);
      bar.wait_parity(parity);
#else
      assert(!alreadyComplete);
      ldgsts::waitGroup<nbVBuffers - 1>();
#endif
    };
    auto testVTileLoad = [&](uint32_t idxVBar, ParityOrNone<grpLoadV> parity) { return test_wait_parity<grpLoadV>(getSmemVBar(idxVBar), parity); };

#if BEAM_WIDTH > 1
    // synchronize first page/cacheIndir loading to shared memory
    ldgsts::commitGroup();
    ldgsts::waitGroup<0>();
    unused(arrive<grpLoadV>(pWarpGrpBar));
    wait_parity<grpLoadV>(pWarpGrpBar, getAndFlip<grpLoadV>(warpGrpBarParityNext));
#endif

    loadVTilePart(seqIterInit, 0, 0, 0);
    commitVTileLoad(idxCurrSMemVBuf.next());
    idxCurrSMemVBuf++;
    ParityOrNone<grpLoadV> vBarParity{};
    // @fixme: do prefetch for next iter tile if last part

    ThrdRegRowMax globalRowMax;
    globalRowMax.fill(SAFE_INIT_ROW_MAX);
    ThrdRegRowMax globalRowSum;
    globalRowSum.fill(0);
    // the accumulator
    WarpAcc acc{};
    if (grpLoadV) {
      unused(pWarpGrpBar->arrive());
    }
    bool xBarProducedParityNext = false;
    for (uint32_t seqIter = seqIterInit; seqIter < nbSeqIters; seqIter += seqStrideIters) {
#pragma unroll
      for (uint32_t xIter = 0; xIter < nbXItersPerCtaTile; xIter++) {
        uint32_t const idxXTile = xIter * nbXTilesPerXIter + warpGrpIdx / nbCacheVTilesPerXTile;
        assert(idxXTile < ctaShapeInWarps.x);
#if SHORT_SEQ_OPT
        if (ctaTile.x * seqIter + warpTile.x * idxXTile >= cacheSeqLen) {
          break;
        }
#endif
        auto const& smemXTile = smem.x[warpIdx.y][idxXTile];
        auto& xBar = smem.xBarriers[warpIdx.y][idxXTile];
        ThrdRegRowMax xRowScales;
        UniformRescaleMask xRowNeedRescaleMask;  // expect storage in UR
        bool skipXRowRescale;
        for (uint32_t idxBeam = 0; idxBeam < (isConvergedTile(seqIter) ? 1U : beamWidth); idxBeam++) {
#pragma unroll
          for (uint32_t vIter = 0; vIter < nbVItersPerXIter; vIter++) {
            bool const vTestConsumed = test_wait_parity<grpLoadV>(pWarpGrpBar, warpGrpBarParityNext);
            constexpr bool syncVTileEarly = (beamWidth > 1);  // alternative is to use double buffer for cacheIndir and pages
            bool vTestProduced = syncVTileEarly && testVTileLoad(idxCurrSMemVBuf, vBarParity);
            auto isLastVBuf = [&] { return (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1); };
            unused(isLastVBuf);
            uint32_t const idxVTileInsideXIter = gemm1NbWarpGrps * vIter + warpGrpIdx;
            uint32_t const idxVTile = idxVTileInsideXIter % nbCacheVTilesPerXTile;  // inside XTile.
            assert(idxVTile < nbCacheVTilesPerXTile);
            uint32_t nNext, xIterNext, vIterNext, idxBeamNext;
            mha::tie(nNext, xIterNext, vIterNext, idxBeamNext) = nextStep(seqIter, xIter, vIter, idxBeam);
            if constexpr (syncVTileEarly) {
              // sync early to make sure that cacheIndir and pages has been loaded. The last loaded V tile
              // is also sync'ed at the same time.
              syncVTileLoad(idxCurrSMemVBuf, vBarParity, vTestProduced);
              if (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1) {
                flip<grpLoadV>(vBarParity);
              }
            }
            if (!vTestConsumed) {
              wait_parity<grpLoadV>(pWarpGrpBar, warpGrpBarParityNext);
            }
            flip<grpLoadV>(warpGrpBarParityNext);
            loadVTilePart(nNext, xIterNext, vIterNext, idxBeamNext);
            commitVTileLoad(idxCurrSMemVBuf.next());
            // @fixme: do L2 cache prefetch for next iter tile

            if constexpr (!syncVTileEarly) {
              vTestProduced = testVTileLoad(idxCurrSMemVBuf, vBarParity);
            }

            if (idxBeam == 0 && vIter == 0) {
              xBar.produced.wait_parity(xBarProducedParityNext);
              auto const& smemRowMax = smem.warpRowMax[warpIdx.y][idxXTile];
              auto const& smemRowSum = smem.warpRowSum[warpIdx.y][idxXTile];
              // update globalRowMax
              ThrdRegRowMax xTileRowMax;
              ThrdRegRowMax xTileRowSum;
              UniformRescaleMask needRescaleMask;
#pragma unroll
              for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
                xTileRowMax[i] = smemRowMax[warp_size * i + laneId()];
                xTileRowSum[i] = smemRowSum[warp_size * i + laneId()];
                assert(__ballot_sync(~0U, laneId() == 0) == 1U);
                assert(__ballot_sync(~0U, laneId() == 0) == 1U);
                needRescaleMask[i] = __ballot_sync(~0U, xTileRowMax[i] != globalRowMax[i]);
              }
              bool const skipAllRescale = !any(needRescaleMask);
              if (skipAllRescale) {
                skipXRowRescale = true;
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
                if (idxXTile == warpIdx.x) {
                  unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
                }
#endif
              } else {
                ThrdRegRowMax const globalRowMaxOld = globalRowMax;
                UniformRescaleMask accRowNeedRescaleMask;
#pragma unroll
                for (uint32_t i = 0; i < ThrdRegRowMax::size; i++) {
                  accRowNeedRescaleMask[i] = __ballot_sync(~0U, xTileRowMax[i] > globalRowMaxOld[i]);
                  xRowNeedRescaleMask[i] = (needRescaleMask[i] & ~accRowNeedRescaleMask[i]);
                  assert(xRowNeedRescaleMask[i] == __ballot_sync(~0U, xTileRowMax[i] < globalRowMaxOld[i]));
                  globalRowMax[i] = fmaxf(globalRowMaxOld[i], xTileRowMax[i]);
                }
                skipXRowRescale = !any(xRowNeedRescaleMask);

#if CTA_ROW_MAX_BACKWARD_METHOD == 1 || CTA_ROW_MAX_BACKWARD_METHOD == 2 || CTA_ROW_MAX_BACKWARD_METHOD == 3
                // update smem.ctaRowMax.
                if (idxXTile == warpIdx.x) {
                  smem.ctaRowMax[warpIdx.y][warpIdx.x].storeFromReg<false>(warp, globalRowMax);
#if CTA_ROW_MAX_BACKWARD_METHOD == 3
                  unused(smem.ctaRowMaxBwdBarriers[warpIdx.y][warpIdx.x].arrive());
#endif
                }
#elif CTA_ROW_MAX_BACKWARD_METHOD == 4
                // update smem.ctaRowMax.
                // smem.ctaRowMax[warpIdx.y].storeFromReg<true>(warp, globalRowMax);
                smem.ctaRowMax[warpIdx.y].atomicMaxUpdate(warp, globalRowMax);
#endif
                // update row sum and acc
                if (!enableMicroFastPath || any(accRowNeedRescaleMask)) {
                  ThrdRegRowMax const accRowScales = expf(globalRowMaxOld - globalRowMax);
                  globalRowSum = globalRowSum * accRowScales;
                  // @fixme: when tmpAcc is used, this can be delayed.
                  rescaleAcc(warp, acc, accRowNeedRescaleMask, accRowScales);
                }
                if (!enableMicroFastPath || !skipXRowRescale) {
                  xRowScales = skipXRowRescale ? xRowScales : expf(xTileRowMax - globalRowMax);
                  xTileRowSum = skipXRowRescale ? xTileRowSum : xTileRowSum * xRowScales;
                }
              }
              globalRowSum = globalRowSum + xTileRowSum;
            }
            if constexpr (!syncVTileEarly) {
              syncVTileLoad(idxCurrSMemVBuf, vBarParity, vTestProduced);
              if (idxCurrSMemVBuf == idxCurrSMemVBuf.nbBuffers - 1) {
                flip<grpLoadV>(vBarParity);
              }
            }
            auto const& smemVTile = getSmemVTile(idxCurrSMemVBuf);
            // do computation from shared memory X and V tiles
#if BEAM_WIDTH == 1
            smemXVPartGemm<CacheElem>(warp, acc, skipXRowRescale, xRowNeedRescaleMask, xRowScales,
                                      smemXTile, idxVTile, smemVTile, grpLoadV ? warpIdxInGrp : 0);
#else
            WarpAcc tmpAcc{};
            smemXVPartGemm<CacheElem>(warp, tmpAcc, skipXRowRescale, xRowNeedRescaleMask, xRowScales,
                                      smemXTile, idxVTile, smemVTile, grpLoadV ? warpIdxInGrp : 0);
            pickAccRowsForBeamSearch(
                warp, acc, tmpAcc, isConvergedTile(seqIter), idxBeam, [](float& d, float s) { d += s; });
#endif
            if (grpLoadV) {
              unused(pWarpGrpBar->arrive());
            }
            idxCurrSMemVBuf++;
          }
        }  // idxBeam
        xBar.consumed.arrive();
      }  // xIter
      flip(xBarProducedParityNext);
    }  // seqIter

    auto const fullRescaleMask = UniformRescaleMask::filled(~0U);

    constexpr bool needMergeGlobal = (gemm1NbWarpGrps > 1 && nbXTilesPerXIter > 1);
    if constexpr (needMergeGlobal) {
      assert(gemm1NbWarpGrps != 1);
      __syncthreads();
      smem.warpRowMax[warpIdx.y][warpIdx.x].template storeFromReg<false>(warp, globalRowMax);
      smem.warpRowSum[warpIdx.y][warpIdx.x].template storeFromReg<false>(warp, globalRowSum);
      __syncthreads();
      for (uint32_t i = 1; i < nbXTilesPerXIter; i++) {  // i = 0 is for self and we can skip
        static_assert(nbXTilesPerXIter * nbWarpGrpsPerXTile == gemm1NbWarpGrps);
        uint32_t const otherWarpGrpIdx = (warpGrpIdx + nbWarpGrpsPerXTile * i) % gemm1NbWarpGrps;
        uint32_t const otherWarpIdx = warpIdxInGrp + gemm1WarpsPerGrp * otherWarpGrpIdx;
#ifndef NDEBUG
        {
          auto const v1 = smem.warpRowMax[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
          auto const v2 = smem.warpRowMax[warpIdx.y][otherWarpIdx - warpIdxInGrp].template loadToReg<false>(warp);
#pragma unroll
          for (uint32_t k = 0; k < ThrdRegRowMax::size; k++) {
            assert(__float_as_int(v1[k]) == __float_as_int(v2[k]));
          }
        }
#endif
        auto const otherRowMax = smem.warpRowMax[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
        auto const otherRowSum = smem.warpRowSum[warpIdx.y][otherWarpIdx].template loadToReg<false>(warp);
        auto const globalRowMaxNew = fmaxf(globalRowMax, otherRowMax);
        auto const scaleForThis = expf(globalRowMax - globalRowMaxNew);
        auto const scaleForOther = expf(otherRowMax - globalRowMaxNew);
        rescaleAcc(warp, acc, fullRescaleMask, scaleForThis);
        globalRowSum = globalRowSum * scaleForThis + otherRowSum * scaleForOther;
        globalRowMax = globalRowMaxNew;
      }
    }

    float voScale = (isKVCacheQuantized ? kvCacheScale[0] : 1.F);
    if (seqIterInit < nbSeqIters) {  // otherwise rcpRowSum will be NAN.
      // The attention sinks are moved to the multi-block reduction part if the multi-block is enabled.
      if (!isMultiBlock && attentionSinks != nullptr) {
        // Attention sinks are per head.
        addAttentionSinks(globalRowSum, globalRowMax, attentionSinks + headGrpSize * idxHeadGrp);
      }
      ThrdRegRowMax const rcpRowSum = __frcp_rn(globalRowSum);
#if LOW_PREC_OUTPUT
      voScale *= rcpOutScale[0];
#endif
      rescaleAcc(warp, acc, fullRescaleMask, rcpRowSum * ThrdRegRowMax::filled(voScale));
    }
    GemmOutRegTile const outTile = toFp16(acc);

    auto mergeAndSaveOutTile = [&](GemmOutRegTile const& tile, bool reorder) {
      if constexpr (gemm1NbWarpGrps == 1) {
        // swizzle in shared memory and write output global memory
        auto& outSwizzleBuffer = smem.x[warpIdx.y][warpIdx.x];
        __syncthreads();
        storeGemmOutTile(warp, outSwizzleBuffer, tile, reorder);
        __syncwarp();
        return &outSwizzleBuffer;
      } else {
        __syncthreads();
        // store to shared memory, then merge groups.
        using PostProcSMem = SharedMem::XSmemBuffer[ctaShapeInWarps.y][gemm1WarpsPerGrp][gemm1NbWarpGrps];
        static_assert(sizeof(PostProcSMem) <= smemSize);
        SharedMem::XSmemBuffer(&postSMem)[gemm1NbWarpGrps] = reinterpret_cast<PostProcSMem&>(smem)[warpIdx.y][warpIdxInGrp];
        storeGemmOutTile(warp, postSMem[warpGrpIdx], tile, reorder);
        __syncthreads();
        smemFp16ArraySum<true, gemm1NbWarpGrps, gemm1NbWarpGrps>(warpGrpIdx, postSMem[0], postSMem);
        __syncthreads();
        return &postSMem[0];
      }
    };

    // merge results from different warp groups
    SharedMem::XSmemBuffer* smemOutTile = mergeAndSaveOutTile(outTile, inputElemSize == 2 && cacheElemSize == 1);
    if (isMultiBlock) {
      static_assert(ctaShapeInWarps.y == 1, "not implemented");
#if SPEC_DEC
      // Includes both kHeads and qTokens.
      uint32_t const nbIndepHeadTokens = gridDim.y;
      uint32_t const indepHeadTokenIdx = blockIdx.y;
      uint32_t const nbSeq = nbIndepHeadTokens * batchSize;
#else
      uint32_t const nbSeq = nbKHeads * batchSize;
#endif
      uint32_t const nbSubSeq = nbSubSeqPerSeq * nbSeq;
      MemSegmenter<false> segmenter{scratch};

#if SPEC_DEC
      uint32_t const idxSeq = nbIndepHeadTokens * idxReq + indepHeadTokenIdx;
#else
      uint32_t const idxSeq = nbKHeads * idxReq + idxHeadGrp;
#endif
      uint32_t const idxBufBase = nbSubSeqPerSeq * idxSeq;
      uint32_t const idxBuf = idxBufBase + idxSubSeqInSeq;
      // copy row max/sum
      TinyPtr<SMemWarpRowMax> const rowMaxBuffers = segmenter.newSeg<SMemWarpRowMax>(nbSubSeq);
      TinyPtr<SMemWarpRowMax> const rowSumBuffers = segmenter.newSeg<SMemWarpRowMax>(nbSubSeq);
      if (warpGrpIdx == 0 && warpIdxInGrp == 0) {
        rowMaxBuffers[idxBuf].storeFromReg<false>(warp, globalRowMax);
        rowSumBuffers[idxBuf].storeFromReg<false>(warp, globalRowSum);
      }
      using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
      TinyPtr<Vec<ScratchBuf, gemm1WarpsPerGrp>> const scratchBuffers = segmenter.newSeg<Vec<ScratchBuf, gemm1WarpsPerGrp>>(nbSubSeq);
      // copy output to scratch
      copyGrains<false, nbValidRows * ScratchBuf::cols, gemm1NbWarpGrps>(
          warpGrpIdx, &scratchBuffers[idxBuf][warpIdxInGrp](0, 0), &(*smemOutTile)(0, 0));
      __syncthreads();
      constexpr uint32_t nbTileBuffers = 2;

      struct MultiBlockSMem {
        bool isLastCta;

        struct MBBuf {
          SMemWarpRowMax rowMax;
          SMemWarpRowMax rowSum;
          SharedMem::XSmemBuffer tiles[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax tileRowMax[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax tileRowSums[gemm1NbWarpGrps][gemm1WarpsPerGrp][nbTileBuffers];
          SMemWarpRowMax mergedRowSum[gemm1NbWarpGrps];
        };

        MBBuf storage[ctaShapeInWarps.y];
      };

      static_assert(sizeof(MultiBlockSMem) <= smemSize);
      MultiBlockSMem& mbsmem = reinterpret_cast<MultiBlockSMem&>(smem);
      // increase the semaphore by 1
      if (warpIdx.y == 0 && warpGrpIdx == 0 && warpIdxInGrp == 0 && laneId() == 0) {
        uint32_t old;
        uint32_t const lastOld = nbSubSeqPerSeq - 1;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                     : "=r"(old)
                     : "l"(&semaphores[idxSeq]), "r"(lastOld));
        assert(old < nbSubSeqPerSeq);
        mbsmem.isLastCta = (old == lastOld);
      }
      __syncthreads();

      // merge if we are the last CTA.
      bool const isLastCta = mbsmem.isLastCta;
      if (isLastCta) {
        MultiBlockSMem::MBBuf& mbbuf = mbsmem.storage[warpIdx.y];
        SMemWarpRowMax& smemRowMax = reinterpret_cast<SMemWarpRowMax&>(smem);
        // get row max.
        if (warpIdx.x == 0) {
          ThrdRegRowMax const mergedRowMax = mergeRowMax<8>(warp, rowMaxBuffers + idxBufBase, nbSubSeqPerSeq);
          smemRowMax.storeFromReg<false>(warp, mergedRowMax);
        }
        __syncthreads();
        ThrdRegRowMax const mergedRowMax = smemRowMax.loadToReg<false>(warp);

        // rescale and accumulate
        auto getTileBuf = [&](auto& buffers, uint32_t d) -> decltype(buffers[0][0][0])& { return buffers[warpGrpIdx][warpIdxInGrp][d]; };
        auto loadBufAsync = [&](uint32_t n) {
          uint32_t const d = n / gemm1NbWarpGrps % nbTileBuffers;
          SharedMem::XSmemBuffer& dstTile = getTileBuf(mbbuf.tiles, d);
          SMemWarpRowMax& dstRowSum = getTileBuf(mbbuf.tileRowSums, d);
          SMemWarpRowMax& dstRowMax = getTileBuf(mbbuf.tileRowMax, d);
          copyGrains<true, sizeof(ScratchBuf) / grainBytes, 1, true>(
              0, &dstTile(0, 0), &scratchBuffers[idxBufBase + n][warpIdxInGrp](0, 0));
          constexpr uint32_t nbGrainsPerRowMaxBuf = exactDiv(sizeof(SMemWarpRowMax), grainBytes);
          copyGrains<true, roundUp(nbGrainsPerRowMaxBuf, 32u), 1, nbGrainsPerRowMaxBuf % 32 == 0>(0,
                                                                                                  reinterpret_cast<LdGrain*>(&dstRowSum),
                                                                                                  reinterpret_cast<LdGrain const*>(&rowSumBuffers[idxBufBase + n]), nbGrainsPerRowMaxBuf);
          copyGrains<true, roundUp(nbGrainsPerRowMaxBuf, 32u), 1, nbGrainsPerRowMaxBuf % 32 == 0>(0,
                                                                                                  reinterpret_cast<LdGrain*>(&dstRowMax),
                                                                                                  reinterpret_cast<LdGrain const*>(&rowMaxBuffers[idxBufBase + n]), nbGrainsPerRowMaxBuf);
        };
        loadBufAsync(warpGrpIdx);
        ldgsts::commitGroup();
        WarpAcc sumAcc{};
        ThrdRegRowMax partialMergedRowSum{};
        for (uint32_t n = warpGrpIdx; n < nbSubSeqPerSeq; n += gemm1NbWarpGrps) {
          if (n + gemm1NbWarpGrps < nbSubSeqPerSeq) {
            loadBufAsync(n + gemm1NbWarpGrps);
          }
          ldgsts::commitGroup();
          ldgsts::waitGroup<1>();
          uint32_t const d = n / gemm1NbWarpGrps % nbTileBuffers;
          WarpAcc tile = toWarpAcc(loadGemmOutTile(warp, mbbuf.tiles[warpGrpIdx][warpIdxInGrp][d]));
          ThrdRegRowMax const tileRowMax = getTileBuf(mbbuf.tileRowMax, d).loadToReg<false>(warp);
          ThrdRegRowMax const tileRowSum = getTileBuf(mbbuf.tileRowSums, d).loadToReg<false>(warp);
          ThrdRegRowMax const tileRowScales = expf(tileRowMax - mergedRowMax);
          ThrdRegRowMax const scaledTileRowSum = tileRowSum * tileRowScales;
          partialMergedRowSum = partialMergedRowSum + scaledTileRowSum;
          assert(isfinite(partialMergedRowSum[0]));
          rescaleAcc(warp, tile, fullRescaleMask, scaledTileRowSum);
          sumAcc = sumAcc + tile;
        }

        ThrdRegRowMax mergedRowSum{};
        if (gemm1NbWarpGrps == 1) {
          mergedRowSum = partialMergedRowSum;
        } else {
          if (warpIdxInGrp == 0) {
            mbbuf.mergedRowSum[warpGrpIdx].storeFromReg<false>(warp, partialMergedRowSum);
          }
          __syncthreads();
#ifndef NDEBUG
#pragma unroll
          for (uint32_t k = 0; k < ThrdRegRowMax::size; k++) {
            assert(__float_as_int(mbbuf.mergedRowSum[warpGrpIdx].loadToReg<false>(warp)[k]) == __float_as_int(partialMergedRowSum[k]));
          }
          __syncthreads();
#endif
#pragma unroll
          for (uint32_t i = 0; i < gemm1NbWarpGrps; i++) {
            mergedRowSum = mergedRowSum + mbbuf.mergedRowSum[i].loadToReg<false>(warp);
            assert(isfinite(mergedRowSum[0]));
          }
        }
        if (attentionSinks != nullptr) {
          // Attention sinks are per head.
          addAttentionSinks(mergedRowSum, mergedRowMax, attentionSinks + headGrpSize * idxHeadGrp);
        }
        __syncthreads();
        rescaleAcc(warp, sumAcc, fullRescaleMask, __frcp_rn(mergedRowSum));
        GemmOutRegTile const mergedOutTile = toFp16(sumAcc);
        smemOutTile = mergeAndSaveOutTile(mergedOutTile, false);
      }
    }
    if (warpGrpIdx == 0) {
#if SPEC_DEC
      copyOutputToGlobalMem(warp, &output[reqSeqOffset * nbQHeads], nbQHeads, headGrpSize,
                            (idxHeadGrp * headGrpSize), nbValidHeadTokens,
                            uint2{warpTile.x * warpIdxInGrp, nbValidRows * warpIdx.y + idxHeadTokenInGrp}, *smemOutTile);
#else
      copyOutputToGlobalMem(warp, &output[nbQHeads * beamWidth * idxReq], nbQHeads, idxHeadGrp,
                            uint2{warpTile.x * warpIdxInGrp, nbValidRows * warpIdx.y}, *smemOutTile);
#endif
    }
  }
}

#if SPEC_DEC
#if __CUDA_ARCH__ == 900 && M_TILESIZE == 16
constexpr uint32_t nbCtaPerSM = 2;
#else
constexpr uint32_t nbCtaPerSM = 1;
#endif
#else
#if __CUDA_ARCH__ == 900
constexpr uint32_t nbCtaPerSM = 2;
#else
constexpr uint32_t nbCtaPerSM = 1;
#endif
#endif

[[maybe_unused]] CUBIN_EXPORT __device__ constexpr XQAKernelType kernelType = XQAKernelType::kAMPERE_WARP_SPECIALIZED;

#ifdef NDEBUG
CUBIN_EXPORT __global__ __launch_bounds__(256, nbCtaPerSM) void kernel_mha(
#if SPEC_DEC
    uint32_t const qSeqLen, uint32_t const nbKHeads, uint32_t const headGrpSize, SeqLenDataType const* qCuSeqLens,
#else
    uint32_t const nbKHeads,
#endif
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale,
    OutputHead* __restrict__ const output,  // [nbReq][beamWidth][nbQHeads]
#if LOW_PREC_OUTPUT
    float const* rcpOutScale,
#endif
    IOHead const* __restrict__ const q,  // [nbReq][beamWidth][nbQHeads],
#if SPEC_DEC
    MaskType const* __restrict__ mask,  // [qSeqLen, divUp(qSeqLen, 32))] uint2 (each bit represents mask for one col
                                        // position).
#endif
    float const* attentionSinks,  // [headGrpSize]
    KVCacheList<usePagedKVCache> const cacheList,
#if BEAM_WIDTH > 1
    BeamSearchParams const beamSearchParams,
#endif
    uint32_t const batchSize,
    float const* __restrict__ kvCacheScale,  // Device memory scalar. Same scale for K and V cache. Used only for
                                             // int8/fp8 KV cache.
    uint32_t* __restrict__ semaphores = nullptr, void* __restrict__ scratch = nullptr) {
#if SPEC_DEC
  kernel_mha_impl(qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
  kernel_mha_impl(nbKHeads,
#endif
#if SLIDING_WINDOW
                  slidingWinSize,
#endif
                  qScale, output,
#if LOW_PREC_OUTPUT
                  rcpOutScale,
#endif
                  q,
#if SPEC_DEC
                  mask,
#endif
                  attentionSinks, cacheList,
#if BEAM_WIDTH > 1
                  beamSearchParams,
#endif
                  batchSize, kvCacheScale, semaphores, scratch);
}
#else
static constexpr auto kernel_mha = kernel_mha_impl;
#endif

#ifndef GENERATE_CUBIN
uint32_t computeNbSubSeqPerSeqMHA(cudaDeviceProp const& prop, uint32_t batchSize, uint32_t nbKHeads, uint32_t maxSeqLen) {
  if (!allowMultiBlockMode) {
    return 1;
  }
  auto const env = std::getenv("XQA_NB_SUB_SEQ");
  if (env != nullptr) {
    int32_t const val = std::stoi(env);
    if (val > 0) {
      return val;
    }
  }
  return std::min<uint32_t>(
      std::max<uint32_t>(1U, prop.multiProcessorCount / (batchSize * nbKHeads)), divUp(maxSeqLen, ctaTile.x));
}

void launchMHA(cudaDeviceProp const& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
               uint32_t slidingWinSize,
#endif
               float qScale, OutputHead* output,
#if LOW_PREC_OUTPUT
               float const* rcpOutScale,
#endif
#if USE_INPUT_KV
               InputHead const* qkv,
#if ROPE_STYLE != 0
               Vec<float, validElemsPerHead> const* ropeCosSin,
#endif
#else
               InputHead const* q,
#endif
               float const* attentionSinks,  // [headGrpSize]
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
               GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#else
               GMemCacheHead* pool,  // global pool of pages
#endif
               KVCachePageIndex const*
                   kvCachePageList,  // device pointer. shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
#else
               GMemKVCacheHead* kCacheData,
               GMemKVCacheHead* vCacheData,
               bool isBSNH,
#endif
               uint32_t maxSeqLen, uint32_t const* seqLen,
#if BEAM_WIDTH > 1
               BeamSearchParams const& beamSearchParams,
#endif
               uint32_t batchSize,
               float const* __restrict__ kvCacheScale,  // Device memory scalar. Same scale for K and V cache. Used only for
                                                        // int8/fp8 KV cache.
#if SPEC_DEC
               SpecDecParams const& specDecParams,
#endif
#if SKIP_SOFTMAX_ATTN
               float const skipSoftmaxThresholdScaleFactor,  // for compatibility with mha_sm90.cu only
#if SKIP_SOFTMAX_ATTN_BLOCK_STATS
               uint32_t* __restrict__ skippedBlockCount,  // for compatibility with mha_sm90.cu only
               uint32_t* __restrict__ totalBlockCount,    // for compatibility with mha_sm90.cu only
#endif
#endif
               uint32_t* semaphores, void* scratch, cudaStream_t stream) {
#if SPEC_DEC
  auto const qSeqLen = specDecParams.qSeqLen;
  auto const qCuSeqLens = specDecParams.qCuSeqLens;
  auto const mask = specDecParams.mask;
#endif
#if USE_INPUT_KV
  throw std::runtime_error("not implemented");
#else
  static uint32_t const hostSmemSize = [&]() {
    uint32_t size;
    checkCuda(cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)));
    checkCuda(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
    return size;
  }();
  uint32_t const nbVHeads = nbKHeads;
  unused(nbVHeads);
  uint32_t const nbQHeads = nbKHeads * headGrpSize;
  unused(nbQHeads);

  // const uint32_t nbSubSeqPerSeq = allowMultiBlockMode ? DBG_NB_CTAS_PER_SEQ : 1;
  uint32_t const nbSubSeqPerSeq = computeNbSubSeqPerSeqMHA(prop, batchSize, nbKHeads, maxSeqLen);
  // printf("DEBUG: launchMHA: batch=%u, nbKHeads=%u, maxSeq=%u, nbSubSeqPerSeq=%u\n", batchSize, nbKHeads, maxSeqLen, nbSubSeqPerSeq);
  // gridDim.z == batchSize && gridDim.y == nbKHeads && gridDim.x == nbSubSeqPerSeq
#if SPEC_DEC
  const uint32_t nbTokenBlocksPerGrp = divUp(qSeqLen * headGrpSize, rowsPerBlock);
  dim3 const dimGrid{nbSubSeqPerSeq, nbKHeads * nbTokenBlocksPerGrp, batchSize};
#else
  dim3 const dimGrid{nbSubSeqPerSeq, nbKHeads, batchSize};
#endif
  dim3 const dimCta{warp_size * ctaShapeInWarps.x, ctaShapeInWarps.y, ctaShapeInWarps.z};
#if defined(NDEBUG) || USE_PAGED_KV_CACHE
  auto const launchCfg = makeLaunchConfig(dimGrid, dimCta, hostSmemSize, stream, ENABLE_PDL != 0);
#endif
#if USE_PAGED_KV_CACHE
  uint32_t const maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
#if PAGED_KV_CACHE_LAYOUT == 1
  KVCacheList<true> const cacheList{kCacheVLLM, vCacheVLLM, kvCachePageList, seqLen, maxNbPagesPerSeq};
#else
  KVCacheList<true> const cacheList{pool, kvCachePageList, seqLen, maxNbPagesPerSeq};
#endif
  cudaLaunchKernelEx(&launchCfg, kernel_mha,
#if SPEC_DEC
                     qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
                     nbKHeads,
#endif
#if SLIDING_WINDOW
                     slidingWinSize,
#endif
                     qScale, output,
#if LOW_PREC_OUTPUT
                     rcpOutScale,
#endif
                     q,
#if SPEC_DEC
                     mask,
#endif
                     attentionSinks, cacheList,
#if BEAM_WIDTH > 1
                     beamSearchParams,
#endif
                     batchSize, kvCacheScale, semaphores, scratch);
#else
  KVCacheList<false> const cacheList{kCacheData, vCacheData, seqLen, maxSeqLen, isBSNH, 1};
#ifndef NDEBUG
  kernel_mha<<<dimGrid, dimCta, hostSmemSize, stream>>>(
#else
  cudaLaunchKernelEx(&launchCfg, kernel_mha,
#endif
#if SPEC_DEC
      qSeqLen, nbKHeads, headGrpSize, qCuSeqLens,
#else
                     nbKHeads,
#endif
#if SLIDING_WINDOW
      slidingWinSize,
#endif
      qScale, output,
#if LOW_PREC_OUTPUT
      rcpOutScale,
#endif
      q,
#if SPEC_DEC
      mask,
#endif
      attentionSinks, cacheList,
#if BEAM_WIDTH > 1
      beamSearchParams,
#endif
      batchSize, kvCacheScale, semaphores, scratch);
#endif
  checkCuda(cudaPeekAtLastError());
#endif  // USE_INPUT_KV
}
#endif
#endif

__device__ __host__ inline size_t GetScratchSize(uint32_t nbSeq, uint32_t nbSubSeqPerSeq) {
  uint32_t const nbSubSeq = nbSubSeqPerSeq * nbSeq;
  size_t offset = 0;

  // 1. rowMax
  offset = roundUp<size_t>(offset, sizeof(SMemWarpRowMax));
  offset += sizeof(SMemWarpRowMax) * nbSubSeq;

  // 2. rowSum
  offset = roundUp<size_t>(offset, sizeof(SMemWarpRowMax));
  offset += sizeof(SMemWarpRowMax) * nbSubSeq;

  // 3. scratchBuffers
  using ScratchBuf = Array2D<LdGrain, nbValidRows, SharedMem::XSmemBuffer::cols>;
  using VecT = Vec<ScratchBuf, gemm1WarpsPerGrp>;

  // size_t sem_size = roundUp<size_t>(nbSeq * sizeof(uint32_t), 128);
  // if (nbSubSeqPerSeq > 1) {
  //   printf("[MHA_IMPL] GetScratchSize: nbSeq=%u, nbSubSeqPerSeq=%u, sizeof(SMemWarpRowMax)=%zu, sizeof(VecT)=%zu, nbValidRows=%u, XS_cols=%u\n",
  //           nbSeq, nbSubSeqPerSeq, (size_t)sizeof(SMemWarpRowMax), (size_t)sizeof(VecT), (uint32_t)nbValidRows, (uint32_t)SharedMem::XSmemBuffer::cols);
  // }

  offset = roundUp<size_t>(offset, sizeof(VecT));
  offset += sizeof(VecT) * nbSubSeq;

  return offset;
}
