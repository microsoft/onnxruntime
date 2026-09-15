/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mma.cuh"
#include "utils.cuh"

using InstAcc = Array2D<float, 2, 2>;

template <uint32_t m, uint32_t n>
using WarpAccT = Array2D<InstAcc, exactDiv(m, 16), exactDiv(n, 8)>;

template <uint32_t accRows, uint32_t accCols>
__device__ inline void applyMask(
    const Warp& warp, Array2D<InstAcc, accRows, accCols>& acc, uint32_t validColBeg, uint32_t validColEnd) {
  const uint32_t idxInQuad = laneId() % 4;
  const uint32_t idxQuad = laneId() / 4;
#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::cols; j++) {
      const uint32_t col = 8 * n + InstAcc::cols * idxInQuad + j;
      if (col >= validColBeg && col < validColEnd) {
        continue;
      }
#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++) {
          acc(m, n)(i, j) = mha::numeric_limits<float>::lowest();
        }
      }
    }
  }
}

template <uint32_t tileM>
using QuadRegRowMaxT = Vec<float, divUp(tileM, warp_size) * 4>;  // data is replicated across 4 threads in a MMA quad.
template <uint32_t tileM>
using ThrdRegRowMaxT = Vec<float, divUp(tileM, warp_size)>;  // unlike QuadRegRowMax, not replicated.
template <uint32_t tileM>
using UniformRescaleMaskT = Vec<uint32_t, divUp(tileM, warp_size)>;  // uniform and stored in UR
inline constexpr uint32_t quadPerWarp = warp_size / 4;

// idxMat8 is the reduced row index in 8-row unit.
template <uint32_t n>
__device__ inline float replicateValForQuad(const Warp& warp, const Vec<float, n>& src, uint32_t idxMat8) {
  assertWarpConverged();
  const uint32_t i = idxMat8 / 4;
  const uint32_t j = idxMat8 % 4;
  return __shfl_sync(~0U, src[i], quadPerWarp * j + laneId() / 4);
}

template <uint32_t n>
__device__ inline QuadRegRowMaxT<n * warp_size> replicateForQuad(const Warp& warp, const Vec<float, n>& src) {
  assertWarpConverged();
  QuadRegRowMaxT<n * warp_size> dst{};
#pragma unroll
  for (uint32_t i = 0; i < src.size; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
      dst[i * 4 + j] = __shfl_sync(~0U, src[i], quadPerWarp * j + laneId() / 4);
      assert(__float_as_int(dst[i * 4 + j]) == __float_as_int(replicateValForQuad(warp, src, i * 4 + j)));
    }
  }
  return dst;
}

template <uint32_t n>
__device__ inline ThrdRegRowMaxT<warp_size * exactDiv(n, 4)> dedupFromQuad(const Warp& warp, const Vec<float, n>& src) {
#ifndef NDEBUG
  for (uint32_t i = 0; i < src.size; i++) {
    assert(__float_as_int(src[i]) == __float_as_int(__shfl_sync(~0U, src[i], laneId() / 4 * 4)));
  }
#endif
  ThrdRegRowMaxT<warp_size * exactDiv(n, 4)> dst{};
  const uint32_t lane = laneId();
  const uint32_t idxMat = lane / 8;
  const uint32_t idxRow = lane % 8;
#pragma unroll
  for (uint32_t i = 0; i < dst.size; i++) {
#pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
      const float val = __shfl_sync(~0U, src[i * 4 + j], 4 * idxRow);
      if (idxMat == j) {
        dst[i] = val;
      }
    }
  }
#ifndef NDEBUG  // refcheck
  QuadRegRowMaxT<warp_size * exactDiv(n, 4)> rep = replicateForQuad(warp, dst);
#pragma unroll
  for (uint32_t i = 0; i < n; i++) {
    assert(__float_as_int(src[i]) == __float_as_int(rep[i]));
    __syncwarp();
  }
#endif
  return dst;
}

template <uint32_t tileM, uint32_t tileN>
__device__ inline ThrdRegRowMaxT<tileM> computeRowSumF8(
    const Warp& warp, const Array2D<Array2D<uint32_t, 2, 1>, exactDiv(tileM, 16), exactDiv(tileN, 16)>& src) {
  using WarpAcc = WarpAccT<tileM, 8>;
  WarpAcc acc{};
  const Vec<__nv_fp8x2_e4m3, 2> bWord = {__nv_fp8x2_e4m3{float2{1, 1}}, __nv_fp8x2_e4m3{float2{1, 1}}};
  const uint32_t b[2][1] = {reinterpret_cast<const uint32_t&>(bWord), reinterpret_cast<const uint32_t&>(bWord)};
#pragma unroll
  for (uint32_t i = 0; i < WarpAcc::rows; i++) {
#pragma unroll
    for (uint32_t k = 0; k < exactDiv(src.cols, 2); k++) {
      mma<__nv_fp8_e4m3>(reinterpret_cast<float (&)[2][2]>(acc(i, 0)),
                         reinterpret_cast<const uint32_t (&)[2][2]>(src(i, k * 2)), b);
    }
  }
  QuadRegRowMaxT<tileM> rowSum;
  for (uint32_t i = 0; i < WarpAcc::rows; i++) {
    for (uint32_t m = 0; m < InstAcc::rows; m++) {
#ifndef NDEBUG
      assert(__float_as_int(acc(i, 0)(m, 0)) == __float_as_int(acc(i, 0)(m, 1)));
      assert(__float_as_int(acc(i, 0)(m, 0)) == __float_as_int(__shfl_sync(~0U, acc(i, 0)(m, 0), laneId() / 4 * 4)));
#endif
      rowSum[i * InstAcc::rows + m] = acc(i, 0)(m, 0);
    }
  }
  return dedupFromQuad(warp, rowSum);
}

template <uint32_t tileM, uint32_t tileN>
__device__ inline ThrdRegRowMaxT<tileM> computeRowSumF32(const Warp& warp, const WarpAccT<tileM, tileN>& src) {
  QuadRegRowMaxT<tileM> rowSum{};
#pragma unroll
  for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < InstAcc::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < src.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++) {
          if (n == 0 && j == 0) {
            rowSum[m * InstAcc::rows + i] = src(m, n)(i, j);
          } else {
            rowSum[m * InstAcc::rows + i] += src(m, n)(i, j);
          }
        }
      }
    }
  }
  const uint32_t lane = laneId();
#pragma unroll
  for (uint32_t mask = 2; mask != 0; mask /= 2) {
#pragma unroll
    for (uint32_t i = 0; i < rowSum.size; i++) {
      rowSum[i] += __shfl_xor_sync(~0U, rowSum[i], mask);
    }
  }
  return dedupFromQuad(warp, rowSum);
}
