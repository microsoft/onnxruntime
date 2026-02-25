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
#include "cuda_hint.cuh"
#include "mha_stdheaders.cuh"
#include "utils.cuh"
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace gmma {

enum class SwizzleMode : uint64_t {
  kNONE = 0,
  k128 = 1,
  k64 = 2,
  k32 = 3
};

struct MatDesc {
  uint64_t addr : 16;
  uint64_t dimKOffset : 16;
  uint64_t dimMNOffset : 16;
  uint64_t pad0 : 1;
  uint64_t baseOffset : 3;
  uint64_t pad1 : 10;
  SwizzleMode swizzle : 2;

  enum class Raw : uint64_t {
  };

  [[nodiscard]] __device__ inline MatDesc withAddr(void const* data) const {
    MatDesc ret = *this;
    ret.addr = encode(__cvta_generic_to_shared(data));
    return ret;
  }

  static __device__ inline uint32_t encode(uint32_t val) {
    return (val & 0x3FFFFU) >> 4;
  }

  __device__ inline bool operator==(MatDesc const& other) const {
    return raw() == other.raw();
  }

  __device__ inline Raw const& raw() const {
    static_assert(sizeof(MatDesc) == 8);
    return reinterpret_cast<Raw const&>(*this);
  }

  static __device__ inline MatDesc fromRaw(Raw const& raw) {
    return reinterpret_cast<MatDesc const&>(raw);
  }
};

static_assert(sizeof(MatDesc) == 8);

[[nodiscard]] __device__ inline MatDesc::Raw addAddr(MatDesc::Raw base, void const* data) {
  assert((uint32_t(__cvta_generic_to_shared(data)) & ~0x3FFFFU) == 0);
  MatDesc::Raw ret = base;
  auto& u32x2 = reinterpret_cast<uint32_t (&)[2]>(ret);
  u32x2[0] += static_cast<uint32_t>(__cvta_generic_to_shared(data)) >> 4;
  return ret;
}

__device__ inline MatDesc makeMatDesc(void const* data, uint32_t dimKByteOffset, uint32_t dimMNByteOffset,
                                      void const* patternStartAddr, SwizzleMode swizzleMode) {
  uint32_t const patternAddr = __cvta_generic_to_shared(patternStartAddr);
  uint32_t const baseAlign = [&]() -> uint32_t {
    switch (swizzleMode) {
      case SwizzleMode::kNONE:
        return 1;
      case SwizzleMode::k128:
        return 1024;
      case SwizzleMode::k64:
        return 512;
      case SwizzleMode::k32:
        return 256;
    }
    asm volatile("trap;\n");
    return 0;
  }();
  assert(__cvta_generic_to_shared(data) % baseAlign == 0);
  uint32_t const baseOffset = ((patternAddr % baseAlign == 0) ? 0U : ((patternAddr >> 0x7) & 0x7));
  return MatDesc{
      /*addr=*/MatDesc::encode(__cvta_generic_to_shared(data)),
      /*dimKOffset=*/MatDesc::encode(dimKByteOffset),
      /*dimMNOffset=*/MatDesc::encode(dimMNByteOffset),
      /*pad0=*/0,
      /*baseOffset=*/baseOffset,
      /*pad1=*/0,
      /*swizzle=*/swizzleMode,
  };
}

__device__ inline MatDesc makeMatDesc(
    void const* data, uint32_t dimKByteOffset, uint32_t dimMNByteOffset, SwizzleMode swizzleMode) {
  return makeMatDesc(data, dimKByteOffset, dimMNByteOffset, data, swizzleMode);
}

inline constexpr uint32_t instM = 64;

template <typename MathElem>
inline constexpr uint32_t instK = 32 / sizeof(MathElem);

inline constexpr uint32_t instNBase = 8;

// for both a and b, outer-dim is gemm-K and inner-dim is gemm-M or gemm-N
// acc is used as both input and output.
template <typename InputElem, uint32_t n, bool transA = false, bool transB = false>
__device__ void mma_async_shmA(
    float (&acc)[exactDiv(n, instNBase)][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal);
template <typename InputElem, uint32_t n, bool transA = false, bool transB = false>
__device__ void mma_async_regA(
    float (&acc)[exactDiv(n, instNBase)][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal);

__device__ inline void fence() {
  asm volatile("wgmma.fence.sync.aligned;\n");
}

__device__ inline void commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n");
}

template <uint32_t targetNbInFlightGroups>
__device__ inline void wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0\n; " ::"n"(targetNbInFlightGroups));
}

template <bool swizzle, typename T, uint32_t rows, uint32_t cols, bool alignedForSwizzle>
constexpr SwizzleMode getSwizzleMode(Array2D<T, rows, cols, alignedForSwizzle> const&) {
  constexpr auto rowBytes = Array2D<T, rows, cols, alignedForSwizzle>::rowBytes;
  if constexpr (!swizzle) {
    return SwizzleMode::kNONE;
  }
  if constexpr (rowBytes % 128 == 0) {
    return SwizzleMode::k128;
  } else if constexpr (rowBytes == 64) {
    return SwizzleMode::k64;
  } else {
    static_assert(rowBytes == 32);
    return SwizzleMode::k32;
  }
}
}  // namespace gmma

#include "gmma_impl.cuh"
