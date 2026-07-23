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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/interleaved_numeric_conversion.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

template <KernelType KT>
struct kernel_type_traits;
#define KERNEL_TYPE_TRAITS_REGISTRY(KT, _isGroupwise, _isInt4) \
  template <>                                                  \
  struct kernel_type_traits<KT> {                              \
    static constexpr bool isGroupwise = _isGroupwise;          \
    static constexpr bool isInt4 = _isInt4;                    \
  };

KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int8Groupwise, true, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int4Groupwise, true, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int8PerChannel, false, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int4PerChannel, false, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int8Groupwise, true, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int4Groupwise, true, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int8PerChannel, false, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int4PerChannel, false, true);
#undef KERNEL_TYPE_TRAITS_REGISTRY

// A generic memory iterator used for coalesced global memory access with optional enablement.
// Template parameters:
//   Enable: If false, disables loading/storing.
//   TVec: Vectorized type (e.g., float4, half2).
//   Strided: Number of rows in a tile.
//   Continuous: Number of contiguous vector elements to load/store at once.
//   Scalar type (e.g., half).
template <bool Enable, typename TVec, int Strided, int Continuous, typename T>
class GMemIterator {
 public:
  __device__ __forceinline__ GMemIterator(T* addr, int offset, int step, int stride)
      : addr_(Enable ? (addr + offset) : nullptr), step_(step), stride_(stride) {
  }

  __device__ __forceinline__ void load(void* dst, int iter, int ii = 0) {
    if constexpr (Enable) {
#pragma unroll
      for (int jj = 0; jj < Continuous; ++jj) {
        reinterpret_cast<TVec*>(dst)[jj] = reinterpret_cast<TVec*>(addr_ + iter * step_ + ii * stride_)[jj];
      }
    }
  }

 private:
  T* addr_;
  int step_;
  int stride_;
};

struct FP16DetailsA {
  using Type = half;
  using Type2 = half2;
  static constexpr int kElemBits = 16;
};

struct BF16DetailsA {
  using Type = __nv_bfloat16;
  using Type2 = __nv_bfloat162;
  static constexpr int kElemBits = 16;
};

struct Int8DetailsW {
  static constexpr int kElemBits = 8;
};

struct Int4DetailsW {
  static constexpr int kElemBits = 4;
};

struct Fp4DetailsW {
  static constexpr int kElemBits = 4;
};

template <typename TypeDetailsW>
struct IsFp4Weight : std::false_type {};
template <>
struct IsFp4Weight<Fp4DetailsW> : std::true_type {};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajor {
  using DetailsA = TypeDetailsA;
  using DetailsW = TypeDetailsW;
  using AccessTypeA = float4;
  using AccessTypeW = int;
  static constexpr int kAccessSize = 128;
  static constexpr int kStepK = kAccessSize / TypeDetailsA::kElemBits;
  static constexpr int kTileSize = TileSizeK;
  static constexpr int kInterleave = 1;

  struct Mapper {
    __device__ __forceinline__ int operator()(int i) {
      return i;
    }
  };
};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajorInterleavedForHopper {
  using DetailsA = TypeDetailsA;
  using DetailsW = TypeDetailsW;
  using AccessTypeA = float4;
  using AccessTypeW = int4;
  static constexpr int kAccessSize = 128;
  static constexpr int kStepK = kAccessSize / TypeDetailsW::kElemBits;
  static constexpr int kTileSize = TileSizeK;
  static constexpr int kInterleave = 1;

  static constexpr int kTypeFactor = 128 * 8 / (TileSizeK * TypeDetailsW::kElemBits);

  // constants for mapper
  static constexpr int kElementGroupSizeA = TileSizeK / 32;
  static constexpr int kElementGroupSizeW = kTypeFactor * kElementGroupSizeA;
  static constexpr int kGroupOffsetA = 4 * kElementGroupSizeA;

  struct Mapper {
    __device__ __forceinline__ int operator()(int i) {
      return i % kElementGroupSizeA + (i % kGroupOffsetA) / kElementGroupSizeA * kElementGroupSizeW + i / kGroupOffsetA * kElementGroupSizeA;
    }
  };
};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajorInterleaved {
  using DetailsA = TypeDetailsA;
  using DetailsW = TypeDetailsW;
  using AccessTypeA = float4;
  using AccessTypeW = int4;
  static constexpr int kAccessSize = 128;
  static constexpr int kStepK = kAccessSize / TypeDetailsW::kElemBits;
  static constexpr int kTileSize = TileSizeK;
  static constexpr int kInterleave = 128 * 8 / (TileSizeK * TypeDetailsW::kElemBits);

  // constants for mapper
  static constexpr int kElementGroupSizeA = TileSizeK / 32;
  static constexpr int kElementGroupSizeW = kInterleave * kElementGroupSizeA;
  static constexpr int kGroupOffsetA = 4 * kElementGroupSizeA;

  struct Mapper {
    __device__ __forceinline__ int operator()(int i) {
      return i % kElementGroupSizeA + (i % kGroupOffsetA) / kElementGroupSizeA * kElementGroupSizeW + i / kGroupOffsetA * kElementGroupSizeA;
    }
  };
};

template <typename TypeDetailsA_, typename TypeDetailsW_, template <typename, typename, int> class LayoutDetails_,
          bool UseInterleavedConverter, int TileSizeK>
struct KernelDetails {
  using TypeDetailsA = TypeDetailsA_;
  using TypeDetailsW = TypeDetailsW_;
  using LayoutDetails = LayoutDetails_<TypeDetailsA, TypeDetailsW, TileSizeK>;
  using AccessTypeA = typename LayoutDetails::AccessTypeA;
  using AccessTypeW = typename LayoutDetails::AccessTypeW;
  static constexpr int kWarpSize = 32;
  static constexpr int kStepK = LayoutDetails::kStepK;
  static constexpr int kAccessNumA = kStepK * TypeDetailsA::kElemBits / (sizeof(AccessTypeA) * 8);
  static constexpr int kAccessNumW = kStepK * TypeDetailsW::kElemBits / (sizeof(AccessTypeW) * 8);
  static constexpr int kInterleave = LayoutDetails::kInterleave;
  static constexpr int kThreadsPerInterleavedTile = LayoutDetails::kTileSize / kStepK;
  static constexpr int kElemsPerByteW = 8 / TypeDetailsW::kElemBits;
  static constexpr bool kUseInterleavedConverter = UseInterleavedConverter;
};

template <typename AType, int WElemBits, bool Interleave>
struct I2FConverter;

template <typename AType, int WElemBits>
struct I2FConverter<AType, WElemBits, true> {
  static_assert(std::is_same_v<AType, half> || std::is_same_v<AType, __nv_bfloat16>);
  static_assert(WElemBits == 4 || WElemBits == 8);
  using CutlassAType = std::conditional_t<std::is_same_v<AType, half>, cutlass::half_t, cutlass::bfloat16_t>;
  using CutlassWType = std::conditional_t<WElemBits == 4, cutlass::uint4b_t, uint8_t>;
  static constexpr int kConvertCount = 32 / WElemBits;
  using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<CutlassAType, CutlassWType, kConvertCount>;
  using CvtSrcType = typename Converter::source_type;
  using CvtResType = typename Converter::result_type;

  template <int N>
  __device__ __forceinline__ static void convert(void* src, void* dst) {
    static_assert(N % kConvertCount == 0);
#pragma unroll
    for (int ii = 0; ii < N / kConvertCount; ++ii) {
      reinterpret_cast<CvtResType*>(dst)[ii] = Converter::convert(reinterpret_cast<CvtSrcType*>(src)[ii]);
    }
  }
};

template <typename AType, int WElemBits>
struct I2FConverter<AType, WElemBits, false> {
  static_assert(std::is_same_v<AType, half> || std::is_same_v<AType, __nv_bfloat16>);
  static_assert(WElemBits == 4 || WElemBits == 8);
  using CutlassAType = std::conditional_t<std::is_same_v<AType, half>, cutlass::half_t, cutlass::bfloat16_t>;
  using CutlassWType = std::conditional_t<WElemBits == 4, cutlass::int4b_t, int8_t>;
  static constexpr int kConvertCount = 32 / WElemBits;
  using Converter = cutlass::NumericArrayConverter<CutlassAType, CutlassWType, kConvertCount>;
  using CvtSrcType = typename Converter::source_type;
  using CvtResType = typename Converter::result_type;

  template <int N>
  __device__ __forceinline__ static void convert(void* src, void* dst) {
    static_assert(N % kConvertCount == 0);
#pragma unroll
    for (int ii = 0; ii < N / kConvertCount; ++ii) {
      reinterpret_cast<CvtResType*>(dst)[ii] = Converter::convert(reinterpret_cast<CvtSrcType*>(src)[ii]);
    }
  }
};

template <typename AType>
struct Fp4I2FConverter {
  static_assert(std::is_same_v<AType, half> || std::is_same_v<AType, __nv_bfloat16>);

  // Branchless E2M1 (FP4) -> half/bf16 decode. E2M1 has only eight magnitudes
  // {0, 0.5, 1, 1.5, 2, 3, 4, 6}, so the 16-bit float bit pattern is built directly with a
  // single prmt.b32 byte-select from packed magnitude constants (two for bf16, whose low byte
  // is not always zero) plus a shifted sign bit. This replaces the per-element float LUT
  // lookup + sign branch + float->AType conversion, which profiling showed to be the dominant
  // ALU-pipeline cost of the small-decode FP4 GEMV (ncu: ALU ~79% of a compute-bound kernel).
  // The magnitude bytes below are the exact half/bf16 encodings of the eight FP4 values, so the
  // result is bit-identical to the previous LUT path.
  __device__ __forceinline__ static AType decode(uint8_t code) {
#if defined(__CUDA_ARCH__)
    uint32_t const sel = code & 0x7u;
    uint32_t const sign = static_cast<uint32_t>(code & 0x8u) << 12;  // FP4 sign bit -> bit 15
    if constexpr (std::is_same_v<AType, half>) {
      // half high byte per magnitude (low byte is always 0):
      //   codes 0..3 -> {0x00, 0x38, 0x3C, 0x3E}, codes 4..7 -> {0x40, 0x42, 0x44, 0x46}.
      uint32_t hb;
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3E3C3800u), "r"(0x46444240u), "r"(sel));
      return __ushort_as_half(static_cast<uint16_t>(((hb & 0xFFu) << 8) | sign));
    } else {
      // bf16 high byte {0x00,0x3F,0x3F,0x3F, 0x40,0x40,0x40,0x40} and
      //      low  byte {0x00,0x00,0x80,0xC0, 0x00,0x40,0x80,0xC0}.
      uint32_t hb, lb;
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3F3F3F00u), "r"(0x40404040u), "r"(sel));
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lb) : "r"(0xC0800000u), "r"(0xC0804000u), "r"(sel));
      return __ushort_as_bfloat16(static_cast<uint16_t>(((hb & 0xFFu) << 8) | (lb & 0xFFu) | sign));
    }
#else
    constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = kValues[code & 0x7];
    return static_cast<AType>((code & 0x8) ? -v : v);
#endif
  }

  template <int N>
  __device__ __forceinline__ static void convert(void* src, void* dst) {
    static_assert(N % 2 == 0);
    uint8_t const* s = reinterpret_cast<uint8_t const*>(src);
    AType* d = reinterpret_cast<AType*>(dst);
#pragma unroll
    for (int i = 0; i < N; i += 2) {
      uint8_t byte = s[i >> 1];
      d[i] = decode(static_cast<uint8_t>(byte & 0x0F));
      d[i + 1] = decode(static_cast<uint8_t>((byte >> 4) & 0x0F));
    }
  }
};

template <typename Details>
struct ConverterWrapper {
  using TypeDetailsA = typename Details::TypeDetailsA;
  using TypeDetailsW = typename Details::TypeDetailsW;
  static constexpr bool kUseInterleavedConverter = Details::kUseInterleavedConverter;
  using Converter = std::conditional_t<
      IsFp4Weight<TypeDetailsW>::value,
      Fp4I2FConverter<typename TypeDetailsA::Type>,
      I2FConverter<typename TypeDetailsA::Type, TypeDetailsW::kElemBits, kUseInterleavedConverter>>;
};

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
