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

// MXFP4 weight element descriptor. Same 4-bit storage as Int4DetailsW, but the
// codes are e2m1 floating-point (not signed integers), so the dequant path uses
// the dedicated Fp4I2FConverter LUT instead of the integer fast converter.
struct Fp4DetailsW {
  static constexpr int kElemBits = 4;
};

// Opt-in trait identifying MXFP4 (e2m1) weight descriptors. Defaults to false so
// the integer weight descriptors stay free of FP4-specific members; only
// Fp4DetailsW specializes it to true.
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

// MXFP4 (e2m1) -> half/bf16 in-register converter for the non-interleaved
// ColumnMajor GEMV layout. The codes are stored two per byte with the low nibble
// holding the even element. Decode reproduces DecodeFp4E2M1 in qmoe_kernels.cu:
// magnitude {0,.5,1,1.5,2,3,4,6} from code&0x7, sign bit = code&0x8.
//
// The decode is done with branch-free integer bit math that assembles the IEEE
// half/bf16 bit pattern directly, instead of an indexed `constexpr float[8]` LUT.
// The LUT version compiled to a data-dependent local-memory load plus a sign
// branch per nibble; ncu showed the GEMV pinned at ~88% SM (compute/LSU) with
// only ~8% DRAM, i.e. the per-element LUT decode -- not weight bandwidth -- was
// the bottleneck. The bit-math path is a few ALU ops per nibble, lowers register
// pressure, and is numerically identical (every e2m1 value is exact in fp16/bf16).
//
// e2m1 magnitude m = code&0x7 maps to an exponent (m>>1) and a half-mantissa bit
// (m&1). For m>=2 the value is normal: fp16 exp = (m>>1)+14, top mantissa bit =
// (m&1); bf16 exp = (m>>1)+126, mantissa bit at position 6. m<2 is the subnormal
// pair {0 -> 0.0, 1 -> 0.5}, handled by a single predicated select.
template <typename AType>
struct Fp4I2FConverter {
  static_assert(std::is_same_v<AType, half> || std::is_same_v<AType, __nv_bfloat16>);

  // Decode a single 4-bit e2m1 code into the raw 16-bit half/bf16 pattern.
  __device__ __forceinline__ static uint16_t decode_bits(uint32_t q) {
    uint32_t const sign = (q & 0x8u) << 12;  // bit3 -> bit15 (half and bf16 share sign position)
    uint32_t const mag = q & 0x7u;
    if constexpr (std::is_same_v<AType, half>) {
      uint32_t const normal = (((mag >> 1) + 14u) << 10) | ((mag & 1u) << 9);
      uint32_t const sub = mag * 0x3800u;  // m=0 -> 0x0000 (0.0), m=1 -> 0x3800 (0.5)
      return static_cast<uint16_t>((mag >= 2u ? normal : sub) | sign);
    } else {
      uint32_t const normal = (((mag >> 1) + 126u) << 7) | ((mag & 1u) << 6);
      uint32_t const sub = mag * 0x3F00u;  // m=0 -> 0x0000 (0.0), m=1 -> 0x3F00 (0.5)
      return static_cast<uint16_t>((mag >= 2u ? normal : sub) | sign);
    }
  }

  __device__ __forceinline__ static AType decode(uint8_t code) {
    if constexpr (std::is_same_v<AType, half>) {
      return __ushort_as_half(decode_bits(code));
    } else {
      return __ushort_as_bfloat16(decode_bits(code));
    }
  }

  // Converts N e2m1 codes (packed two per byte, low nibble first) into N AType.
  // Each input byte holds two codes; decode both, assemble the pair into one
  // 32-bit word, and emit a single aligned store. This halves the dynamic store
  // count vs. two scalar writes and lets the two nibble decodes pipeline, easing
  // the compute-bound decode that dominates the FP4 GEMV. Bit-identical to two
  // decode() calls.
  template <int N>
  __device__ __forceinline__ static void convert(void* src, void* dst) {
    static_assert(N % 2 == 0);
    uint8_t const* s = reinterpret_cast<uint8_t const*>(src);
    uint32_t* d2 = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
    for (int i = 0; i < N; i += 2) {
      uint32_t const byte = s[i >> 1];
      uint32_t const lo = decode_bits(byte & 0x0Fu);
      uint32_t const hi = decode_bits((byte >> 4) & 0x0Fu);
      d2[i >> 1] = lo | (hi << 16);
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
