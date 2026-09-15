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

// Access shape for the CtaN per-column scales a GEMV block loads for one K tile.
//
// The scales of the CtaN columns owned by a block sit `Interleave` elements apart, so for the
// non-interleaved ColumnMajor layout (Interleave == 1) the whole CtaN-wide scale vector is
// contiguous and can be fetched with one wide access instead of CtaN scalar ones.
//
// This matters far more than the byte count suggests. With a groupwise scale (e.g. NVFP4's
// GroupSize = 16) and StepK = 8, a warp's 32 lanes cover 16 distinct scale rows that are `n`
// elements apart, so *every* scale load touches 16 different sectors. Issuing CtaN of them costs
// CtaN * 16 sectors while using only 2 bytes out of each 32-byte sector; a single vector load
// costs 16 sectors and uses CtaN * 2 bytes of each. On the Qwen3.6 NVFP4 MoE decode shape
// (CtaN = 8) that is 128 of the 176 L1 sectors a warp requests per K iteration.
//
// The vector form needs the scale row base to be 16-byte aligned. Callers guarantee
// n % CtaN == 0, and kVectorized implies CtaN * sizeof(TypeA) % 16 == 0, so both the row stride
// (a multiple of n) and the column offset (a multiple of CtaN) are 16-byte aligned.
template <typename T, int CtaN, int Interleave>
struct ScalesAccess {
  static constexpr int kBytes = CtaN * static_cast<int>(sizeof(T));
  static constexpr bool kVectorized = (Interleave == 1) && (kBytes % 16 == 0);
  // GMemIterator<..., TVec, Strided, Continuous, T>: the vector form loads all CtaN scales in
  // `Continuous` 16-byte chunks at ii = 0, the scalar form keeps one element per `ii`.
  using TVec = std::conditional_t<kVectorized, float4, T>;
  static constexpr int kStrided = kVectorized ? 1 : CtaN;
  static constexpr int kContinuous = kVectorized ? kBytes / 16 : 1;
};

// Loads the CtaN scales for K tile `iter` into `vec_scale`, using the access shape `Access`
// describes. `scales_iterator` must have been declared with that same shape.
template <typename Access, int CtaN, typename Iterator, typename T>
__device__ __forceinline__ void load_scales(Iterator& scales_iterator, T* vec_scale, int iter) {
  if constexpr (Access::kVectorized) {
    scales_iterator.load(vec_scale, iter);
  } else {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, iter, i);
    }
  }
}

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

// E2M1 (FP4) -> half/bf16 converter.
//
// ``PairInterleaved`` selects the nibble order of the packed source:
//   false -- linear order, i.e. element ``i`` occupies nibble ``i`` of the 32-bit word. This is
//            what ``LaunchQMoERepackFP4ColToRow`` emits and what the CUTLASS fpA_intB W4_A16
//            preprocessor's layout-only steps 1-3 preserve.
//   true  -- the ``[e0,e2,e4,e6,e1,e3,e5,e7]`` pair-interleave applied by
//            ``interleave_int4s_inplace_kernel`` (preprocessor step 4 without the integer +8
//            bias). This is the layout the SM80 grouped GEMM consumes, so reading it here lets a
//            single pre-packed weight buffer serve both the grouped-GEMM prefill and the fused
//            GEMV decode instead of keeping two full copies of the expert weights.
// The un-permutation is a compile-time index remap of the same eight ``decode`` calls, so it
// costs no extra registers, branches or ALU work.
template <typename AType, bool PairInterleaved = false>
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

#if defined(__CUDA_ARCH__)
  // Decodes four consecutive E2M1 codes into two packed AType2 words.
  //
  // `mag_sel` holds the four 3-bit magnitudes as the four low nibbles (bit 3 cleared so prmt
  // stays in byte-select mode rather than sign-replicate mode) and `sgn_sel` holds the four sign
  // bits as 0/1 nibbles. One prmt then performs *four* magnitude table lookups at once, which is
  // where this beats the per-element `decode()` above: prmt selects four bytes per instruction,
  // so the whole 4-element lookup costs one instruction instead of four.
  //
  // Selector notation below: `prmt.b32 d, a, b, c` views {a0,a1,a2,a3,b0,b1,b2,b3} as source
  // bytes 0..7 and uses nibble j of `c` (nibble 0 is the least significant) to pick source byte
  // for result byte j. All the selectors here are written most-significant nibble first, i.e.
  // 0x1404 means d3=src1, d2=src4, d1=src0, d0=src4.
  __device__ __forceinline__ static void decode_quad(uint32_t mag_sel, uint32_t sgn_sel,
                                                     uint32_t& lo2, uint32_t& hi2) {
    uint32_t sb;
    // Sign bytes. Source bytes are {0x00,0x80,0x00,0x00, 0,0,0,0}, so a `sgn_sel` nibble of 0
    // picks 0x00 and 1 picks 0x80 -- bit 7 of the AType high byte, i.e. the float sign bit.
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sb) : "r"(0x00008000u), "r"(0u), "r"(sgn_sel));
    if constexpr (std::is_same_v<AType, half>) {
      uint32_t hb;
      // Same magnitude table as decode(): codes 0..3 -> {0x00,0x38,0x3C,0x3E},
      // codes 4..7 -> {0x40,0x42,0x44,0x46}. hb byte j is element j's half high byte.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3E3C3800u), "r"(0x46444240u), "r"(mag_sel));
      hb |= sb;
      // half low byte is always 0, so expand {b0,b1,b2,b3} to {0,b0,0,b1} and {0,b2,0,b3} by
      // pulling the zero bytes from the second (all-zero) prmt operand.
      // 0x1404: d = {hb1, 0, hb0, 0} (byte 3..0) = half2{elem0, elem1}.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo2) : "r"(hb), "r"(0u), "n"(0x1404));
      // 0x3424: d = {hb3, 0, hb2, 0} = half2{elem2, elem3}.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi2) : "r"(hb), "r"(0u), "n"(0x3424));
    } else {
      uint32_t hb, lb;
      // Same two bf16 tables as decode(): high byte {0x00,0x3F,0x3F,0x3F, 0x40,0x40,0x40,0x40}
      // and low byte {0x00,0x00,0x80,0xC0, 0x00,0x40,0x80,0xC0}. Byte j is element j.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3F3F3F00u), "r"(0x40404040u), "r"(mag_sel));
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lb) : "r"(0xC0800000u), "r"(0xC0804000u), "r"(mag_sel));
      hb |= sb;
      // bf16 needs both bytes, so source bytes are {lb0..lb3, hb0..hb3}.
      // 0x5140: d = {hb1, lb1, hb0, lb0} = bfloat162{elem0, elem1}.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo2) : "r"(lb), "r"(hb), "n"(0x5140));
      // 0x7362: d = {hb3, lb3, hb2, lb2} = bfloat162{elem2, elem3}.
      asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi2) : "r"(lb), "r"(hb), "n"(0x7362));
    }
  }
#endif

  template <int N>
  __device__ __forceinline__ static void convert(void* src, void* dst) {
    if constexpr (!PairInterleaved) {
      static_assert(N % 2 == 0);
#if defined(__CUDA_ARCH__)
      if constexpr (N % 8 == 0) {
        // Packed path: decode a whole 32-bit weight word (eight codes) at a time. Bit-identical
        // to the scalar path below, but ~3x fewer instructions, which matters because the QMoE
        // FP4 GEMV is instruction-issue bound (ncu on Qwen3.6-35B-A3B NVFP4 decode: ~74% SM
        // throughput at ~12% DRAM throughput, with the dequantize sequence about half of all
        // issued instructions). Measured on H200/sm_90: FP4 GEMV kernel SASS shrinks ~30% and
        // the two QMoE GEMVs drop 33.2 -> 26.2 us (fc1 swiglu) and 30.2 -> 22.2 us (fc2).
        // Only valid for the plain (non pair-interleaved) nibble order: nibble j of the word
        // is logical element j, which is exactly what the prmt selectors below assume.
        //
        // Both operands are re-typed to uint32_t, so callers must supply 4-byte-aligned
        // buffers; the GEMV kernels declare their tiles `alignas(alignof(uint32_t))`.
        uint32_t const* sw = reinterpret_cast<uint32_t const*>(src);
        uint32_t* dw = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
        for (int i = 0; i < N / 8; ++i) {
          uint32_t const w = sw[i];
          uint32_t const mag = w & 0x77777777u;
          uint32_t const sgn = (w >> 3) & 0x11111111u;
          decode_quad(mag, sgn, dw[i * 4 + 0], dw[i * 4 + 1]);
          decode_quad(mag >> 16, sgn >> 16, dw[i * 4 + 2], dw[i * 4 + 3]);
        }
      } else
#endif
      {
        uint8_t const* s = reinterpret_cast<uint8_t const*>(src);
        AType* d = reinterpret_cast<AType*>(dst);
#pragma unroll
        for (int i = 0; i < N; i += 2) {
          uint8_t byte = s[i >> 1];
          d[i] = decode(static_cast<uint8_t>(byte & 0x0F));
          d[i + 1] = decode(static_cast<uint8_t>((byte >> 4) & 0x0F));
        }
      }
    } else {
      uint8_t const* s = reinterpret_cast<uint8_t const*>(src);
      AType* d = reinterpret_cast<AType*>(dst);
      // The pair-interleave permutes whole 32-bit words, so N must cover complete words.
      static_assert(N % 8 == 0, "Pair-interleaved FP4 decode needs a multiple of 8 elements");
      // Packing writes element i to nibble slot (i even ? i/2 : (i - 1)/2 + 4), so logical
      // element i is read back from slot kSlot[i]. Nibble slot j lives in byte j/2, low nibble
      // for even j. kSlot is constexpr and the loops are unrolled, so every index below folds
      // to a compile-time constant.
      constexpr int kSlot[8] = {0, 4, 1, 5, 2, 6, 3, 7};
#pragma unroll
      for (int w = 0; w < N / 8; ++w) {
        uint8_t const* word = s + w * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint8_t const byte = word[kSlot[i] >> 1];
          d[w * 8 + i] = decode(static_cast<uint8_t>((byte >> ((kSlot[i] & 1) * 4)) & 0x0F));
        }
      }
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
      Fp4I2FConverter<typename TypeDetailsA::Type, kUseInterleavedConverter>,
      I2FConverter<typename TypeDetailsA::Type, TypeDetailsW::kElemBits, kUseInterleavedConverter>>;
};

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
