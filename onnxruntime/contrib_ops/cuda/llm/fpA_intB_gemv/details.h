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

template <typename Details>
struct ConverterWrapper {
  using TypeDetailsA = typename Details::TypeDetailsA;
  using TypeDetailsW = typename Details::TypeDetailsW;
  static constexpr bool kUseInterleavedConverter = Details::kUseInterleavedConverter;
  using Converter = I2FConverter<typename TypeDetailsA::Type, TypeDetailsW::kElemBits, kUseInterleavedConverter>;
};

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
