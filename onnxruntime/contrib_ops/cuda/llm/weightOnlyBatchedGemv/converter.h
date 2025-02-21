/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/common.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/interleaved_numeric_conversion.h"

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {
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

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
