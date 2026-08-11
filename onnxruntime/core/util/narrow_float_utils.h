// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <type_traits>

#include "core/common/float16.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

// Batch-convert a narrow float (MLFloat16 or BFloat16) buffer to f32.
// MLFloat16 uses the optimised MLAS vectorised path; BFloat16 uses a portable
// scalar loop (upper 16 bits → f32, no hardware bf16 instructions).
template <typename T>
void NarrowToFloat(const T* src, float* dst, size_t count) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertHalfToFloatBuffer(src, dst, count);
  } else {
    static_assert(std::is_same_v<T, BFloat16>);
    BFloat16ToFloat(src, dst, count);
  }
}

// Batch-convert f32 back to a narrow float (MLFloat16 or BFloat16) buffer.
// MLFloat16 uses the MLAS vectorised path; BFloat16 uses a portable scalar
// truncate-to-nearest-even loop (no hardware bf16 instructions on AVX2).
template <typename T>
void FloatToNarrow(const float* src, T* dst, size_t count) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(src, dst, count);
  } else {
    static_assert(std::is_same_v<T, BFloat16>);
    FloatToBFloat16(src, dst, count);
  }
}

}  // namespace onnxruntime
