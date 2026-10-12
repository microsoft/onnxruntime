// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "core/common/float16.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/prepacked_weights.h"
#include "core/framework/tensor.h"
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
// round-to-nearest-even loop (no hardware bf16 instructions on AVX2).
template <typename T>
void FloatToNarrow(const float* src, T* dst, size_t count) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(src, dst, count);
  } else {
    static_assert(std::is_same_v<T, BFloat16>);
    FloatToBFloat16(src, dst, count);
  }
}

// Type trait: true for MLFloat16 and BFloat16 — the narrow-float types that
// need widen-to-f32 conversion before arithmetic.
template <typename T>
inline constexpr bool is_narrow_float_v = std::is_same_v<T, MLFloat16> ||
                                          std::is_same_v<T, BFloat16>;

// Converts a narrow float (MLFloat16 or BFloat16) initializer to f32 in a kernel's PrePack().
// If `prepacked_weights` is not null, the f32 buffer is handed to it so that the session can cache the
// pre-packed weight and share it across sessions; the kernel then gets the buffer back through
// UseSharedPrePackedBuffers(). Otherwise the kernel keeps the buffer in `dest`.
inline void ConvertNarrowFloatToFloatIfNeeded(const Tensor& tensor, AllocatorPtr alloc,
                                              IAllocatorUniquePtr<void>& dest, bool& is_packed,
                                              PrePackedWeights* prepacked_weights) {
  const bool is_fp16 = tensor.GetElementType() == utils::ToTensorProtoElementType<MLFloat16>();
  const bool is_bf16 = tensor.GetElementType() == utils::ToTensorProtoElementType<BFloat16>();
  if (!is_fp16 && !is_bf16) {
    return;
  }

  const auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
  const size_t buffer_size = SafeInt<size_t>(tensor_size) * sizeof(float);
  auto buffer = IAllocator::MakeUniquePtr<void>(alloc, buffer_size, true);
  if (tensor_size > 0) {
    auto* float_data = static_cast<float*>(buffer.get());
    if (is_fp16) {
      NarrowToFloat(tensor.Data<MLFloat16>(), float_data, tensor_size);
    } else {
      NarrowToFloat(tensor.Data<BFloat16>(), float_data, tensor_size);
    }
  }

  if (prepacked_weights != nullptr) {
    prepacked_weights->buffers_.push_back(std::move(buffer));
    prepacked_weights->buffer_sizes_.push_back(buffer_size);
  } else {
    dest = std::move(buffer);
  }
  is_packed = true;
}

}  // namespace onnxruntime
