// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <limits>

#include "core/common/common.h"

namespace onnxruntime {
namespace rotary_embedding_int32_utils {

inline Status NarrowNonNegativeToInt32Impl(int64_t value, const char* name, const char* error_suffix, int& output) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, "=", value, error_suffix);
  }

  output = static_cast<int>(value);
  return Status::OK();
}

inline Status NarrowNonNegativeToInt32(int64_t value, const char* name, int& output) {
  return NarrowNonNegativeToInt32Impl(value, name, " is out of range for int32", output);
}

inline Status CheckedMulToInt32(int lhs, int rhs, const char* name, int& output) {
  if (lhs < 0 || rhs < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " must be non-negative");
  }

  const int64_t product = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
  return NarrowNonNegativeToInt32Impl(product, name, " overflows int32", output);
}

inline Status CheckedMulToPtrdiff(int lhs, int rhs, const char* name, std::ptrdiff_t& output) {
  if (lhs < 0 || rhs < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " must be non-negative");
  }
  if (lhs != 0 && rhs > std::numeric_limits<std::ptrdiff_t>::max() / lhs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " overflows ptrdiff_t");
  }

  output = static_cast<std::ptrdiff_t>(lhs) * rhs;
  return Status::OK();
}

inline Status CheckedPtrdiffMulToPtrdiff(std::ptrdiff_t lhs, int rhs, const char* name, std::ptrdiff_t& output) {
  if (lhs < 0 || rhs < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " must be non-negative");
  }
  if (rhs != 0 && lhs > std::numeric_limits<std::ptrdiff_t>::max() / rhs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " overflows ptrdiff_t");
  }

  output = lhs * rhs;
  return Status::OK();
}

inline Status CheckedAddToPtrdiff(std::ptrdiff_t lhs, std::ptrdiff_t rhs, const char* name, std::ptrdiff_t& output) {
  if (lhs < 0 || rhs < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " must be non-negative");
  }
  if (lhs > std::numeric_limits<std::ptrdiff_t>::max() - rhs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " overflows ptrdiff_t");
  }

  output = lhs + rhs;
  return Status::OK();
}

inline Status CheckedMulToPtrdiff(int lhs, int rhs, int third, const char* name, std::ptrdiff_t& output) {
  std::ptrdiff_t intermediate = 0;
  ORT_RETURN_IF_ERROR(CheckedMulToPtrdiff(lhs, rhs, name, intermediate));
  if (third < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " must be non-negative");
  }
  if (intermediate != 0 && third > std::numeric_limits<std::ptrdiff_t>::max() / intermediate) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "RotaryEmbedding: ", name, " overflows ptrdiff_t");
  }

  output = intermediate * third;
  return Status::OK();
}

}  // namespace rotary_embedding_int32_utils
}  // namespace onnxruntime
