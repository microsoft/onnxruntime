// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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

}  // namespace rotary_embedding_int32_utils
}  // namespace onnxruntime
