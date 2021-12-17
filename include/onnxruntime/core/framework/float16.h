// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {

// MLFloat16
struct MLFloat16 {
  uint16_t val;

  MLFloat16() : val(0) {}
  explicit MLFloat16(uint16_t x) : val(x) {}
  explicit MLFloat16(float f);

  float ToFloat() const;

  operator float() const { return ToFloat(); }
};

inline bool operator==(const MLFloat16& left, const MLFloat16& right) { return left.val == right.val; }
inline bool operator!=(const MLFloat16& left, const MLFloat16& right) { return left.val != right.val; }
inline bool operator<(const MLFloat16& left, const MLFloat16& right) { return left.val < right.val; }

}  // namespace onnxruntime
