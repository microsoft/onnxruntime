/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_fp16.h

Abstract:

    Define fp16 type before it is available in all compilers

--*/

#pragma once

#include "test_util.h"
#include "mlas_float16.h"

//
// Define our own fp16 type to avoid dragging in big dependencies
//
struct MLFp16 {
  uint16_t val{0};

  MLFp16() = default;
  explicit constexpr MLFp16(uint16_t x) : val(x) {}
  explicit constexpr MLFp16(int32_t x) : val((uint16_t)x) {}
  explicit MLFp16(float ff) : val(MLAS_Float2Half(ff)) {}

  float ToFloat() const {
    return MLAS_Half2Float(val);
  }

  operator float() const { return ToFloat(); }

  MLFp16& operator=(float ff) {
    val = MLAS_Float2Half(ff);
    return *this;
  }
};

inline bool
operator==(const MLFp16& left, const MLFp16& right) {
  return left.val == right.val;
}

inline bool
operator!=(const MLFp16& left, const MLFp16& right) {
  return left.val != right.val;
}

template <typename T>
void SmallFloatFill(T* start, size_t size) {
  constexpr float MinimumFillValue = -11.0f;
  auto* FillAddress = start;
  size_t offset = size % 23;

  for (size_t i = 0; i < size; i++) {
    offset = (offset + 21) % 23;
    *FillAddress++ = T((MinimumFillValue + offset) / 16.0f);
  }
}
