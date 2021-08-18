// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "endian.h"

namespace onnxruntime
{

// MLFloat16
struct MLFloat16 {
  uint16_t val;

  MLFloat16() : val(0) {}
  explicit MLFloat16(uint16_t x) : val(x) {}
  explicit MLFloat16(float f);

  float ToFloat() const;

  operator float() const {
    return ToFloat();
  }
};

inline bool operator==(const MLFloat16& left, const MLFloat16& right) {
  return left.val == right.val;
}

inline bool operator!=(const MLFloat16& left, const MLFloat16& right) {
  return left.val != right.val;
}

inline bool operator<(const MLFloat16& left, const MLFloat16& right) {
  return left.val < right.val;
}

//BFloat16
struct BFloat16 {
  uint16_t val{0};
  explicit BFloat16() = default;
  explicit BFloat16(uint16_t v) : val(v) {}
  explicit BFloat16(float v) {
    ORT_IF_CONSTEXPR(endian::native == endian::little) {
      std::memcpy(&val, reinterpret_cast<char*>(&v) + sizeof(uint16_t), sizeof(uint16_t));
    } else {
      std::memcpy(&val, &v, sizeof(uint16_t));
    }
  }

  float ToFloat() const {
    float result;
    char* const first = reinterpret_cast<char*>(&result);
    char* const second = first + sizeof(uint16_t);
    ORT_IF_CONSTEXPR(endian::native == endian::little) {
      std::memset(first, 0, sizeof(uint16_t));
      std::memcpy(second, &val, sizeof(uint16_t));
    } else {
      std::memcpy(first, &val, sizeof(uint16_t));
      std::memset(second, 0, sizeof(uint16_t));
    }
    return result;
  }

  operator float() const {
    return ToFloat();
  }
};

inline void BFloat16ToFloat(const BFloat16* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToBFloat16(const float* flt, BFloat16* blf, size_t size) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) BFloat16(*src);
  }
}

inline bool operator==(const BFloat16& left, const BFloat16& right) {
  return left.val == right.val;
}

inline bool operator!=(const BFloat16& left, const BFloat16& right) {
  return left.val != right.val;
}

inline bool operator<(const BFloat16& left, const BFloat16& right) {
  return left.val < right.val;
}

}