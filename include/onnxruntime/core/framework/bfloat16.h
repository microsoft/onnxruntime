// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "endian.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include "cuda_bf16.h"
#endif

#include "core/common/common.h"

namespace onnxruntime {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORT_HOST_DEVICE __host__ __device__
#else
#define ORT_HOST_DEVICE
#endif

// BFloat16
struct BFloat16 {
  uint16_t val{0};
#if defined(USE_ROCM)
  ORT_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) : val(bits){};

  inline ORT_HOST_DEVICE BFloat16(float v) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    val = __bfloat16_as_ushort(__float2bfloat16(v));
#else
    ORT_IF_CONSTEXPR(endian::native == endian::little) {
      std::memcpy(&val, reinterpret_cast<char*>(&v) + sizeof(uint16_t), sizeof(uint16_t));
    }
    else {
      std::memcpy(&val, &v, sizeof(uint16_t));
    }
#endif
  }

  inline ORT_HOST_DEVICE float ToFloat() const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
#else
    float result;
    char* const first = reinterpret_cast<char*>(&result);
    char* const second = first + sizeof(uint16_t);
    ORT_IF_CONSTEXPR(endian::native == endian::little) {
      std::memset(first, 0, sizeof(uint16_t));
      std::memcpy(second, &val, sizeof(uint16_t));
    }
    else {
      std::memcpy(first, &val, sizeof(uint16_t));
      std::memset(second, 0, sizeof(uint16_t));
    }
    return result;
#endif
  }

  inline ORT_HOST_DEVICE operator float() const { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_HOST_DEVICE BFloat16(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&val); }
#endif
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

/// Arithmetic for BFloat16

inline ORT_HOST_DEVICE BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline ORT_HOST_DEVICE BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline ORT_HOST_DEVICE BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline ORT_HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline ORT_HOST_DEVICE BFloat16 operator-(const BFloat16& a) { return -static_cast<float>(a); }

inline ORT_HOST_DEVICE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

inline ORT_HOST_DEVICE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

inline ORT_HOST_DEVICE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

inline ORT_HOST_DEVICE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline ORT_HOST_DEVICE float operator+(BFloat16 a, float b) { return static_cast<float>(a) + b; }
inline ORT_HOST_DEVICE float operator-(BFloat16 a, float b) { return static_cast<float>(a) - b; }
inline ORT_HOST_DEVICE float operator*(BFloat16 a, float b) { return static_cast<float>(a) * b; }
inline ORT_HOST_DEVICE float operator/(BFloat16 a, float b) { return static_cast<float>(a) / b; }

inline ORT_HOST_DEVICE float operator+(float a, BFloat16 b) { return a + static_cast<float>(b); }
inline ORT_HOST_DEVICE float operator-(float a, BFloat16 b) { return a - static_cast<float>(b); }
inline ORT_HOST_DEVICE float operator*(float a, BFloat16 b) { return a * static_cast<float>(b); }
inline ORT_HOST_DEVICE float operator/(float a, BFloat16 b) { return a / static_cast<float>(b); }

inline ORT_HOST_DEVICE float& operator+=(float& a, const BFloat16& b) { return a += static_cast<float>(b); }
inline ORT_HOST_DEVICE float& operator-=(float& a, const BFloat16& b) { return a -= static_cast<float>(b); }
inline ORT_HOST_DEVICE float& operator*=(float& a, const BFloat16& b) { return a *= static_cast<float>(b); }
inline ORT_HOST_DEVICE float& operator/=(float& a, const BFloat16& b) { return a /= static_cast<float>(b); }

/// Arithmetic with doubles

inline ORT_HOST_DEVICE double operator+(BFloat16 a, double b) { return static_cast<double>(a) + b; }
inline ORT_HOST_DEVICE double operator-(BFloat16 a, double b) { return static_cast<double>(a) - b; }
inline ORT_HOST_DEVICE double operator*(BFloat16 a, double b) { return static_cast<double>(a) * b; }
inline ORT_HOST_DEVICE double operator/(BFloat16 a, double b) { return static_cast<double>(a) / b; }

inline ORT_HOST_DEVICE double operator+(double a, BFloat16 b) { return a + static_cast<double>(b); }
inline ORT_HOST_DEVICE double operator-(double a, BFloat16 b) { return a - static_cast<double>(b); }
inline ORT_HOST_DEVICE double operator*(double a, BFloat16 b) { return a * static_cast<double>(b); }
inline ORT_HOST_DEVICE double operator/(double a, BFloat16 b) { return a / static_cast<double>(b); }

// Overloading < and > operators

inline ORT_HOST_DEVICE bool operator==(BFloat16& lhs, BFloat16& rhs) { return float(lhs) == float(rhs); }
inline ORT_HOST_DEVICE bool operator!=(BFloat16& lhs, BFloat16& rhs) { return float(lhs) != float(rhs); }
inline ORT_HOST_DEVICE bool operator>(BFloat16& lhs, BFloat16& rhs) { return float(lhs) > float(rhs); }
inline ORT_HOST_DEVICE bool operator<(BFloat16& lhs, BFloat16& rhs) { return float(lhs) < float(rhs); }

}  // namespace onnxruntime
