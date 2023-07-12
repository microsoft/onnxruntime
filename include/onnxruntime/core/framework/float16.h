// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>

#include "endian.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include "cuda_bf16.h"
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include "core/common/narrow.h"
#endif

#include "core/common/common.h"

#include "core/session/onnxruntime_float16.h"

namespace onnxruntime {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORT_HOST_DEVICE __host__ __device__
#else
#define ORT_HOST_DEVICE
#endif

// MLFloat16
struct MLFloat16 : onnxruntime_float16::Float16Impl {
 private:
  explicit constexpr MLFloat16(uint16_t x) noexcept { val = x; }

 public:
  using Base = onnxruntime_float16::Float16Impl;

  MLFloat16() = default;
  constexpr static MLFloat16 FromBits(uint16_t x) noexcept { return MLFloat16(x); }
  // We continue to use math impl instead of inherited one
  explicit MLFloat16(float f);

  static const MLFloat16 NaN;
  static const MLFloat16 NegativeNaN;
  static const MLFloat16 Infinity;
  static const MLFloat16 NegativeInfinity;
  static const MLFloat16 Epsilon;
  static const MLFloat16 MinValue;
  static const MLFloat16 MaxValue;
  static const MLFloat16 Zero;
  static const MLFloat16 One;
  static const MLFloat16 MinusOne;

  // We continue to use math impl instead of inherited one
  float ToFloat() const;

  using Base::IsNegative;

  using Base::IsNaN;

  using Base::IsFinite;

  using Base::IsPositiveInfinity;

  using Base::IsNegativeInfinity;

  using Base::IsInfinity;

  using Base::IsNaNOrZero;

  using Base::IsNormal;

  using Base::IsSubnormal;

  MLFloat16 Abs() const noexcept {
    return MLFloat16::FromBits(Base::AbsImpl());
  }

  MLFloat16 Negate() const noexcept {
    return MLFloat16::FromBits(Base::NegateImpl());
  }

  operator float() const noexcept { return ToFloat(); }

  using Base::operator==;
  using Base::operator!=;
  using Base::operator<;
};

// BFloat16
struct BFloat16 : onnxruntime_float16::BFloat16Impl {
  using Base = onnxruntime_float16::BFloat16Impl;

#if defined(__HIP__)
  ORT_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() noexcept { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) noexcept { val = bits; }

  static constexpr ORT_HOST_DEVICE BFloat16 FromBits(uint16_t bits) noexcept {
    return BFloat16(bits, FromBits());
  }

  inline ORT_HOST_DEVICE BFloat16(float v) noexcept {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    val = __bfloat16_as_ushort(__float2bfloat16(v));
#elif defined(__HIP__)
    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
    if (v != v) {  // isnan
      val = UINT16_C(0x7FC0);
    } else {
      union {
        uint32_t U32;
        float F32;
      };

      F32 = v;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      val = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
#else
    if (std::isnan(v)) {
      val = kPositiveQNaNBits;
    } else {
      auto get_msb_half = [](float fl) {
        uint16_t result;
        if constexpr (endian::native == endian::little) {
          std::memcpy(&result, reinterpret_cast<char*>(&fl) + sizeof(uint16_t), sizeof(uint16_t));
        } else {
          std::memcpy(&result, &fl, sizeof(uint16_t));
        }
        return result;
      };

      uint16_t upper_bits = get_msb_half(v);
      union {
        uint32_t U32;
        float F32;
      };
      F32 = v;
      U32 += (upper_bits & 1) + kRoundToNearest;
      val = get_msb_half(F32);
    }
#endif
  }

  inline ORT_HOST_DEVICE float ToFloat() const noexcept {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
#elif defined(__HIP__)
    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
    float result = 0;
    uint32_t tmp = val;
    tmp <<= 16;
    float* tempRes = reinterpret_cast<float*>(&tmp);
    result = *tempRes;
    return result;
#else
    // Test for NaN
    if (static_cast<uint16_t>(val & ~kSignMask) > kPositiveInfinityBits) {
      return std::numeric_limits<float>::quiet_NaN();
    }

    float result = 0;
    char* const first = reinterpret_cast<char*>(&result);
    if constexpr (endian::native == endian::little) {
      char* const second = first + sizeof(uint16_t);
      std::memcpy(second, &val, sizeof(uint16_t));
    } else {
      std::memcpy(first, &val, sizeof(uint16_t));
    }
    return result;
#endif
  }

  static const BFloat16 NaN;
  static const BFloat16 NegativeNaN;
  static const BFloat16 Infinity;
  static const BFloat16 NegativeInfinity;
  static const BFloat16 Epsilon;
  static const BFloat16 MinValue;
  static const BFloat16 MaxValue;
  static const BFloat16 Zero;
  static const BFloat16 One;
  static const BFloat16 MinusOne;

  using Base::IsNegative;

  using Base::IsNaN;

  using Base::IsFinite;

  using Base::IsPositiveInfinity;

  using Base::IsNegativeInfinity;

  using Base::IsInfinity;

  using Base::IsNaNOrZero;

  using Base::IsNormal;

  using Base::IsSubnormal;

  BFloat16 Abs() const noexcept {
    return BFloat16::FromBits(Base::AbsImpl());
  }

  BFloat16 Negate() const noexcept {
    return BFloat16::FromBits(Base::NegateImpl());
  }

  ORT_HOST_DEVICE operator float() const noexcept { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_HOST_DEVICE BFloat16(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&val); }
#endif

  using Base::operator==;
  using Base::operator!=;
  using Base::operator<;
};

// User defined suffixes to make it easier to declare
// initializers with MLFloat16 and BFloat16 from unsigned short
// E.g 10_f16 or 10_b16
#if !defined(__CUDACC__) && !defined(__HIPCC__)
inline MLFloat16 operator"" _f16(unsigned long long int v) noexcept {
  return MLFloat16::FromBits(narrow<uint16_t>(v));
}

inline MLFloat16 operator"" _fp16(long double v) noexcept {
  return MLFloat16(static_cast<float>(v));
}

inline BFloat16 operator"" _b16(unsigned long long int v) noexcept {
  return BFloat16::FromBits((narrow<uint16_t>(v)));
}

inline BFloat16 operator"" _bfp16(long double v) noexcept {
  return BFloat16(static_cast<float>(v));
}
#endif

inline void BFloat16ToFloat(const BFloat16* blf, float* flt, size_t size) noexcept {
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
    *d = BFloat16(*src);
  }
}

}  // namespace onnxruntime
