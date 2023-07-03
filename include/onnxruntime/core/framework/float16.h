// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "endian.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include "cuda_bf16.h"
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include "core/common/narrow.h"
#endif

#include "core/common/common.h"
#include <limits>

namespace onnxruntime {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORT_HOST_DEVICE __host__ __device__
#else
#define ORT_HOST_DEVICE
#endif

// MLFloat16
struct MLFloat16 {
  // uint16_t special values
  static constexpr uint16_t kSignMask = 0x8000U;
  static constexpr uint16_t kBiasedExponentMask = 0x7C00U;
  static constexpr uint16_t kPositiveInfinityBits = 0x7C00U;
  static constexpr uint16_t kNegativeInfinityBits = 0xFC00U;
  static constexpr uint16_t kPositiveQNaNBits = 0x7E00U;
  static constexpr uint16_t kNegativeQNaNBits = 0xFE00U;
  static constexpr uint16_t kEpsilonBits = 0x4170U;
  static constexpr uint16_t kMinValueBits = 0xFBFFU;
  static constexpr uint16_t kMaxValueBits = 0x7BFFU;

  uint16_t val{0};

  MLFloat16() = default;
  explicit constexpr MLFloat16(uint16_t x) noexcept : val(x) {}
  constexpr static MLFloat16 FromBits(uint16_t x) noexcept { return MLFloat16(x); }
  explicit MLFloat16(float f);

  static const MLFloat16 NaN;
  static const MLFloat16 NegativeNaN;
  static const MLFloat16 Infinity;
  static const MLFloat16 NegativeInfinity;
  static const MLFloat16 Epsilon;
  static const MLFloat16 MinValue;
  static const MLFloat16 MaxValue;

  float ToFloat() const;

  bool IsNegative() const noexcept {
    return static_cast<int16_t>(val) < 0;
  }

  bool IsNaN() const noexcept {
    return Abs().val > kPositiveInfinityBits;
  }

  bool IsFinite() const noexcept {
    return Abs().val < kPositiveInfinityBits;
  }

  bool IsPositiveInfinity() const noexcept {
    return val == kPositiveInfinityBits;
  }

  bool IsNegativeInfinity() const noexcept {
    return val == kNegativeInfinityBits;
  }

  bool IsInfinity() const noexcept {
    return Abs().val == kPositiveInfinityBits;
  }

  bool IsNaNOrZero() const noexcept {
    return ((val - 1) & ~kSignMask) >= kPositiveInfinityBits;
  }

  bool IsNormal() const noexcept {
    auto abs = Abs();
    return (abs.val < kPositiveInfinityBits)           // is finite
           && (abs.val != 0)                           // is not zero
           && ((abs.val & kBiasedExponentMask) != 0);  // is not subnormal (has a non-zero exponent)
  }

  bool IsSubnormal() const noexcept {
    auto abs = Abs();
    return (abs.val < kPositiveInfinityBits)           // is finite
           && (abs.val != 0)                           // is not zero
           && ((abs.val & kBiasedExponentMask) == 0);  // is subnormal (has a zero exponent)
  }

  constexpr MLFloat16 Abs() const noexcept {
    return MLFloat16::FromBits(static_cast<uint16_t>(val & ~kSignMask));
  }

  MLFloat16 Negate() const noexcept {
    return IsNaN() ? *this : MLFloat16::FromBits(static_cast<uint16_t>(val ^ kSignMask));
  }

  operator float() const noexcept { return ToFloat(); }
};

inline bool AreZero(const MLFloat16& lhs, const MLFloat16& rhs) noexcept {
  // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
  // for two values by or'ing the private bits together and stripping the sign. They are both zero,
  // and therefore equivalent, if the resulting value is still zero.
  return static_cast<uint16_t>((lhs.val | rhs.val) & ~MLFloat16::kSignMask) == 0;
}

inline bool operator==(const MLFloat16& lhs, const MLFloat16& rhs) noexcept {
  if (lhs.IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is not equal to anything, including itself.
    return false;
  }
  return lhs.val == rhs.val;
}

inline bool operator!=(const MLFloat16& lhs, const MLFloat16& rhs) noexcept {
  return !(lhs == rhs);
}

inline bool operator<(const MLFloat16& lhs, const MLFloat16& rhs) noexcept {
  if (lhs.IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is unordered with respect to everything, including itself.
    return false;
  }

  const bool left_is_negative = lhs.IsNegative();

  if (left_is_negative != rhs.IsNegative()) {
    // When the signs of left and right differ, we know that left is less than right if it is
    // the negative value. The exception to this is if both values are zero, in which case IEEE
    // says they should be equal, even if the signs differ.
    return left_is_negative && !AreZero(lhs, rhs);
  }

  return (lhs.val != rhs.val) && ((lhs.val < rhs.val) ^ left_is_negative);
}

// BFloat16
struct BFloat16 {
  // uint16_t special values
  static constexpr uint16_t kSignMask = 0x8000U;
  static constexpr uint16_t kBiasedExponentMask = 0x7F80U;
  static constexpr uint16_t kPositiveInfinityBits = 0x7F80U;
  static constexpr uint16_t kNegativeInfinityBits = 0xFF80U;
  static constexpr uint16_t kPositiveQNaNBits = 0x7FC1U;
  static constexpr uint16_t kNegativeQNaNBits = 0xFFC1U;
  static constexpr uint16_t kSignaling_NaNBits = 0x7F80U;
  static constexpr uint16_t kEpsilonBits = 0x0080U;
  static constexpr uint16_t kMinValueBits = 0xFF7FU;
  static constexpr uint16_t kMaxValueBits = 0x7F7FU;
  static constexpr uint16_t kRoundToNearest = 0x7FFFU;

  uint16_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() noexcept { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) noexcept : val(bits) {}

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

    if (v != v) {
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

    // float NaN encodings are not specified and encoded
    // differently on different processors, so we use limits
    // Infinities will be propagated as is.
    if (IsNaN()) {
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

  ORT_HOST_DEVICE bool IsNegative() const noexcept {
    return static_cast<int16_t>(val) < 0;
  }

  ORT_HOST_DEVICE bool IsNaN() const noexcept {
    return Abs().val > kPositiveInfinityBits;
  }

  ORT_HOST_DEVICE bool IsFinite() const noexcept {
    return Abs().val < kPositiveInfinityBits;
  }

  ORT_HOST_DEVICE bool IsPositiveInfinity() const noexcept {
    return val == kPositiveInfinityBits;
  }

  ORT_HOST_DEVICE bool IsNegativeInfinity() const noexcept {
    return val == kNegativeInfinityBits;
  }

  ORT_HOST_DEVICE bool IsInfinity() const noexcept {
    return Abs().val == kPositiveInfinityBits;
  }

  ORT_HOST_DEVICE bool IsNaNOrZero() const noexcept {
    return ((val - 1) & ~kSignMask) >= kPositiveInfinityBits;
  }

  ORT_HOST_DEVICE bool IsNormal() const noexcept {
    auto abs = Abs();
    return (abs.val < kPositiveInfinityBits)           // is finite
           && (abs.val != 0)                           // is not zero
           && ((abs.val & kBiasedExponentMask) != 0);  // is not subnormal (has a non-zero exponent)
  }

  ORT_HOST_DEVICE bool IsSubnormal() const noexcept {
    auto abs = Abs();
    return (abs.val < kPositiveInfinityBits)           // is finite
           && (abs.val != 0)                           // is not zero
           && ((abs.val & kBiasedExponentMask) == 0);  // is subnormal (has a zero exponent)
  }

  ORT_HOST_DEVICE BFloat16 Abs() const noexcept {
    return BFloat16::FromBits(static_cast<uint16_t>(val & ~kSignMask));
  }

  ORT_HOST_DEVICE BFloat16 Negate() const noexcept {
    return IsNaN() ? *this : BFloat16::FromBits(static_cast<uint16_t>(val ^ kSignMask));
  }

  ORT_HOST_DEVICE operator float() const noexcept { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_HOST_DEVICE BFloat16(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&val); }
#endif
};

inline ORT_HOST_DEVICE bool AreZero(const BFloat16& lhs, const BFloat16& rhs) noexcept {
  // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
  // for two values by or'ing the private bits together and stripping the sign. They are both zero,
  // and therefore equivalent, if the resulting value is still zero.
  return static_cast<uint16_t>((lhs.val | rhs.val) & ~BFloat16::kSignMask) == 0;
}

inline ORT_HOST_DEVICE bool operator==(const BFloat16& lhs, const BFloat16& rhs) noexcept {
  if (lhs.IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is not equal to anything, including itself.
    return false;
  }
  return lhs.val == rhs.val;
}

inline ORT_HOST_DEVICE bool operator!=(const BFloat16& lhs, const BFloat16& rhs) noexcept {
  return !(lhs == rhs);
}

inline ORT_HOST_DEVICE bool operator<(const BFloat16& lhs, const BFloat16& rhs) noexcept {
  if (lhs.IsNaN() || rhs.IsNaN()) {
    // IEEE defines that NaN is unordered with respect to everything, including itself.
    return false;
  }

  const bool left_is_negative = lhs.IsNegative();

  if (left_is_negative != rhs.IsNegative()) {
    // When the signs of left and right differ, we know that left is less than right if it is
    // the negative value. The exception to this is if both values are zero, in which case IEEE
    // says they should be equal, even if the signs differ.
    return left_is_negative && !AreZero(lhs, rhs);
  }

  return (lhs.val != rhs.val) && ((lhs.val < rhs.val) ^ left_is_negative);
}

// User defined suffixes to make it easier to declare
// initializers with MLFloat16 and BFloat16 from unsigned short
// E.g 10_f16 or 10_b16
#if !defined(__CUDACC__) && !defined(__HIPCC__)
inline MLFloat16 operator"" _f16(unsigned long long int v) noexcept {
  return MLFloat16(narrow<uint16_t>(v));
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
