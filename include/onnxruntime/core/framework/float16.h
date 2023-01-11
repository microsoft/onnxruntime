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

namespace onnxruntime {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORT_HOST_DEVICE __host__ __device__
#else
#define ORT_HOST_DEVICE
#endif

// MLFloat16
struct MLFloat16 {
  uint16_t val{0};

  MLFloat16() = default;
  explicit constexpr MLFloat16(uint16_t x) : val(x) {}
  explicit MLFloat16(float f);

  float ToFloat() const;

  operator float() const { return ToFloat(); }
};

inline bool operator==(const MLFloat16& left, const MLFloat16& right) { return left.val == right.val; }
inline bool operator!=(const MLFloat16& left, const MLFloat16& right) { return left.val != right.val; }
inline bool operator<(const MLFloat16& left, const MLFloat16& right) { return left.val < right.val; }

// BFloat16
struct BFloat16 {
  uint16_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) : val(bits) {}

  inline ORT_HOST_DEVICE BFloat16(float v) {
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
    if constexpr(endian::native == endian::little) {
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
#elif defined(__HIP__)
    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
    float result = 0;
    uint32_t tmp = val;
    tmp <<= 16;
    float* tempRes = reinterpret_cast<float*>(&tmp);
    result = *tempRes;
    return result;
#else
    float result;
    char* const first = reinterpret_cast<char*>(&result);
    char* const second = first + sizeof(uint16_t);
    if constexpr(endian::native == endian::little) {
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

inline ORT_HOST_DEVICE bool operator==(const BFloat16& left, const BFloat16& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const BFloat16& left, const BFloat16& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const BFloat16& left, const BFloat16& right) { return left.val < right.val; }


// User defined suffixes to make it easier to declare
// initializers with MLFloat16 and BFloat16 from unsigned short
// E.g 10_f16 or 10_b16
#if !defined(__CUDACC__) && !defined(__HIPCC__)
inline MLFloat16 operator"" _f16(unsigned long long int v) {
  return MLFloat16(narrow<uint16_t>(v));
}

inline MLFloat16 operator"" _fp16(long double v) {
  return MLFloat16(static_cast<float>(v));
}

inline BFloat16 operator"" _b16(unsigned long long int v) {
  return BFloat16(narrow<uint16_t>(v), BFloat16::FromBits());
}

inline BFloat16 operator"" _bfp16(long double v) {
  return BFloat16(static_cast<float>(v));
}

#endif

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

}  // namespace onnxruntime
