// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "endian.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
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

// MLFloatE4M3
struct MLFloatE4M3 {
  uint8_t val{0};

  MLFloatE4M3() = default;
  explicit constexpr MLFloatE4M3(uint8_t x) : val(x) {}
  explicit MLFloatE4M3(float f);

  float ToFloat() const;

  operator float() const { return ToFloat(); }
};

inline bool operator==(const MLFloatE4M3& left, const MLFloatE4M3& right) { return left.val == right.val; }
inline bool operator!=(const MLFloatE4M3& left, const MLFloatE4M3& right) { return left.val != right.val; }
inline bool operator<(const MLFloatE4M3& left, const MLFloatE4M3& right) { return left.val < right.val; }

// MLFloatE5M2
struct MLFloatE5M2 {
  uint8_t val{0};

  MLFloatE5M2() = default;
  explicit constexpr MLFloatE5M2(uint8_t x) : val(x) {}
  explicit MLFloatE5M2(float f);

  float ToFloat() const;

  operator float() const { return ToFloat(); }
};

inline bool operator==(const MLFloatE5M2& left, const MLFloatE5M2& right) { return left.val == right.val; }
inline bool operator!=(const MLFloatE5M2& left, const MLFloatE5M2& right) { return left.val != right.val; }
inline bool operator<(const MLFloatE5M2& left, const MLFloatE5M2& right) { return left.val < right.val; }

// FloatE4M3
struct FloatE4M3 {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE FloatE4M3() = default;
#else
  FloatE4M3() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE FloatE4M3(unsigned char bits, FromBitsT) : val(bits) {}

  inline ORT_HOST_DEVICE FloatE4M3(float v) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    val = __nv_cvt_float_to_fp8(v);
#else
    uint32_t* pv = reinterpret_cast<uint32_t*>(&v);
    uint32_t b = *pv;

    val = static_cast<uint8_t>((b & 0x80000000) >> 24);  // sign
    if ((b & 0x7fc00000) == 0x7fc00000) {
      val |= 0xff;
    } else {
      uint8_t e = static_cast<uint8_t>((b & 0x7F800000) >> 23);  // exponent
      uint32_t m = static_cast<uint32_t>(b & 0x007FFFFF);        // mantissa
      if (e != 0) {
        if (e < 117) { // 0b1110101
        } else if (e < 118) { // 0b1110110
          val |= 1;
          if ((m >> 23) & 1) {
            // rounding
            val += 1;
          }
        } else if (e < 121) {  // 127 - 7 + 1 // 0b1111001
          auto d = 120 - e;  // 0b1111000
          val |= 1 << (2 - d);
          val |= m >> (21 + d);
          if ((m >> (20 + d)) & 1) {
            // rounding
            val += 1;
          }
        } else if (e < 136) {  // 127 + 8 + 1 // 0b10001000
          auto ex = e - 120;   // 127 - 7
          if (ex == 0) {
            val |= 0x4;
            val |= m >> 21;
          } else {
            val |= ex << 3;
            val |= m >> 20;
          }
          if (m & 0x80000) {
            // rounding
            val += 1;
          }
        } else {
          val |= 126;  // 0b01111110
        }
      }
    }
#endif
  }

  inline ORT_HOST_DEVICE float ToFloat() const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    return __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3));
#else
    uint32_t res;
    if (val == 255) {
      res = 0xffc00000;
    } else if (val == 127) {
      res = 0x7fc00000;
    } else {
      uint32_t expo = (val & 0x78) >> 3;
      uint32_t mant = val & 0x07;
      uint32_t sign = val & 0x80;
      res = sign << 24;
      if (expo == 0) {
        if (mant > 0) {
          expo = 0x7F - 7;
          if ((mant & 0x4) == 0) {
            mant &= 0x3;
            mant <<= 1;
            expo -= 1;
          }
          if ((mant & 0x4) == 0) {
            mant &= 0x3;
            mant <<= 1;
            expo -= 1;
          }
          if ((mant & 0x4) == 0) {
            mant &= 0x3;
            mant <<= 1;
            expo -= 1;
          }
          res |= (mant & 0x3) << 21;
          res |= expo << 23;
        }
      } else {
        res |= mant << 20;
        expo -= 0x7;
        expo += 0x7F;
        res |= expo << 23;
      }
    }
    float float_res;
    std::memcpy(&float_res, &res, sizeof(float));
    return float_res;
#endif
  }

  inline ORT_HOST_DEVICE operator float() const { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  ORT_HOST_DEVICE FloatE4M3(const __nv_fp8_e4m3& value) { val = *reinterpret_cast<const unsigned char*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_fp8_e4m3() const { return *reinterpret_cast<const __nv_fp8_e4m3*>(&val); }
#endif
};

inline ORT_HOST_DEVICE bool operator==(const FloatE4M3& left, const FloatE4M3& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const FloatE4M3& left, const FloatE4M3& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const FloatE4M3& left, const FloatE4M3& right) { return left.val < right.val; }


// User defined suffixes to make it easier to declare
// initializers with MLFloatE4M3 and FloatE4M3 from unsigned char
// E.g 10_f16 or 10_b16
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline FloatE4M3 operator"" _fe4m3(unsigned long long int v) {
  return FloatE4M3(narrow<uint8_t>(v), FloatE4M3::FromBits());
}

inline FloatE4M3 operator"" _fe4m3p8(long double v) {
  return FloatE4M3(static_cast<float>(v));
}

#endif

inline void FloatE4M3ToFloat(const FloatE4M3* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloatE4M3(const float* flt, FloatE4M3* blf, size_t size) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) FloatE4M3(*src);
  }
}

// FloatE5M2
struct FloatE5M2 {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE FloatE5M2() = default;
#else
  FloatE5M2() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE FloatE5M2(unsigned char bits, FromBitsT) : val(bits) {}

  inline ORT_HOST_DEVICE FloatE5M2(float v) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    val = __fe5m2_as_uchar(__float2fe5m2(v));
#else
    uint32_t* pv = reinterpret_cast<uint32_t*>(&v);
    uint32_t b = *pv;

    val = (b & 0x80000000) >> 24;           // sign
    if ((b & 0x7fc00000) == 0x7fc00000) {
      val |= 0xff;
    } else {
      uint32_t e = (b & 0x7F800000) >> 23;  // exponent
      uint32_t m = b & 0x007FFFFF;          // mantissa

      if (e != 0) {
        if (e < 110) {
        } else if (e < 111) {
          val |= 1;
          if ((m >> 23) & 1) {
            // rounding
            val += 1;
          }
        } else if (e < 113) {  // 127 - 15 + 1
          auto d = 112 - e;
          val |= 1 << (1 - d);
          val |= m >> (22 + d);
          if ((m >> (21 + d)) & 1) {
            // rounding
            val += 1;
          }
        } else if (e < 144) {  // 127 + 16 + 1
          auto ex = e - 112;   // 127 - 15
          val |= ex << 2;
          val |= m >> 21;
          if (m & 0x100000) {
            // rounding
            val += 1;
          }
        } else if ((e == 255) && (m == 0)) {  // inf
          val |= 124;
        } else {
          val |= 123;
        }
      }
    }
#endif
  }

  inline ORT_HOST_DEVICE float ToFloat() const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    return __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2));
#else
    uint32_t res;
    if (val >= 253) {
      res = 0xffc00000;
    } else if (val >= 125 && val <= 127) {
      res = 0x7fc00000;
    } else if (val == 252) {
      res = 0xff800000;
    } else if (val == 124) {
      res = 0x7f800000;
    } else {
      uint32_t expo = (val & 0x7C) >> 2;
      uint32_t mant = val & 0x03;
      uint32_t sign = val & 0x80;
      res = sign << 24;
      if (expo == 0) {
        if (mant > 0) {
          expo = 0x7F - 15;
          if ((mant & 0x2) == 0) {
            mant &= 0x1;
            mant <<= 1;
            expo -= 1;
          }
          if ((mant & 0x2) == 0) {
            mant &= 0x1;
            mant <<= 1;
            expo -= 1;
          }
          res |= (mant & 0x1) << 22;
          res |= expo << 23;
        }
      } else {
        res |= mant << 21;
        expo -= 15;
        expo += 0x7F;
        res |= expo << 23;
      }
    }

    float float_res;
    std::memcpy(&float_res, &res, sizeof(float));
    return float_res;
#endif
  }

  inline ORT_HOST_DEVICE operator float() const { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  ORT_HOST_DEVICE FloatE5M2(const __nv_fp8_e5m2& value) { val = *reinterpret_cast<const unsigned char*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_fp8_e5m2() const { return *reinterpret_cast<const __nv_fp8_e5m2*>(&val); }
#endif
};

inline ORT_HOST_DEVICE bool operator==(const FloatE5M2& left, const FloatE5M2& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const FloatE5M2& left, const FloatE5M2& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const FloatE5M2& left, const FloatE5M2& right) { return left.val < right.val; }

// User defined suffixes to make it easier to declare
// initializers with MLFloatE5M2 and FloatE5M2 from unsigned char
// E.g 10_f16 or 10_b16
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline FloatE5M2 operator"" _fe5m2(unsigned long long int v) {
  return FloatE5M2(narrow<uint8_t>(v), FloatE5M2::FromBits());
}

inline FloatE5M2 operator"" _fe5m2p8(long double v) {
  return FloatE5M2(static_cast<float>(v));
}

#endif

inline void FloatE5M2ToFloat(const FloatE5M2* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloatE5M2(const float* flt, FloatE5M2* blf, size_t size) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) FloatE5M2(*src);
  }
}

}  // namespace onnxruntime
