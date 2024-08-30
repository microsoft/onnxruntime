// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_FLOAT8_TYPES)

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

// Float8E4M3FN
struct Float8E4M3FN {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE Float8E4M3FN() = default;
#else
  Float8E4M3FN() = default;
#endif
  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE Float8E4M3FN(unsigned char bits, FromBitsT) : val(bits) {}

  inline explicit ORT_HOST_DEVICE Float8E4M3FN(float v, bool saturate = true) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    val = __nv_cvt_float_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3);
#else
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));

    val = static_cast<uint8_t>((b & 0x80000000) >> 24);  // sign
    if ((b & 0x7fffffff) == 0x7f800000) {                // infinity
      if (saturate) {
        val |= 126;
      } else {
        val |= 0x7f;
      }
    } else if ((b & 0x7F800000) == 0x7F800000) {  // NaN
      val |= 0x7f;
    } else {
      uint8_t e = static_cast<uint8_t>((b & 0x7F800000) >> 23);  // exponent
      uint32_t m = static_cast<uint32_t>(b & 0x007FFFFF);        // mantissa
      if (e != 0) {
        if (e < 117) {
        } else if (e < 121) {
          // denormalized number
          auto d = 120 - e;
          if (d < 3) {
            val |= 1 << (2 - d);
            val |= m >> (21 + d);
          } else if (m > 0) {
            val |= 1;
          }
          auto mask = 1 << (20 + d);
          if ((m & mask) && ((val & 1) || ((m & (mask - 1)) > 0) || ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
            // rounding
            val += 1;
          }
        } else if (e < 136) {
          // normalized number
          auto ex = e - 120;
          if (ex == 0) {
            val |= 0x4;
            val |= m >> 21;
          } else {
            val |= ex << 3;
            val |= m >> 20;
            if ((val & 0x7F) == 0x7F) {
              val &= 0xFE;
            }
          }
          if ((m & 0x80000) && ((m & 0x100000) || (m & 0x7FFFF))) {
            if ((val & 0x7F) < 0x7E) {
              // rounding
              val += 1;
            } else if (!saturate) {
              val |= 0x7F;
            }
          }
        } else if (saturate) {
          val |= 126;  // 0b01111110
        } else {
          val |= 0x7F;
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
  explicit ORT_HOST_DEVICE Float8E4M3FN(const __nv_fp8_e4m3& value) { val = *reinterpret_cast<const unsigned char*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_fp8_e4m3() const { return *reinterpret_cast<const __nv_fp8_e4m3*>(&val); }
#endif
};

inline ORT_HOST_DEVICE bool operator==(const Float8E4M3FN& left, const Float8E4M3FN& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const Float8E4M3FN& left, const Float8E4M3FN& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const Float8E4M3FN& left, const Float8E4M3FN& right) { return left.val < right.val; }

// User defined suffixes to make it easier to declare
// initializers with MLFloat8E4M3FN and Float8E4M3FN from unsigned char
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline Float8E4M3FN operator"" _f8e4m3fn(unsigned long long int v) {
  return Float8E4M3FN(narrow<uint8_t>(v), Float8E4M3FN::FromBits());
}

inline Float8E4M3FN operator"" _f8e4m3fnp8(long double v) {
  return Float8E4M3FN(static_cast<float>(v), true);
}

#endif

inline void Float8E4M3FNToFloat(const Float8E4M3FN* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloat8E4M3FN(const float* flt, Float8E4M3FN* blf, size_t size, bool saturate) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) Float8E4M3FN(*src, saturate);
  }
}

// Float8E4M3FNUZ
struct Float8E4M3FNUZ {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE Float8E4M3FNUZ() = default;
#else
  Float8E4M3FNUZ() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE Float8E4M3FNUZ(unsigned char bits, FromBitsT) : val(bits) {}

  inline explicit ORT_HOST_DEVICE Float8E4M3FNUZ(float v, bool saturate = true) {
    // This type does not exist on CUDA.
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));

    val = static_cast<uint8_t>((b & 0x80000000) >> 24);  // sign
    if ((b & 0x7fffffff) == 0x7f800000) {                // infinity
      if (saturate) {
        // the highest available value
        val |= 0x7F;
      } else {
        // NaN
        val = 0x80;
      }
    } else if ((b & 0x7F800000) == 0x7F800000) {  // NaN
      val = 0x80;
    } else {
      uint8_t e = static_cast<uint8_t>((b & 0x7F800000) >> 23);  // exponent
      uint32_t m = static_cast<uint32_t>(b & 0x007FFFFF);        // mantissa
      if (e != 0) {
        if (e < 116) {
        } else if (e < 120) {
          // denormalized number
          auto d = 119 - e;
          if (d < 3) {
            val |= 1 << (2 - d);
            val |= m >> (21 + d);
          } else if (m > 0) {
            val |= 1;
          }
          auto mask = 1 << (20 + d);
          if ((m & mask) && ((val & 1) || ((m & (mask - 1)) > 0) || ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
            // rounding
            val += 1;
          }
        } else if (e < 135) {
          // normalized number
          auto ex = e - 119;
          if (ex == 0) {
            val |= 0x4;
            val |= m >> 21;
          } else {
            val |= ex << 3;
            val |= m >> 20;
          }
          if ((m & 0x80000) && ((m & 0x100000) || (m & 0x7FFFF))) {
            if ((val & 0x7F) < 0x7F) {
              // rounding
              val += 1;
            } else if (!saturate) {
              val = 0x80;
            }
          }
        } else if (saturate) {
          val |= 0x7F;
        } else {
          val = 0x80;
        }
      } else if (m == 0) {
        // -0
        val = 0;
      }
    }
  }

  inline ORT_HOST_DEVICE float ToFloat() const {
    // This type does not exist on CUDA.
    uint32_t res;
    if (val == 0x80) {
      res = 0xffc00000;
    } else {
      uint32_t expo = (val & 0x78) >> 3;
      uint32_t mant = val & 0x07;
      uint32_t sign = val & 0x80;
      res = sign << 24;
      if (expo == 0) {
        if (mant > 0) {
          expo = 0x7F - 8;
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
        expo -= 8;
        expo += 0x7F;
        res |= expo << 23;
      }
    }
    float float_res;
    std::memcpy(&float_res, &res, sizeof(float));
    return float_res;
  }

  inline ORT_HOST_DEVICE operator float() const { return ToFloat(); }
};

inline ORT_HOST_DEVICE bool operator==(const Float8E4M3FNUZ& left, const Float8E4M3FNUZ& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const Float8E4M3FNUZ& left, const Float8E4M3FNUZ& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const Float8E4M3FNUZ& left, const Float8E4M3FNUZ& right) { return left.val < right.val; }

// User defined suffixes to make it easier to declare
// initializers with MLFloat8E4M3FN and Float8E4M3FN from unsigned char
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline Float8E4M3FNUZ operator"" _f8e4m3p8fnuz(unsigned long long int v) {
  return Float8E4M3FNUZ(narrow<uint8_t>(v), Float8E4M3FNUZ::FromBits());
}

inline Float8E4M3FNUZ operator"" _f8e4m3fnuzp8(long double v) {
  return Float8E4M3FNUZ(static_cast<float>(v), true);
}

#endif

inline void Float8E4M3FNUZToFloat(const Float8E4M3FNUZ* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloat8E4M3FNUZ(const float* flt, Float8E4M3FNUZ* blf, size_t size, bool saturate) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) Float8E4M3FNUZ(*src, saturate);
  }
}

// Float8E5M2
struct Float8E5M2 {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE Float8E5M2() = default;
#else
  Float8E5M2() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE Float8E5M2(unsigned char bits, FromBitsT) : val(bits) {}

  inline explicit ORT_HOST_DEVICE Float8E5M2(float v, bool saturate = true) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    val = __nv_cvt_float_to_fp8(v, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E5M2);
#else
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));

    val = (b & 0x80000000) >> 24;          // sign
    if ((b & 0x7FFFFFFF) == 0x7F800000) {  // inf
      if (saturate) {
        // the highest available value
        val |= 0x7B;
      } else {
        // the infinity
        val |= 0x7C;
      }
    } else if ((b & 0x7F800000) == 0x7F800000) {  // NaN
      val |= 0x7f;
    } else {
      uint32_t e = (b & 0x7F800000) >> 23;  // exponent
      uint32_t m = b & 0x007FFFFF;          // mantissa

      if (e != 0) {
        if (e < 110) {
        } else if (e < 113) {
          // denormalized number
          auto d = 112 - e;
          if (d < 2) {
            val |= 1 << (1 - d);
            val |= m >> (22 + d);
          } else if (m > 0) {
            val |= 1;
          }
          auto mask = 1 << (21 + d);
          if ((m & mask) && ((val & 1) || ((m & (mask - 1)) > 0) || ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
            // rounding
            val += 1;
          }
        } else if (e < 143) {  // 127 + 15 + 1
          auto ex = e - 112;   // 127 - 15
          val |= ex << 2;
          val |= m >> 21;
          if ((m & 0x100000) && ((m & 0xFFFFF) || (m & 0x200000))) {
            if ((val & 0x7F) < 0x7B) {
              // rounding
              val += 1;
            } else if (saturate) {
              val |= 0x7B;
            } else {
              val |= 0x7C;
            }
          }
        } else if (saturate) {
          val |= 0x7B;
        } else {
          val |= 0x7C;
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
  ORT_HOST_DEVICE Float8E5M2(const __nv_fp8_e5m2& value) { val = *reinterpret_cast<const unsigned char*>(&value); }
  explicit ORT_HOST_DEVICE operator __nv_fp8_e5m2() const { return *reinterpret_cast<const __nv_fp8_e5m2*>(&val); }
#endif
};

inline ORT_HOST_DEVICE bool operator==(const Float8E5M2& left, const Float8E5M2& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const Float8E5M2& left, const Float8E5M2& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const Float8E5M2& left, const Float8E5M2& right) { return left.val < right.val; }

// User defined suffixes to make it easier to declare
// initializers with MLFloat8E5M2 and Float8E5M2 from unsigned char
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline Float8E5M2 operator"" _f8e5m2fn(unsigned long long int v) {
  return Float8E5M2(narrow<uint8_t>(v), Float8E5M2::FromBits());
}

inline Float8E5M2 operator"" _f8e5m2fnp8(long double v) {
  return Float8E5M2(static_cast<float>(v), true);
}

#endif

inline void Float8E5M2ToFloat(const Float8E5M2* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloat8E5M2(const float* flt, Float8E5M2* blf, size_t size, bool saturate) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) Float8E5M2(*src, saturate);
  }
}

// Float8E5M2FNUZ
struct Float8E5M2FNUZ {
  uint8_t val{0};
#if defined(__HIP__)
  ORT_HOST_DEVICE Float8E5M2FNUZ() = default;
#else
  Float8E5M2FNUZ() = default;
#endif

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE Float8E5M2FNUZ(unsigned char bits, FromBitsT) : val(bits) {}

  inline explicit ORT_HOST_DEVICE Float8E5M2FNUZ(float v, bool saturate = true) {
    // This type does not exist on CUDA.
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));

    val = (b & 0x80000000) >> 24;          // sign
    if ((b & 0x7FFFFFFF) == 0x7F800000) {  // inf
      if (saturate) {
        val |= 0x7F;
      } else {
        val = 0x80;
      }
    } else if ((b & 0x7F800000) == 0x7F800000) {  // NaN
      val = 0x80;
    } else {
      uint32_t e = (b & 0x7F800000) >> 23;  // exponent
      uint32_t m = b & 0x007FFFFF;          // mantissa

      if (e != 0) {
        if (e < 109) {
        } else if (e < 112) {
          // denormalized number
          auto d = 111 - e;
          if (d < 2) {
            val |= 1 << (1 - d);
            val |= m >> (22 + d);
          } else if (m > 0) {
            val |= 1;
          }
          auto mask = 1 << (21 + d);
          if ((m & mask) && ((val & 1) || ((m & (mask - 1)) > 0) || ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
            // rounding
            val += 1;
          }
        } else if (e < 143) {
          // normalized number
          auto ex = e - 111;
          val |= ex << 2;
          val |= m >> 21;
          if ((m & 0x100000) && ((m & 0xFFFFF) || (m & 0x200000))) {
            if ((val & 0x7F) < 0x7F) {
              // rounding
              val += 1;
            } else if (!saturate) {
              val = 0x80;
            }
          }
        } else if ((e == 255) && (m == 0)) {
          val = 0x80;
        } else if (saturate) {
          val |= 0x7F;
        } else {
          val = 0x80;
        }
      } else if (m == 0) {
        // -0
        val = 0;
      }
    }
  }

  inline ORT_HOST_DEVICE float ToFloat() const {
    // This type does not exist on CUDA.
    uint32_t res;
    if (val == 0x80) {
      res = 0xffc00000;
    } else {
      uint32_t expo = (val & 0x7C) >> 2;
      uint32_t mant = val & 0x03;
      uint32_t sign = val & 0x80;
      res = sign << 24;
      if (expo == 0) {
        if (mant > 0) {
          expo = 0x7F - 16;
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
        expo -= 16;
        expo += 0x7F;
        res |= expo << 23;
      }
    }

    float float_res;
    std::memcpy(&float_res, &res, sizeof(float));
    return float_res;
  }

  inline ORT_HOST_DEVICE operator float() const { return ToFloat(); }
};

inline ORT_HOST_DEVICE bool operator==(const Float8E5M2FNUZ& left, const Float8E5M2FNUZ& right) { return left.val == right.val; }
inline ORT_HOST_DEVICE bool operator!=(const Float8E5M2FNUZ& left, const Float8E5M2FNUZ& right) { return left.val != right.val; }
inline ORT_HOST_DEVICE bool operator<(const Float8E5M2FNUZ& left, const Float8E5M2FNUZ& right) { return left.val < right.val; }

// User defined suffixes to make it easier to declare
// initializers with MLFloat8E5M2 and Float8E5M2 from unsigned char
#if !defined(__CUDACC__) && !defined(__HIPCC__)

inline Float8E5M2FNUZ operator"" _f8e5m2fnuz(unsigned long long int v) {
  return Float8E5M2FNUZ(narrow<uint8_t>(v), Float8E5M2FNUZ::FromBits());
}

inline Float8E5M2FNUZ operator"" _f8e5m2fnuzp8(long double v) {
  return Float8E5M2FNUZ(static_cast<float>(v), true);
}

#endif

inline void Float8E5M2FNUZToFloat(const Float8E5M2FNUZ* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToFloat8E5M2FNUZ(const float* flt, Float8E5M2FNUZ* blf, size_t size, bool saturate) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) Float8E5M2FNUZ(*src, saturate);
  }
}

}  // namespace onnxruntime

#endif  // DISABLE_FLOAT8_TYPES
