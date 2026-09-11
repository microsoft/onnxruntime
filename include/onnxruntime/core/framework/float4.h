// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// IMPORTANT NOTE: Users of this file MUST include "cuda.h" before including this header
// if they would like to leverage the CUDA implementation for the conversion routines
// in their HOST code (code compiled by MSVC/GCC).
// This is because there is a check on CUDA_VERSION which is a macro defined in cuda.h.
// We can't include cuda.h in this header unconditionally because this header is also
// included in core framework files which are CUDA-agnostic.
// Not including "cuda.h" in GCC/MSVC will fall-back to the CPU conversion routines
// implemented in this file.
// For code compiled by NVCC which includes this header, this file will automatically
// include cuda.h (based on the CUDA_CC macro).

#pragma once

#if !defined(DISABLE_FLOAT4_TYPES)

#if defined(__CUDACC__)
// Needed for CUDA_VERSION check below
#include <cuda.h>
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

#if defined(_MSC_VER)
#pragma warning(push)
// 'fp4_interpretation' : unreferenced parameter
#pragma warning(disable : 4100)
#endif

#include <cuda_fp4.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

#include <cassert>
#include <cmath>
#include <gsl/gsl>
#include <utility>

#include "core/common/common.h"

namespace onnxruntime {

#if defined(__CUDACC__)
#define ORT_HOST_DEVICE __host__ __device__
#else
#define ORT_HOST_DEVICE
#endif

struct Float4E2M1x2 {
  uint8_t val_{0};
  using UnpackedType = float;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  using PackedCudaType = __nv_fp4x2_e2m1;
  using PackedCudaStorageType = __nv_fp4x2_storage_t;
#endif

 private:
  ORT_HOST_DEVICE UnpackedType Fp4ToFloatConversionCpuHelper(uint8_t fp4x2, size_t shift) const {
    assert(shift == 0 || shift == 4);

    constexpr uint8_t sign_bitmask = 0x08;
    constexpr uint8_t exponent_bitmask = 0x06;
    constexpr uint8_t mantissa_bitmask = 0x01;

    uint8_t bits_shifted = (fp4x2 >> shift);

    float sign = 1.f;
    if (bits_shifted & sign_bitmask) {
      sign = -1.f;
    }

    int exponent = static_cast<int>((bits_shifted & exponent_bitmask) >> 1);
    float mantissa = static_cast<float>(bits_shifted & mantissa_bitmask);

    return (exponent == 0) ? (sign * (mantissa / 2.f)) : (sign * (1.f + mantissa / 2.f) * static_cast<float>(1 << (exponent - 1)));
  }

  ORT_HOST_DEVICE uint8_t FloatToFp4ConversionCpuHelper(float f, size_t shift) const {
    assert(shift == 0 || shift == 4);

    constexpr uint32_t sign_bitmask = 0x80000000;
    constexpr uint32_t exponent_bitmask = 0x7F800000;
    constexpr uint32_t mantissa_bitmask = 0x007FFFFF;
    constexpr uint32_t zero = 0x00000000;

    uint8_t res = 0;

    uint32_t float_bits = 0;
    std::memcpy(&float_bits, &f, sizeof(f));

    // NaN always maps to +6 (irrespective of sign)
    // https://github.com/onnx/onnx/blob/main/docs/docsgen/source/technical/float4.md
    if (((float_bits & exponent_bitmask) == exponent_bitmask) && (float_bits & mantissa_bitmask)) {
      return (0x07 << shift);
    }

    if (float_bits & sign_bitmask) {
      res = 0x08;
    }

    // Infinity is sign preserving - magnitude is 6
    if (((float_bits & exponent_bitmask) == exponent_bitmask) && ((float_bits & mantissa_bitmask) == zero)) {
      return ((res | 0x07) << shift);
    }

    float f_abs = std::abs(f);
    if (f_abs > 0.25 && f_abs < 0.75) {
      res |= 0x01;
    } else if (f_abs >= 0.75 && f_abs <= 1.25) {
      res |= 0x02;
    } else if (f_abs > 1.25 && f_abs < 1.75) {
      res |= 0x03;
    } else if (f_abs >= 1.75 && f_abs <= 2.5) {
      res |= 0x04;
    } else if (f_abs > 2.5 && f_abs < 3.5) {
      res |= 0x05;
    } else if (f_abs >= 3.5 && f_abs <= 5.0) {
      res |= 0x06;
    } else if (f_abs > 5.0) {
      res |= 0x07;
    }

    return res << shift;
  }

 public:
  Float4E2M1x2() = default;

  struct FromBitsT {};
  static constexpr ORT_HOST_DEVICE FromBitsT FromBits() { return FromBitsT(); }
  constexpr ORT_HOST_DEVICE Float4E2M1x2(unsigned char bits, FromBitsT) : val_(bits) {}

  inline explicit ORT_HOST_DEVICE Float4E2M1x2(UnpackedType f1, UnpackedType f2) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    float2 temp;
    temp.x = f1;
    temp.y = f2;

    // Converts input vector of two single precision numbers packed in float2 x
    // into a vector of two values of fp4 type of the requested kind using specified
    // rounding mode and saturating the out-of-range values.
    val_ = __nv_cvt_float2_to_fp4x2(temp, __NV_E2M1, cudaRoundNearest);
#else
    val_ = (FloatToFp4ConversionCpuHelper(f1, 0) | FloatToFp4ConversionCpuHelper(f2, 4));
#endif
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  inline explicit ORT_HOST_DEVICE Float4E2M1x2(float2 f2) {
    val_ = __nv_cvt_float2_to_fp4x2(f2, __NV_E2M1, cudaRoundNearest);
  }

  inline explicit ORT_HOST_DEVICE Float4E2M1x2(const __nv_fp4x2_e2m1& value) {
    val_ = *reinterpret_cast<const unsigned char*>(&value);
  }

  inline explicit ORT_HOST_DEVICE operator __nv_fp4x2_e2m1() const {
    return *reinterpret_cast<const __nv_fp4x2_e2m1*>(&val_);
  }

  inline ORT_HOST_DEVICE float2 ToCudaFloat2() const {
    return __half22float2(__nv_cvt_fp4x2_to_halfraw2(static_cast<PackedCudaStorageType>(val_), __NV_E2M1));
  }
#endif

  inline ORT_HOST_DEVICE std::pair<float, float> ToFloat2() const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    float2 temp = ToCudaFloat2();
    return std::make_pair(temp.x, temp.y);
#else
    return std::make_pair(Fp4ToFloatConversionCpuHelper(val_, 0), Fp4ToFloatConversionCpuHelper(val_, 4));
#endif
  }

  inline ORT_HOST_DEVICE uint8_t ToBits() const {
    return val_;
  }

  static size_t CalcNumFloat4Pairs(size_t num_float4_elems) {
    return (num_float4_elems + 1) / 2;
  }

  static void UnpackFloat4E2M1ToFloat(const Float4E2M1x2* fp4x2_arr,
                                      UnpackedType* flt_arr, size_t size) {
    auto src = fp4x2_arr;
    auto dst = flt_arr;

    size_t dst_i = 0;

    for (; dst_i < size - 1; dst_i += 2) {
      auto src_i = dst_i >> 1;
      auto flt_pair = src[src_i].ToFloat2();
      dst[dst_i] = flt_pair.first;
      dst[dst_i + 1] = flt_pair.second;
    }

    if (dst_i < size) {
      auto src_i = dst_i >> 1;
      dst[dst_i] = fp4x2_arr[src_i].ToFloat2().first;
    }
  }

  static void PackFloatToFloat4E2M1(const UnpackedType* flt_arr,
                                    Float4E2M1x2* fp4x2_arr, size_t size) {
    auto src = flt_arr;
    auto dst = fp4x2_arr;

    size_t src_i = 0;

    for (; src_i < size - 1; src_i += 2) {
      new (dst) Float4E2M1x2(src[src_i], src[src_i + 1]);
      ++dst;
    }

    if (src_i < size) {
      new (dst) Float4E2M1x2(src[src_i], 0);
    }
  }

  static inline std::pair<size_t, size_t> GetTensorElemIndices(size_t index) {
    return {index >> 1, index & 0x1};
  }

  inline UnpackedType GetElem(size_t index) const {
    assert(index <= 1);
    auto pair = ToFloat2();
    if (index == 0) {
      return static_cast<UnpackedType>(pair.first);
    }

    return static_cast<UnpackedType>(pair.second);
  }
};

inline ORT_HOST_DEVICE bool operator==(const Float4E2M1x2& left, const Float4E2M1x2& right) { return left.val_ == right.val_; }
inline ORT_HOST_DEVICE bool operator!=(const Float4E2M1x2& left, const Float4E2M1x2& right) { return left.val_ != right.val_; }

static_assert(sizeof(Float4E2M1x2) == sizeof(uint8_t));
}  // namespace onnxruntime

namespace std {
// TODO (hasesh): Does numeric_limits make sense for packed types ?
// For now, produce limits of each element in a packed format, refine
// this based on usage later
template <>
class numeric_limits<onnxruntime::Float4E2M1x2> {
 public:
  static constexpr onnxruntime::Float4E2M1x2 lowest() {
    return onnxruntime::Float4E2M1x2(0xFF, onnxruntime::Float4E2M1x2::FromBits());  // -6.0
  }

  static constexpr onnxruntime::Float4E2M1x2 max() {
    return onnxruntime::Float4E2M1x2(0x77, onnxruntime::Float4E2M1x2::FromBits());  // +6.0
  }

  static constexpr onnxruntime::Float4E2M1x2 min() {
    return onnxruntime::Float4E2M1x2(0x22, onnxruntime::Float4E2M1x2::FromBits());  // +1.0
  }

  static constexpr onnxruntime::Float4E2M1x2 denorm_min() {
    return onnxruntime::Float4E2M1x2(0x11, onnxruntime::Float4E2M1x2::FromBits());  // +0.5
  }

  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = round_to_nearest;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 2;        // (1 mantissa bit + 1 implicit bit)
  static constexpr int digits10 = 0;      // (digits -1) * std::log10(2) rounded down
  static constexpr int max_digits10 = 1;  // Mantissa bits
  static constexpr int radix = 2;
  static constexpr int min_exponent = 1;    // 2 ^ (1-1) = 1 is the valid normalized value min ceiling we can reach
  static constexpr int min_exponent10 = 0;  // 10 ^ 0 is the valid normalized value min ceiling we can reach
  static constexpr int max_exponent = 3;    // 2 ^ (3-1) = 4 is valid normalized value max ceiling we can reach
  static constexpr int max_exponent10 = 0;  // 10 ^ 0 is the valid normalized value max ceiling we can reach
  static constexpr auto traps = false;
  static constexpr auto tinyness_before = false;
};
}  // namespace std

#endif  // DISABLE_FLOAT4_TYPES
