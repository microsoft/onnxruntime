// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_FLOAT4_TYPES)

#ifdef USE_CUDA
#include<cuda.h>
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

#include <gsl/gsl>

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
        // TODO(hasesh): Add support for fp4 type creation on CPU when really needed.
        // For now, the usage of these types is heavily CUDA-centric.
        ORT_UNUSED_PARAMETER(f1);
        ORT_UNUSED_PARAMETER(f2);
        ORT_ENFORCE(false, "Creation of float4 types requires CUDA enabled builds with CUDA version >= 12.8"); 
    #endif
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  inline explicit ORT_HOST_DEVICE Float4E2M1x2(const __nv_fp4x2_e2m1& value) { val_ = *reinterpret_cast<const unsigned char*>(&value); }
  inline explicit ORT_HOST_DEVICE operator __nv_fp4x2_e2m1() const { return *reinterpret_cast<const __nv_fp4x2_e2m1*>(&val_); }
  inline ORT_HOST_DEVICE float2 ToCudaFloat2() const {
    return __half22float2(__nv_cvt_fp4x2_to_halfraw2(static_cast<PackedCudaStorageType>(val_), __NV_E2M1));
  }
#endif

  inline ORT_HOST_DEVICE std::pair<float, float> ToFloat2() const {
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
         float2 temp = ToCudaFloat2();
         return std::make_pair(temp.x, temp.y);
    #else
        ORT_ENFORCE(false, "Conversion to floats from float4 types requires CUDA enabled builds with CUDA version >= 12.8");
    #endif
  }

  inline ORT_HOST_DEVICE uint8_t ToBits() const {
    return val_;
  }

  static size_t CalcNumFloat4Pairs(size_t num_float4_elems) {
    return (num_float4_elems + 1) / 2;
  }

  inline void UnpackFloat4E2M1ToFloat(const Float4E2M1x2* fp4x2_arr, UnpackedType* flt_arr, size_t size) {
    auto src = fp4x2_arr;
    auto dst = flt_arr;

    size_t dst_i = 0;

    for (; dst_i < size - 1; dst_i += 2) {
      auto src_i = dst_i >> 2;
      auto flt_pair = src[src_i].ToFloat2();
      dst[dst_i] = flt_pair.first;
      dst[dst_i + 1] = flt_pair.second;

    }

    if (dst_i < size) {
      auto src_i = dst_i >> 2;
      dst[dst_i] = fp4x2_arr[src_i].ToFloat2().first;
    }
  }

  inline void PackFloatToFloat4E2M1(const UnpackedType* flt_arr, Float4E2M1x2* fp4x2_arr, size_t size) {
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

  /*
  inline ORT_HOST_DEVICE UnpackedType GetElem(size_t index) const {
    ORT_UNUSED_PARAMETER(index);
    return 0.f;
  }

  inline ORT_HOST_DEVICE void SetElem(size_t index, UnpackedType val) {
     assert(index <= 1);
  }


    static ORT_HOST_DEVICE size_t CalcNumFloat4Pairs(size_t num_float4_elems) {
      return (num_float4_elems + 1) / 2;
    }


    static bool Unpack(gsl::span<UnpackedType> dst, gsl::span<const Float4E2M1x2> src) {
      if (CalcNumFloat4Pairs(dst.size()) != src.size()) {
        return false;
      }

      if (src.empty()) {
        return true;
      }

      for (size_t i = 0; i < dst.size(); i++) {
        size_t r = i >> 1;   // i / 2;
        size_t c = i & 0x1;  // i % 2;
        dst[i] = src[r].GetElem(c);
      }

      return true;
    }

      static bool Pack(gsl::span<Float4E2M1x2> dst, gsl::span<const UnpackedType> src) {
      if (CalcNumFloat4Pairs(src.size()) != dst.size()) {
        return false;
      }

      if (src.empty()) {
        return true;
      }

      size_t src_i = 0;
      size_t dst_i = 0;

      for (; src_i < src.size() - 1; src_i += 2) {
        dst[dst_i++] = Float4E2M1x2(src[src_i], src[src_i + 1]);
      }

      if (src_i < src.size()) {
        dst[dst_i] = Float4E2M1x2(src[src_i], 0);
      }

      return true;
    }
*/
};

inline ORT_HOST_DEVICE bool operator==(const Float4E2M1x2& left, const Float4E2M1x2& right) { return left.val_ == right.val_; }
inline ORT_HOST_DEVICE bool operator!=(const Float4E2M1x2& left, const Float4E2M1x2& right) { return left.val_ != right.val_; }
inline ORT_HOST_DEVICE bool operator<(const Float4E2M1x2& left, const Float4E2M1x2& right) { return left.val_ < right.val_; }


static_assert(sizeof(Float4E2M1x2) == sizeof(uint8_t));
}  // namespace onnxruntime


namespace std
{
  /*
    // TODO (hasesh): Does numeric_limits make sense for packed types ?
    // For now, produce limits of each element in a packed format, refine
    // this based on usage later
    template <>
    class numeric_limits<onnxruntime::Float4E2M1x2> {
     public:
      static constexpr onnxruntime::Float4E2M1x2 lowest() {
      }

      static constexpr onnxruntime::Float4E2M1x2 max() {

      }

      static constexpr onnxruntime::Float4E2M1x2 min() {

      }

      static constexpr onnxruntime::Float4E2M1x2 denorm_min() {
      }

      static constexpr onnxruntime::Float4E2M1x2 epsilon() {
      }

      static constexpr onnxruntime::Float4E2M1x2 round_error() {
      }

      static constexpr onnxruntime::Float4E2M1x2 infinity() {
      }

      static constexpr onnxruntime::Float4E2M1x2 quiet_NaN() {
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
      static constexpr int digits = 3;
      static constexpr int digits10 = 0;
      static constexpr int max_digits10 = 2;
      static constexpr int radix = 2;
      static constexpr int min_exponent = -14;
      static constexpr int min_exponent10 = -4;
      static constexpr int max_exponent = 16;
      static constexpr int max_exponent10 = 4;
      static constexpr auto traps = false;
      static constexpr auto tinyness_before = false;
    };
   */

}  // namespace std

#endif  // DISABLE_FLOAT4_TYPES
