// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if defined(ENABLE_FP8) && !defined(DISABLE_FLOAT8_TYPES)
#include <cuda_fp8.h>
#endif
#if defined(ENABLE_FP4) && !defined(DISABLE_FLOAT4_TYPES)
#include <cuda_fp4.h>
#endif
#include <type_traits>
#include <cstdint>
#include "core/framework/int4.h"
#include "core/framework/float8.h"
#include "core/framework/float16.h"
#include "core/framework/float4.h"

namespace onnxruntime {
namespace cuda {

// Type mapping for ORT Type to CUDA Type
template <typename T>
struct OrtToCudaType {
  using type = T;

  static type FromFloat(float f) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    return static_cast<T>(f);
  }
};

template <>
struct OrtToCudaType<Int4x2> {
  using type = int8_t;
};

template <>
struct OrtToCudaType<UInt4x2> {
  using type = uint8_t;
};

template <>
struct OrtToCudaType<MLFloat16> {
  using type = __half;
  static type FromFloat(float f) {
    return type(f);
  }
};

template <>
struct OrtToCudaType<BFloat16> {
  using type = __nv_bfloat16;
  static type FromFloat(float f) {
    return type(f);
  }
};

#if defined(ENABLE_FP8) && !defined(DISABLE_FLOAT8_TYPES)
template <>
struct OrtToCudaType<Float8E4M3FN> {
  using type = __nv_fp8_e4m3;
  static type FromFloat(float f) {
    return type(f);
  }
};

template <>
struct OrtToCudaType<Float8E4M3FNUZ> {
  using type = __nv_fp8_e4m3;
  static type FromFloat(float f) {
    return type(f);
  }
};

template <>
struct OrtToCudaType<Float8E5M2> {
  using type = __nv_fp8_e5m2;
  static type FromFloat(float f) {
    return type(f);
  }
};

template <>
struct OrtToCudaType<Float8E5M2FNUZ> {
  using type = __nv_fp8_e5m2;
  static type FromFloat(float f) {
    return type(f);
  }
};
#endif

#if defined(ENABLE_FP4) && !defined(DISABLE_FLOAT4_TYPES)
template <>
struct OrtToCudaType<Float4E2M1x2> {
  using type = Float4E2M1x2::PackedCudaType;
};
#endif

}  // namespace cuda
}  // namespace onnxruntime
