// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ROCm equivalent of cuda/cuda_type_conversion.h

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <type_traits>
#include <cstdint>
#include "core/framework/int4.h"
#include "core/common/float8.h"
#include "core/common/float16.h"

namespace onnxruntime {
namespace rocm {

// Type mapping for ORT Type to ROCm/HIP type
template <typename T>
struct OrtToRocmType {
  using type = T;

  static type FromFloat(float f) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    return static_cast<T>(f);
  }
};

template <>
struct OrtToRocmType<Int4x2> {
  using type = int8_t;
};

template <>
struct OrtToRocmType<UInt4x2> {
  using type = uint8_t;
};

template <>
struct OrtToRocmType<MLFloat16> {
  using type = __half;
  static type FromFloat(float f) {
    return type(f);
  }
};

template <>
struct OrtToRocmType<BFloat16> {
  using type = hip_bfloat16;
  static type FromFloat(float f) {
    return type(f);
  }
};

// Also provide OrtToCudaType as an alias for compatibility with hipified code
// that calls cuda::OrtToCudaType<T> (hipify renames cuda→rocm in namespaces
// but keeps struct name; some files use the alias directly).
template <typename T>
using OrtToCudaType = OrtToRocmType<T>;

// Hipify also renames OrtToCudaType → OrtToHipType in namespace rocm, so
// provide that alias as well.
template <typename T>
using OrtToHipType = OrtToRocmType<T>;

}  // namespace rocm
}  // namespace onnxruntime
