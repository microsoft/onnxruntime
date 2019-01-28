// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "shared_inc/cuda_call.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace cuda {

// Type mapping for MLFloat16 to half
template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToCudaType<MLFloat16> {
 public:
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

}  // namespace cuda
}  // namespace onnxruntime
