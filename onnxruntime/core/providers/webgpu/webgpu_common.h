// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/float16.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
class ToWebGpuType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToWebGpuType<MLFloat16> {
 public:
  typedef MLFloat16 MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
