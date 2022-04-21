// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace xnnpack {
inline bool IsAllDimKnown(const TensorShape& s) {
  size_t len = s.NumDimensions();
  for (size_t i = 0; i != len; ++i) {
    if (s[i] < 0) return false;
  }
  return true;
}
}  // namespace xnnpack
}  // namespace onnxruntime