// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace hip {

template <typename T>
void DropoutKernelImpl(
  const hipDeviceProp_t& prop,
  const int64_t N,
  const float ratio,
  PhiloxGenerator& generator,
  const T* X_data,
  T* Y_data,
  bool* mask_data);

template <typename T>
void DropoutGradientKernelImpl(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float ratio,
  T* dX_data);

}  // namespace hip
}  // namespace onnxruntime
