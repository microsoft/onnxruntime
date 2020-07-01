// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void DropoutGradientKernelImpl(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float ratio,
  T* dX_data);

template <typename T>
void BiasDropoutKernelImpl(
  const cudaDeviceProp& prop,
  const int64_t N,
  const fast_divmod fdm_dim,
  const float ratio,
  PhiloxGenerator& generator,
  const T* X_data,
  const T* bias_data,
  const T* residual_data,
  T* Y_data,
  bool* mask_data);

}  // namespace cuda
}  // namespace onnxruntime
