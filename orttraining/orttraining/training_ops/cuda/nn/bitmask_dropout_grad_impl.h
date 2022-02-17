// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void BitmaskDropoutGradientKernelImpl(
  const cudaDeviceProp& prop,
  cudaStream_t stream,
  const int64_t N,
  const T* dY_data,
  const uint32_t* mask_data,
  const float ratio,
  T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime
