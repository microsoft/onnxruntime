// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
namespace onnxruntime {
namespace cuda {

template <typename T>
void DropoutKernelImpl(
  const int64_t N,
  const float ratio,
  const T* X_data,
  const float* random_data,
  T* Y_data,
  bool* mask_data);

template <typename T>
void DropoutGradientKernelImpl(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float scale,
  T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime
