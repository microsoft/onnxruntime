// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void LayerNormLinearKernel(
    const int64_t N,
    const int64_t M,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <typename T>
void LayerNormGradInternalKernel(
    const int64_t N,
    const int64_t M,
    const T* Y_grad,
    const T* X_data,
    const T* X_mean,
    const T* X_inv_std_var,
    const T* scale,
    T* A,
    T* B,
    T* C);

template <typename T>
void LayerNormGradXKernel(
    const int64_t N,
    const int64_t M,
    const T* X_data,
    const T* X_mean,
    const T* B,
    const T* mean_B,
    const T* mean_C,
    const T* X_inv_std_var,
    T* X_grad);

}  // namespace cuda
}  // namespace onnxruntime
