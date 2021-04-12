// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace onnxruntime {
namespace cuda {

// ---------------------------------------------------------------------------
// _ComputeGradScale -- helper to calculate gradient scales based on global norms
// ---------------------------------------------------------------------------

template <typename TLossScale, typename TGradNorm, typename TFinalScale>
__device__ __forceinline__ TFinalScale _ComputeGradScale(
    const TLossScale* loss_scale,    // Scale of the gradient (called "scaled_g_norm" below)
    const TGradNorm* scaled_g_norm,  // Scaled gradient norm is an optimizer input
    const TFinalScale max_g_norm) {
  const TFinalScale scale = loss_scale != nullptr ? TFinalScale(*loss_scale) : TFinalScale(1.f);
  const TFinalScale scaled_max_g_norm = TFinalScale(scale * max_g_norm);

  // This number is used to divide the scaled gradient before applying optimizers.
  TFinalScale scaled_g_scaling_factor = scale;
  if (scaled_g_norm != nullptr && TFinalScale(*scaled_g_norm) > scaled_max_g_norm) {
    scaled_g_scaling_factor = TFinalScale(*scaled_g_norm) / max_g_norm;
  }
  return scaled_g_scaling_factor;
}
}  // namespace cuda
}  // namespace onnxruntime