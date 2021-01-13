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

template<typename TLossScale, typename TGradNorm, typename TFinalScale>
__device__ __forceinline__ TFinalScale _ComputeGradScale(
const TLossScale* loss_scale,
const TGradNorm* scaled_g_norm,
const TFinalScale max_norm) {
TFinalScale scale = loss_scale != nullptr ? TFinalScale(*loss_scale) : TFinalScale(1.f);
TFinalScale scaled_max_norm = TFinalScale(scale * max_norm);
TFinalScale scaled_reciprocal_of_clipping_factor = scale;
if (scaled_g_norm != nullptr && TFinalScale(*scaled_g_norm) > scaled_max_norm) {
    scaled_reciprocal_of_clipping_factor = TFinalScale(*scaled_g_norm) / max_norm;
}
return scaled_reciprocal_of_clipping_factor;
}
}  // namespace cuda
}  // namespace onnxruntime