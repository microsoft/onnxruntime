// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Maximum matrix order the warp-per-matrix kernel can handle: one lane owns one row/column sum.
constexpr int kSinkhornMaxOrder = 32;

Status LaunchSinkhornNormalize(cudaStream_t stream,
                               const float* input,
                               float* output,
                               int num_matrices,
                               int order,
                               int iterations,
                               float epsilon);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
