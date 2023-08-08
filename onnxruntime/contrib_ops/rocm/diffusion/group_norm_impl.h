// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

using onnxruntime::rocm::tunable::RocmTuningContext;

namespace onnxruntime {
namespace contrib {
namespace rocm {

constexpr size_t kMaxGroupNormBatchSize = 32;
constexpr size_t kGroupNormNumberOfGroups = 32;

constexpr size_t GetGroupNormWorkspaceSizeInBytes() {
  // Two buffers for sum and squared sum
  return (sizeof(float) * 2) * kMaxGroupNormBatchSize * kGroupNormNumberOfGroups;
}

template <typename T>
Status LaunchGroupNormKernel(
    RocmTuningContext* tuning_ctx,
    hipStream_t stream,
    T* output,                 // normalized output tensor
    const T* input,            // input tensor
    const float* gamma,        // gamma (also known as weight or scale)
    const float* beta,         // beta (also known as bias)
    void* workspace,           // Work space
    float epsilon,             // epsilon used normalization
    int batch_size,            // N
    int num_channels,          // C
    int height,                // H
    int width,                 // W
    int num_groups,            // number of groups
    bool use_swish_activation  // Whether there is Swish activation after group normalization
);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
