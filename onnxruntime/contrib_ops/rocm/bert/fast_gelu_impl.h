// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove cudaDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchFastGeluKernel(RocmTuningContext* tuning_ctx, hipStream_t stream, int input_length, int bias_length,
                            const T* input, const T* bias, T* output);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
