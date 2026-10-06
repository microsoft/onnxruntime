// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
Status LaunchHyperConnectionPostMix(
    cudaStream_t stream, const T* streams, const T* branch_output,
    const void* post_mix, const void* stream_mix, int mix_type, T* y,
    int64_t count, int branches, int hidden, int gate_layout);

}  // namespace onnxruntime::contrib::cuda
