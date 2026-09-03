// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchEngramGateKernel(
    cudaStream_t stream,
    const T* key,
    const T* query,
    const T* value,
    const T* key_norm_scale,
    const T* query_norm_scale,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    float epsilon);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
