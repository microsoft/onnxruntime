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
    const T* embeddings,
    const T* hidden_states,
    const T* key_weight,
    const T* key_bias,
    const T* value_weight,
    const T* value_bias,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const T* conv_norm_scale,
    T* output,
    T* output_normed,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
