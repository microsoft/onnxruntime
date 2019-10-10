// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool launchFullyConnect(
    const void* input,
    const void* weights,
    const void* bias,
    void* output,
    const int batch_size,      // B
    const int sequence_length, // S, and m=B*S
    const int hidden_size,     // k
    const int ld,              // n
    cublasHandle_t& cublas,
    const size_t element_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
