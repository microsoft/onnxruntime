// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

Status BuildTableForSoftmaxPowerOf(
    cudaStream_t stream, const double base, float* table);

// mask_index is (batch, sequence_len)
Status QOrderMaskedSoftmax(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, const float* lookup_table,
    const int32_t* mask_index,
    int8_t* dst, const float scale_dst,
    const unsigned batch, const unsigned num_heads, const unsigned sequence_len);

Status QOrderBatchTransposeInt8Matrix(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int batch_size, const int rows, const int cols,
    const int8_t* input, int8_t* output);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
