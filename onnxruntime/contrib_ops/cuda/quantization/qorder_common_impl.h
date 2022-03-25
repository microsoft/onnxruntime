
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void QOrderQuantizeHalfRow_S8Col32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const half* src, int8_t* dst, float scale, int batch, size_t rows, size_t cols);

void QOrderDequantizeS8Col32_HalfRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, half* dst, float scale, int batch, size_t rows, size_t cols);

void QOrderLayerNorm(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
    const __half* gamma, const __half* beta, const float epsilon,
    const unsigned batch, const unsigned rows, const unsigned cols);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
