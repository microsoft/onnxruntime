
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void QOrderQuantizeHalfRow_S8Col32(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                   const half* src, int8_t* dst, float scale, int batch, size_t rows, size_t cols);

void QOrderDequantizeS8Col32_HalfRow(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                     const int8_t* src, half* dst, float scale, int batch, size_t rows, size_t cols);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
