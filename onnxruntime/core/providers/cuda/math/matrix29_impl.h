// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status Gemm29AddBias(cudaStream_t stream, T* output, const T* bias, float beta,
                     int64_t count, int rank, const int64_t* dimensions, const int64_t* bias_strides);

template <typename T>
Status GroupedMatMulBias(cudaStream_t stream, T* output, const T* const* bias, int64_t count, int64_t n);

template <typename T>
Status GroupedMatMulGatherRows(cudaStream_t stream, const T* input, T* packed, const int64_t* order,
                               int64_t count, int64_t k, int64_t selections);

template <typename T>
Status GroupedMatMulScatterRows(cudaStream_t stream, const T* packed, T* output, const int64_t* order,
                                const T* const* bias, int64_t count, int64_t n);

}  // namespace cuda
}  // namespace onnxruntime
