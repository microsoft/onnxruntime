// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

Status MLFloat16ToFloat8E4M3FN(cudaStream_t stream, const Tensor* src, Tensor* dest);

Status TransposeMatrix(cudaStream_t stream, const Tensor* src, Tensor* dest, int src_rows, int src_cols);

Status ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, Float8E4M3FN* h_scale_coef);

// Debugging utility that prints on device tensor data for all indices < last_index (or all data if last_index is -1).
template<typename T>
void PrintTensorData(cudaStream_t stream,  const Tensor* tensor, int last_index = -1);

}  // namespace cuda
}  // namespace onnxruntime
