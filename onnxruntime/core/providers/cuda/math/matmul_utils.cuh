// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

Status MLFloat16ToFloat8E4M3FN(cudaStream_t stream, const Tensor* src, void* dest);

Status ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, MLFloat16* h_scale_coef);

// Debugging utility that prints on device tensor data for all indices < last_index (or all data if last_index is -1).
template <typename T>
void PrintTensorData(cudaStream_t stream,  const void* data, int num_elems, int last_index = -1);

}  // namespace cuda
}  // namespace onnxruntime
