// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

#define BINARY_ELEMENTWISE_IMPL_DECLARATION() \
  template <typename T>                           \
  void Impl_CosGrad(cudaStream_t stream,           \
                   const T* lhs_data,             \
                   const T* rhs_data,             \
                   T* output_data,                \
                   const CosGrad<T>* func_ctx,     \
                   size_t count)

BINARY_ELEMENTWISE_IMPL_DECLARATION();

// template <typename T>
// Status CosGradImpl(cudaStream_t stream, const T* dy, const T* Y, T* output, size_t N);

}
}  // namespace onnxruntime
