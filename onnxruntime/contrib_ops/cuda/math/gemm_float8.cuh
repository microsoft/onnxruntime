// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace cuda {

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
struct GemmFloat8_Impl {
  void CudaCompute() const;
};
}  // namespace cuda
}  // namespace onnxruntime
