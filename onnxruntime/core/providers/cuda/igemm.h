// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

inline int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

#if CUDA_VERSION >= 10010
void LtIgemmTensor(int m,
                   int n,
                   int k,
                   int32_t alpha_matmul,
                   int32_t beta_matmul,
                   const int8_t* a,
                   int lda,
                   const int8_t* b,
                   int ldb,
                   int32_t* c,
                   int ldc,
                   const CudaKernel* cuda_kernel,
                   cublasLtHandle_t lt_handle);
#endif
}
}