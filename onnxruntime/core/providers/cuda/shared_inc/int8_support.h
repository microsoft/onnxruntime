//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Use cublasLtMatmul to perform the tensor op Igemm with the memory
// order transforms on all buffers.
//
// For better performance the data order transforms should be offline
// as much as possible.
//
// Transa, transb assumed N; alpha, beta are host pointers; Tensor ops
// allowed. Alpha assumed 1, beta assumed 0, and stream assumed 0.

void LtIgemmTensor(cublasLtHandle_t ltHandle,
                          int m,
                          int n,
                          int k,
                          int32_t alpha,
                          int32_t beta,
                          const int8_t* A,
                          int lda,
                          const int8_t* B,
                          int ldb,
                          int32_t* C,
                          int ldc,
                          const CudaKernel* cuda_kernel);
}
}