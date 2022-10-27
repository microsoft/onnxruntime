// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm_common.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

// RAII style guard to set stream and restore original stream for rocblas_handle
class RocblasHandleStreamGuard {
 public:
  RocblasHandleStreamGuard(rocblas_handle handle, hipStream_t stream) : handle_{handle} {
    ROCBLAS_CALL_THROW(rocblas_get_stream(handle_, &original_stream_));
    ROCBLAS_CALL_THROW(rocblas_set_stream(handle_, stream));
  }

  ~RocblasHandleStreamGuard() {
    ROCBLAS_CALL_THROW(rocblas_set_stream(handle_, original_stream_));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(RocblasHandleStreamGuard);

 private:
  rocblas_handle handle_;
  hipStream_t original_stream_;
};

template <typename T>
Status RocBlasGemmOp(const GemmParams<T>* params) {
  RocblasHandleStreamGuard guard(params->handle, params->stream);
  // NOTE: rocblas assumes the storage is column-majored, swapping A and B makes it have the same interface
  // as those with row-majored convention. That is, if you treat the storage as row-majored but view the matrices as
  // transposed, then by using the property Transpose(A*B) = Tranpose(B)*Transpose(A), the correctness is obvious.
  return ROCBLAS_CALL(rocblasGemmHelper(
      params->handle,
      params->opb == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->opa == BlasOp::N ? rocblas_operation_none : rocblas_operation_transpose,
      params->n, params->m, params->k,
      &(params->alpha),
      params->b, params->ldb,
      params->a, params->lda,
      &(params->beta),
      params->c, params->ldc));
}

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
