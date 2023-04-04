// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

#include "core/common/common.h"
#include "core/providers/rocm/tunable/gemm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {
namespace internal {

#ifdef USE_HIPBLASLT

template <typename T>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const T*) {
  static_assert(sizeof(T) == 0, "Unsupported data type for hipBLASLt operation.");
  // Compiler will complain if we don't return something.
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const float*) {
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const half*) {
  return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const BFloat16*) {
  return HIPBLAS_R_16B;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor(const double*) {
  return HIPBLAS_R_64F;
}

template <typename T, typename ParamsT>
class HipBlasLtGemmOp {
 public:
  Status operator()(const ParamsT* params) {

    hipblasLtHandle_t handle;
    HIPBLASLT_CALL_THROW(hipblasLtCreate(&handle));

    // Note: properties of original matrices A and B are swapped.
    int64_t lda = (params->opb == BlasOp::N) ? params->n : params->k;
    int64_t ldb = (params->opa == BlasOp::N) ? params->k : params->m;
    int64_t ldc = params->n;
    int64_t stride_a = params->n * params->k;
    int64_t stride_b = params->k * params->m;
    int64_t stride_c = params->n * params->m;
    float alpha = static_cast<float>(params->alpha);
    float beta = static_cast<float>(params->beta);
    int row_a, col_a, row_b, col_b, row_c, col_c;
    row_a = lda;
    col_a = (params->opb == BlasOp::N) ? params->k : params->n;
    row_b = ldb;
    col_b = (params->opa == BlasOp::N) ? params->m : params->k;
    row_c = ldc;
    col_c = params->m;

    hipblasDatatype_t in_out_datatype = HipBlasDataTypeFor(params->a);
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
    hipblasLtMatmulDesc_t matmul;
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutCreate(&mat_a, in_out_datatype, row_a, col_a, lda));
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutCreate(&mat_b, in_out_datatype, row_b, col_b, ldb));
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutCreate(&mat_c, in_out_datatype, row_c, col_c, ldc));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32, HIPBLAS_R_32F));

    if (params->batch > 1) {
      int batch_count = params->batch;
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutSetAttribute(
          mat_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
    }

    hipblasOperation_t trans_a = (params->opb == BlasOp::N) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t trans_b = (params->opa == BlasOp::N) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    HIPBLASLT_CALL_THROW(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue;
    HIPBLASLT_CALL_THROW(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    hipblasLtMatmulPreference_t pref;
    void* workspace;
    HIP_CALL_THROW(hipMalloc(&workspace, workspace_size_));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulPreferenceCreate(&pref));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    const int heuristic_result_count = 3;
    hipblasLtMatmulHeuristicResult_t heuristic_result[heuristic_result_count] = {0};
    int ret_algo_count = 0;
    HIPBLASLT_CALL_THROW(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                         matmul,
                                                         mat_a,
                                                         mat_b,
                                                         mat_c,
                                                         mat_c,
                                                         pref,
                                                         heuristic_result_count,
                                                         heuristic_result,
                                                         &ret_algo_count));

    HIPBLASLT_CALL_THROW(hipblasLtMatmul(handle,
                                         matmul,
                                         &alpha,
                                         params->b,
                                         mat_a,
                                         params->a,
                                         mat_b,
                                         &beta,
                                         params->c,
                                         mat_c,
                                         params->c,
                                         mat_c,
                                         &heuristic_result[0].algo,
                                         workspace,
                                         workspace_size_,
                                         params->stream));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulPreferenceDestroy(pref));
    HIPBLASLT_CALL_THROW(hipblasLtMatmulDescDestroy(matmul));
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CALL_THROW(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CALL_THROW(hipblasLtDestroy(handle));
    return Status::OK();
  }

  Status IsSupported(const GemmParams<T>* params) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      (std::is_same_v<T, double>), "hipBLASLt does not support double inputs");
    ORT_UNUSED_PARAMETER(params);
    return Status::OK();
  }

 private:
  uint64_t workspace_size_ = 1024 * 1024;
};

#endif  // USE_HIPBLASLT

}  // namespace internal
}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
