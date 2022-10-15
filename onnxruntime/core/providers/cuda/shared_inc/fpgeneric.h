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
#include <chrono>

using namespace std::chrono;
using namespace onnxruntime;
using namespace onnxruntime::cuda;

// Generalize library calls to be use in template functions
inline cublasStatus_t
cublasGemmHelper(cublasHandle_t handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m, int n, int k,
                 const float* alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 const float* beta,
                 float* C, int ldc,
                 const cudaDeviceProp& prop) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // TF32 uses 10 bit mantissa which has sufficient margin of precision for most use cases. It gets 8x throughput than FP32 in A100.
  // It can be overrided by setting environment variable NVIDIA_TF32_OVERRIDE = 0 to disable TF32
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  return cublasSgemm(handle,
                     transa,
                     transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const double* alpha,
                                       const double* A, int lda,
                                       const double* B, int ldb,
                                       const double* beta,
                                       double* C, int ldc,
                                       const cudaDeviceProp& /*prop*/) {
  return cublasDgemm(handle,
                     transa,
                     transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const half* alpha,
                                       const half* A, int lda,
                                       const half* B, int ldb,
                                       const half* beta,
                                       half* C, int ldc,
                                       const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        alpha,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        beta,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        &h_a,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        &h_b,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const float* alpha,
                                       const half* A, int lda,
                                       const half* B, int ldb,
                                       const float* beta,
                                       half* C, int ldc,
                                       const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = onnxruntime::math::floatToHalf(*alpha);
    uint16_t h_b = onnxruntime::math::floatToHalf(*beta);
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        &h_a,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        &h_b,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        alpha,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        beta,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                       int n, int k, const BFloat16* alpha, const BFloat16* A, int lda,
                                       const BFloat16* B, int ldb, const BFloat16* beta, BFloat16* C, int ldc,
                                       const cudaDeviceProp& /*prop*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &h_b, C,
                      CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                       const BFloat16*, const BFloat16*, int, const BFloat16*, int, const BFloat16*,
                                       BFloat16*, int, const cudaDeviceProp&) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// CublasLtMatmul
static Status InitializeCublasLtMatmulDescAndOperationHelper(cublasLtMatrixLayout_t& A_desc, int lda,
                                                             cublasOperation_t transa,
                                                             cublasLtMatrixLayout_t& B_desc, int ldb,
                                                             cublasOperation_t transb,
                                                             cublasLtMatrixLayout_t& C_desc, int ldc,
                                                             cudaDataType_t data_type,
                                                             int m, int n, int k,
                                                             cublasLtMatmulDesc_t& operation_desc,
                                                             cublasComputeType_t compute_type,
                                                             cudaDataType_t scale_type) {
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&A_desc, data_type,
                                                    (transa == CUBLAS_OP_N) ? m : k,
                                                    (transa == CUBLAS_OP_N) ? k : m, lda));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&B_desc, data_type,
                                                    (transb == CUBLAS_OP_N) ? k : n,
                                                    (transb == CUBLAS_OP_N) ? n : k, ldb));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&C_desc, data_type, m, n, ldc));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operation_desc, compute_type, scale_type));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operation_desc,
                                                        CUBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa, sizeof(cublasOperation_t)));

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(operation_desc,
                                                        CUBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb, sizeof(cublasOperation_t)));

  return Status::OK();
}

inline cublasStatus_t cublasLtMatmulHelper(cublasLtHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const half* alpha,
                                           const half* A, int lda,
                                           const half* B, int ldb,
                                           const half* beta,
                                           half* C, int ldc,
                                           const half* bias,
                                           bool gelu_activation,
                                           void* workspace_memory,
                                           size_t workspace_size,
                                           cudaStream_t stream) {
  std::cout << "Reached";

  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();

  cudaDataType_t data_type = CUDA_R_16F;
  cudaDataType_t scale_type = CUDA_R_16F;
  cublasComputeType_t compute_type = half_options->GetComputeType();

  float f_alpha, f_beta;  // use (if needed) to convert the half 'alpha' and 'beta' into float

  bool is_compute_16f = half_options->IsCompute16F();
  if (!is_compute_16f) {
    scale_type = CUDA_R_32F;

    // alpha and beta need to be the same type as the scale type
    f_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    f_beta = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  }

  cublasLtMatrixLayout_t A_desc = NULL, B_desc = NULL, C_desc = NULL;
  cublasLtMatmulDesc_t operation_desc = NULL;
    /*
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
                            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspace_size, sizeof(workspace_size));
    */
  auto clean_desc_A = gsl::finally([&A_desc]() {
    if (A_desc) {
      cublasLtMatrixLayoutDestroy(A_desc);
    }
  });

  auto clean_desc_B = gsl::finally([&B_desc]() {
    if (B_desc) {
      cublasLtMatrixLayoutDestroy(B_desc);
    }
  });

  auto clean_desc_C = gsl::finally([&C_desc]() {
    if (C_desc) {
      cublasLtMatrixLayoutDestroy(C_desc);
    }
  });

  auto clean_matmul_desc = gsl::finally([&operation_desc]() {
    if (operation_desc) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  });

  if (Status::OK() != InitializeCublasLtMatmulDescAndOperationHelper(A_desc, lda,
                                                                     transa,
                                                                     B_desc, ldb,
                                                                     transb,
                                                                     C_desc, ldc,
                                                                     data_type,
                                                                     m, n, k,
                                                                     operation_desc,
                                                                     compute_type,
                                                                     scale_type)) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }

  if (gelu_activation && bias != nullptr) {
    cublasLtEpilogue_t epilogue_gelu_bias = CUBLASLT_EPILOGUE_GELU_BIAS;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_gelu_bias, sizeof(epilogue_gelu_bias)));

  } else if (bias != nullptr) {
    cublasLtEpilogue_t epilogue_bias = CUBLASLT_EPILOGUE_BIAS;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_bias, sizeof(epilogue_bias)));
  } else if (gelu_activation) {
    cublasLtEpilogue_t epilogue_gelu = CUBLASLT_EPILOGUE_GELU;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_gelu, sizeof(epilogue_gelu)));
  }

  if (bias != nullptr) {
    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &bias, sizeof(bias)));
  }

    /*
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtMatmulAlgoGetHeuristic(handle, operation_desc, A_desc, B_desc, C_desc,
                                                     C_desc, preference, 1, &heuristicResult,
                                                     &returnedResults);

    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");
    */

      cudaStreamSynchronize(stream);
      auto start = high_resolution_clock::now();

  // TODO (hasesh): Allow CublasLtMatmul tuning for clients by allowing them to pass in the
  // workspace and algo of their choice.
  // According to the cublasLtMatmul documentation, passing in NULL for the algo means that
  // "an implicit heuristics query with
  // default search preferences will be performed to determine actual algorithm to use".
  // Source: cublasLtMatmul documentation.
  auto status =  cublasLtMatmul(
      handle, operation_desc,
      is_compute_16f ? reinterpret_cast<const void*>(alpha) : reinterpret_cast<const void*>(&f_alpha),
      A, A_desc, B, B_desc,
      is_compute_16f ? reinterpret_cast<const void*>(beta) : reinterpret_cast<const void*>(&f_beta),
      C, C_desc,
      C, C_desc,
      NULL,
      workspace_memory, workspace_size,
      stream);

      cudaStreamSynchronize(stream);
      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop - start);
    
      std::cout << "M" << duration.count() << std::endl;

      return status;
}

inline cublasStatus_t cublasLtMatmulHelper(cublasLtHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const float* alpha,
                                           const float* A, int lda,
                                           const float* B, int ldb,
                                           const float* beta,
                                           float* C, int ldc,
                                           const float* bias,
                                           bool gelu_activation,
                                           void* workspace_memory,
                                           size_t workspace_size,
                                           cudaStream_t stream) {
  cudaDataType_t data_type = CUDA_R_32F;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

  cublasLtMatrixLayout_t A_desc = NULL, B_desc = NULL, C_desc = NULL;
  cublasLtMatmulDesc_t operation_desc = NULL;

  auto clean_desc_A = gsl::finally([&A_desc]() {
    if (A_desc) {
      cublasLtMatrixLayoutDestroy(A_desc);
    }
  });

  auto clean_desc_B = gsl::finally([&B_desc]() {
    if (B_desc) {
      cublasLtMatrixLayoutDestroy(B_desc);
    }
  });

  auto clean_desc_C = gsl::finally([&C_desc]() {
    if (C_desc) {
      cublasLtMatrixLayoutDestroy(C_desc);
    }
  });

  auto clean_matmul_desc = gsl::finally([&operation_desc]() {
    if (operation_desc) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  });

  if (Status::OK() != InitializeCublasLtMatmulDescAndOperationHelper(A_desc, lda,
                                                                     transa,
                                                                     B_desc, ldb,
                                                                     transb,
                                                                     C_desc, ldc,
                                                                     data_type,
                                                                     m, n, k,
                                                                     operation_desc,
                                                                     compute_type,
                                                                     scale_type)) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }

  if (gelu_activation && bias != nullptr) {
    cublasLtEpilogue_t epilogue_gelu_bias = CUBLASLT_EPILOGUE_GELU_BIAS;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_gelu_bias, sizeof(epilogue_gelu_bias)));

  } else if (bias != nullptr) {
    cublasLtEpilogue_t epilogue_bias = CUBLASLT_EPILOGUE_BIAS;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_bias, sizeof(epilogue_bias)));
  } else if (gelu_activation) {
    cublasLtEpilogue_t epilogue_gelu = CUBLASLT_EPILOGUE_GELU;

    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                     &epilogue_gelu, sizeof(epilogue_gelu)));
  }

  if (bias != nullptr) {
    CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(operation_desc,
                                                     CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &bias, sizeof(bias)));
  }

  // TODO (hasesh): Allow CublasLtMatmul tuning for clients by allowing them to pass in the
  // workspace and algo of their choice.
  // According to the cublasLtMatmul documentation, passing in NULL for the algo means that
  // "an implicit heuristics query with
  // default search preferences will be performed to determine actual algorithm to use".
  // Source: cublasLtMatmul documentation.
  return cublasLtMatmul(
      handle, operation_desc,
      reinterpret_cast<const void*>(alpha),
      A, A_desc, B, B_desc,
      reinterpret_cast<const void*>(beta),
      C, C_desc,
      C, C_desc,
      /*algo*/ NULL,
      workspace_memory, workspace_size,
      stream);
}

inline cublasStatus_t cublasLtMatmulHelper(cublasLtHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const double* alpha,
                                           const double* A, int lda,
                                           const double* B, int ldb,
                                           const double* beta,
                                           double* C, int ldc,
                                           const double* bias,
                                           bool gelu_activation,
                                           void* workspace_memory,
                                           size_t workspace_size,
                                           cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(handle);
  ORT_UNUSED_PARAMETER(transa);
  ORT_UNUSED_PARAMETER(transb);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(A);
  ORT_UNUSED_PARAMETER(lda);
  ORT_UNUSED_PARAMETER(B);
  ORT_UNUSED_PARAMETER(ldb);
  ORT_UNUSED_PARAMETER(beta);
  ORT_UNUSED_PARAMETER(C);
  ORT_UNUSED_PARAMETER(ldc);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(gelu_activation);
  ORT_UNUSED_PARAMETER(workspace_memory);
  ORT_UNUSED_PARAMETER(workspace_size);
  ORT_UNUSED_PARAMETER(stream);

  return CUBLAS_STATUS_NOT_SUPPORTED;
}

inline cublasStatus_t cublasLtMatmulHelper(cublasLtHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const BFloat16* alpha,
                                           const BFloat16* A, int lda,
                                           const BFloat16* B, int ldb,
                                           const BFloat16* beta,
                                           BFloat16* C, int ldc,
                                           const BFloat16* bias,
                                           bool gelu_activation,
                                           void* workspace_memory,
                                           size_t workspace_size,
                                           cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(handle);
  ORT_UNUSED_PARAMETER(transa);
  ORT_UNUSED_PARAMETER(transb);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(A);
  ORT_UNUSED_PARAMETER(lda);
  ORT_UNUSED_PARAMETER(B);
  ORT_UNUSED_PARAMETER(ldb);
  ORT_UNUSED_PARAMETER(beta);
  ORT_UNUSED_PARAMETER(C);
  ORT_UNUSED_PARAMETER(ldc);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(gelu_activation);
  ORT_UNUSED_PARAMETER(workspace_memory);
  ORT_UNUSED_PARAMETER(workspace_size);
  ORT_UNUSED_PARAMETER(stream);

  return CUBLAS_STATUS_NOT_SUPPORTED;
}

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const float* alpha,
                                              const float* Aarray[], int lda,
                                              const float* Barray[], int ldb,
                                              const float* beta,
                                              float* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  return cublasSgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const double* alpha,
                                              const double* Aarray[], int lda,
                                              const double* Barray[], int ldb,
                                              const double* beta,
                                              double* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& /*prop*/) {
  return cublasDgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const half* alpha,
                                              const half* Aarray[], int lda,
                                              const half* Barray[], int ldb,
                                              const half* beta,
                                              half* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m, n, k,
                               alpha,
                               (const void**)Aarray, CUDA_R_16F, lda,
                               (const void**)Barray, CUDA_R_16F, ldb,
                               beta,
                               (void**)Carray, CUDA_R_16F, ldc,
                               batch_count,
                               half_options->GetComputeType(),
                               CUBLAS_GEMM_DEFAULT);
  } else {
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m, n, k,
                               &h_a,
                               (const void**)Aarray, CUDA_R_16F, lda,
                               (const void**)Barray, CUDA_R_16F, ldb,
                               &h_b,
                               (void**)Carray, CUDA_R_16F, ldc,
                               batch_count,
                               half_options->GetComputeType(),
                               CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                              int m, int n, int k, const BFloat16* alpha, const BFloat16* Aarray[],
                                              int lda, const BFloat16* Barray[], int ldb, const BFloat16* beta,
                                              BFloat16* Carray[], int ldc, int batch_count,
                                              const cudaDeviceProp& /*prop*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return cublasGemmBatchedEx(handle, transa, transb, m, n, k, &h_a, (const void**)Aarray, CUDA_R_16BF, lda,
                             (const void**)Barray, CUDA_R_16BF, ldb, &h_b, (void**)Carray, CUDA_R_16BF, ldc,
                             batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                              const BFloat16*, const BFloat16*[], int, const BFloat16*[], int,
                                              const BFloat16*, BFloat16*[], int, int, const cudaDeviceProp&) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// strided batched gemm
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const float* A, int lda,
                                                     long long int strideA,
                                                     const float* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     float* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
#ifdef ENABLE_TRAINING
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  return cublasSgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, lda, strideA,
                                   B, ldb, strideB,
                                   beta,
                                   C, ldc, strideC,
                                   batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const double* alpha,
                                                     const double* A, int lda,
                                                     long long int strideA,
                                                     const double* B, int ldb,
                                                     long long int strideB,
                                                     const double* beta,
                                                     double* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& /*prop*/) {
  return cublasDgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, lda, strideA,
                                   B, ldb, strideB,
                                   beta,
                                   C, ldc, strideC,
                                   batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const __half* alpha,
                                                     const __half* A, int lda,
                                                     long long int strideA,
                                                     const __half* B, int ldb,
                                                     long long int strideB,
                                                     const __half* beta,
                                                     __half* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      alpha,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      beta,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      &h_a,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      &h_b,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const __half* A, int lda,
                                                     long long int strideA,
                                                     const __half* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     __half* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = onnxruntime::math::floatToHalf(*alpha);
    uint16_t h_b = onnxruntime::math::floatToHalf(*beta);
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      &h_a,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      &h_b,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      alpha,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      beta,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle, cublasOperation_t transa,
                                                     cublasOperation_t transb, int m, int n, int k,
                                                     const BFloat16* alpha, const BFloat16* A, int lda,
                                                     long long int strideA, const BFloat16* B, int ldb,
                                                     long long int strideB, const BFloat16* beta, BFloat16* C, int ldc,
                                                     long long int strideC, int batch_count,
                                                     const cudaDeviceProp& /*prop*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();
  // accumulating in FP32
  return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16BF, lda, strideA, B, CUDA_R_16BF,
                                    ldb, strideB, &h_b, C, CUDA_R_16BF, ldc, strideC, batch_count, CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
                                                     int, const BFloat16*, const BFloat16*, int, long long int,
                                                     const BFloat16*, int, long long int, const BFloat16*, BFloat16*,
                                                     int, long long int, int, const cudaDeviceProp&) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// copy
inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t handle, int n, const half* x, int incx, half* y, int incy);
cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t handle, int n, const BFloat16* x, int incx, BFloat16* y, int incy);
