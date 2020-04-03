// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "igemm.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {

void LtIgemmTensor( int m,
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
                          cublasLtHandle_t lt_handle) {
  // Create descriptors for the original matrices
  cublasLtMatrixLayout_t a_desc = NULL;
  cublasLtMatrixLayout_t b_desc = NULL;
  cublasLtMatrixLayout_t c_desc = NULL;
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8I, m, k, lda));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, n, k, ldb));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32I, m, n, ldc));

  // Set A and C row major order.
  // No need for B because B need to be transposed
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute( a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute( a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  int lda_transform = 32 * m;
  int ldb_transform = 32 * roundoff(n, 8);
  int ldc_transform = 32 * m;

  // Allocate memory for transform
  IAllocatorUniquePtr<int8_t> a_transform = cuda_kernel->GetScratchBuffer<int8_t>(roundoff(k, 32) / 32 * lda_transform);
  IAllocatorUniquePtr<int8_t> b_transform = cuda_kernel->GetScratchBuffer<int8_t>(roundoff(k, 32) / 32 * ldb_transform);
  IAllocatorUniquePtr<int32_t> c_transform = cuda_kernel->GetScratchBuffer<int32_t>(roundoff(k, 32) / 32 * ldc_transform);

  // Create descriptors for the transformed matrices
  cublasLtMatrixLayout_t a_transform_desc = NULL;
  cublasLtMatrixLayout_t b_transform_desc = NULL;
  cublasLtMatrixLayout_t c_transform_desc = NULL;
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&a_transform_desc, CUDA_R_8I, m, k, lda_transform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&b_transform_desc, CUDA_R_8I, n, k, ldb_transform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&c_transform_desc, CUDA_R_32I, m, n, ldc_transform));

  // The tensor operations IGEMM kernels require specialized memory order of data.
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(a_transform_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(b_transform_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(c_transform_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  CUBLAS_CALL_THROW(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));

  float alpha_transform = 1.0f;
  float beta_transform = 0.0f;
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(lt_handle,
                                            transform_desc,
                                            &alpha_transform,
                                            a,
                                            a_desc,
                                            &beta_transform,
                                            NULL,
                                            NULL,
                                            a_transform.get(),
                                            a_transform_desc,
                                            0));

  CUBLAS_CALL_THROW(cublasLtMatrixTransform(lt_handle,
                                            transform_desc,
                                            &alpha_transform,
                                            b,
                                            b_desc,
                                            &beta_transform,
                                            NULL,
                                            NULL,
                                            b_transform.get(),
                                            b_transform_desc,
                                            0));

    if(beta_matmul == 1){
      CUBLAS_CALL_THROW(cublasLtMatrixTransform(lt_handle,
                                          transform_desc,
                                          &alpha_transform,
                                          c,
                                          c_desc,
                                          &beta_transform,
                                          NULL,
                                          NULL,
                                          c_transform.get(),
                                          c_transform_desc,
                                          0));  
    }

  // Tensor op igemm kernels only support NT gemm
  cublasLtMatmulDesc_t matmul_desc = NULL;
  cublasOperation_t op_trans = CUBLAS_OP_T;
  CUBLAS_CALL_THROW(cublasLtMatmulDescCreate(&matmul_desc, CUDA_R_32I));
  CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_trans, sizeof(op_trans)));

  CUBLAS_CALL_THROW(cublasLtMatmul(lt_handle,
                                   matmul_desc,
                                   &alpha_matmul,
                                   a_transform.get(),
                                   a_transform_desc,
                                   b_transform.get(),
                                   b_transform_desc,
                                   &beta_matmul,
                                   c_transform.get(),
                                   c_transform_desc,
                                   c_transform.get(),
                                   c_transform_desc,
                                   NULL,
                                   NULL,
                                   0,
                                   0));

      CUBLAS_CALL_THROW(cublasLtMatrixTransform(lt_handle,
                                          transform_desc,
                                          &alpha_transform,
                                          c_transform.get(),
                                          c_transform_desc,
                                          &beta_transform,
                                          NULL,
                                          NULL,
                                          c,
                                          c_desc,
                                          0));

  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(c_transform_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(b_transform_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(a_transform_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CALL_THROW(cublasLtMatmulDescDestroy(matmul_desc));
  CUBLAS_CALL_THROW(cublasLtMatrixTransformDescDestroy(transform_desc));
}

}
}