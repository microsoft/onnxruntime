// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "igemv_mkl.h"

namespace onnxruntime {
#ifdef NUPHAR_USE_MKL
void MKLIntGemvS16S16S32R(
    int16_t* matrixA,
    int16_t* matrixB,
    int M,
    int N,
    int K,
    int32_t* output) {
  MKL_INT32 co = 0;
  cblas_gemm_s16s16s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset,
                       M, N, K,
                       1, matrixA, K,
                       0, matrixB, K, 0, 0, output, M, &co);
}
void MKLIntGemvS8U8S32R(
    int8_t* matrixA,
    uint8_t* matrixB,
    int M,
    int N,
    int K,
    int32_t* output) {
  MKL_INT32 co = 0;
  cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset,
                     M, N, K,
                     1, matrixA, K,
                     0, matrixB, K, 0, 0, output, M, &co);
}
#endif

}  // namespace onnxruntime
