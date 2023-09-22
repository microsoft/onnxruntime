//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <stdint.h>
enum JBLAS_CODE {
  JblasSuccess = 0,
  JblasInvalidParam = -1,
  JblasInvalidISA = -2,
  JblasRuntimeError = -3,
  JblasNotSupport = -4,
};
enum JBLAS_ISA {
  JblasNoSIMD = 10,
  JblasAVX = 11,
  JblasAVX2 = 12,
  JblasAVX_VNNI = 13,
  JblasAVX512F = 14,
  JblasAVX512_VNNI = 15,
  JblasAMX_BF16 = 16,
  JblasAMX_INT8 = 17,
  JblasAVX512_FP16 = 18,
};
enum JBLAS_DTYPE {
  JblasF64 = 59,
  JblasF32 = 60,
  JblasBF16 = 61,
  JblasS8 = 63,
  JblasU8 = 64,
  JblasF32F8 = 65,
};
enum JBLAS_FP8_ENCODING {
  JblasFp8_e4m3 = 80,
  JblasFp8_e5m2 = 81,
  JblasFp8_e3m4 = 82,
};
enum JBLAS_LAYOUT { JblasRowMajor = 101, JblasColMajor = 102 };
enum JBLAS_TRANSPOSE {
  JblasNoTrans = 111,
  JblasTrans = 112,
  JblasConjTrans = 113,
};
enum JBLAS_ELTWISEOP {
  GELU,
  SWISH,
  TANH,
  EXP,
  LOW_PRECISION_EXP,
  RELU,
  LINEAR,
};
enum JBLAS_F4_TYPE {
  F4_UNDEF,
  FP4_BNB,
  FP4_E2M1,
  NF4,
};
enum JBLAS_SIGN_INT_TYPE {
  S8,
  S4_CLIP,
  S4_FULLRANGE,
  S4_UNDEF,
};
void jblas_sgemm(const JBLAS_LAYOUT Layout, const JBLAS_TRANSPOSE TransA, const JBLAS_TRANSPOSE TransB, const int M,
                 const int N, const int K, const float alpha, const float* A, const int lda, const float* B,
                 const int ldb, float* C, const int ldc, const float beta, const float* D, const int ldd);

void jblas_bf16gemm(const JBLAS_LAYOUT Layout, const JBLAS_TRANSPOSE TransA, const JBLAS_TRANSPOSE TransB, const int M,
                    const int N, const int K, const float alpha, const uint16_t* A, const int lda, const uint16_t* B,
                    const int ldb, uint16_t* C, const int ldc, const float beta, const uint16_t* D, const int ldd);

void jblas_u8s8gemm(const JBLAS_LAYOUT Layout, const JBLAS_TRANSPOSE TransA, const JBLAS_TRANSPOSE TransB, const int M,
                    const int N, const int K, const float alpha, const uint8_t* A, const int lda, const float scaleA,
                    const int zeropointA, const int8_t* B, const int ldb, const float scaleB, uint8_t* C, const int ldc,
                    const float scaleC, const int zeropointC, const float beta, uint8_t* D, const int ldd,
                    const float scaleD, const int zeropointD);

void jblas_set_num_threads(int num);
