/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemm_sve.h

Abstract:

    Declarations of the SVE FP16 GEMM kernels used by hgemm.cpp, implemented
    in sve/halfgemm_kernel_sve.cpp and aarch64/halfgemm_sve_asm.S.

--*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef PACKED_B_BLOCK_WIDTH_FP16
#define PACKED_B_BLOCK_WIDTH_FP16 32
#endif

typedef uint16_t mlas_sve_fp16_t;

extern "C" {

//
// C = alpha * A * PackedB. Returns the number of rows handled.
//
size_t
MlasHgemmKernelZero_sve(
    const mlas_sve_fp16_t* A,
    const mlas_sve_fp16_t* B,
    mlas_sve_fp16_t* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    );

//
// C += alpha * A * PackedB. Returns the number of rows handled.
//
size_t
MlasHgemmKernelAdd_sve(
    const mlas_sve_fp16_t* A,
    const mlas_sve_fp16_t* B,
    mlas_sve_fp16_t* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    );

void
MlasHgemmCopyPackB_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    );

void
MlasHgemmTransposePackB_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    );

//
// D[y, x] = A[x * lda + y]
//
void
MlasHgemmTransposeA_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* A,
    size_t lda,
    size_t CountY,
    size_t CountX
    );

void
MlasHgemvFloat16Kernel_sve(
    const mlas_sve_fp16_t* A,
    const mlas_sve_fp16_t* B,
    mlas_sve_fp16_t* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    );

void
MlasHgemv2Float16Kernel_sve(
    const mlas_sve_fp16_t* A,
    size_t lda,
    const mlas_sve_fp16_t* B,
    mlas_sve_fp16_t* C,
    size_t ldc,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    );

}  // extern "C"
