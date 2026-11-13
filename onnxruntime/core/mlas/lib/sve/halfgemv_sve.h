/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemv_sve.h

Abstract:

    Declaration of the SVE FP16 matrix-vector kernel used by hgemm.cpp for N == 1.

--*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef MLAS_SVE_FP16_T_DEFINED
#define MLAS_SVE_FP16_T_DEFINED
typedef uint16_t mlas_sve_fp16_t;
#endif

extern "C" {

//
// C[m] = alpha * dot(A[m, 0..CountK), B[0..CountK)). B must be contiguous.
//
void
MlasHgemvNKernel_sve(
    const mlas_sve_fp16_t* A,
    const mlas_sve_fp16_t* B,
    mlas_sve_fp16_t* C,
    size_t CountM,
    size_t CountK,
    size_t lda,
    size_t ldc_elem,
    float alpha,
    bool ZeroMode
    );

}  // extern "C"
