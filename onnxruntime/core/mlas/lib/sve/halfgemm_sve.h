/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemm_sve.h

Abstract:

    Interface between the HGEMM driver (hgemm.cpp) and the SVE FP16 compute
    kernels. The kernels have two interchangeable implementations -- the SVE
    intrinsics reference in sve/halfgemm_kernel_sve.cpp and the frozen
    machine code in aarch64/halfgemm_sve_asm.S -- so the symbols are
    extern "C" and this header is the single declaration both the driver and
    the reference translation unit agree on.

--*/

#pragma once

#include <stddef.h>
#include <stdint.h>

//
// Physical width, in FP16 elements, of one packed B block. The driver sizes
// and strides the packed-B panel with this and the kernels consume it, and
// the two are compiled separately (and one of them may not be compiled at
// all), so the value lives in the one header both include rather than in
// per-target compile flags where they could drift apart.
//
// Must equal MLAS_HGEMM_STRIDEN_THREAD_ALIGN; hgemm.cpp static_asserts this.
//
#ifndef PACKED_B_BLOCK_WIDTH_FP16
#define PACKED_B_BLOCK_WIDTH_FP16 32
#endif

//
// The raw FP16 storage word. Spelled locally so the reference translation
// unit stays self-sufficient (see the note in halfgemm_kernel_sve.cpp).
//
typedef uint16_t mlas_sve_fp16_t;

extern "C" {

//
// C = alpha * A * PackedB. Returns the number of M rows handled.
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
// C += alpha * A * PackedB. Returns the number of M rows handled.
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

//
// Pack a CountX x CountY block of row-major B into PACKED_B_BLOCK_WIDTH_FP16
// wide blocks, zero padding the tail of each block.
//
void
MlasHgemmCopyPackB_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    );

//
// As MlasHgemmCopyPackB_sve, transposing B on the way in.
//
void
MlasHgemmTransposePackB_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    );

//
// Transpose a CountX x CountY block of A into the row-major CountY x CountX
// panel the NoTrans kernels consume: D[y, x] = A[x * lda + y].
//
void
MlasHgemmTransposeA_sve(
    mlas_sve_fp16_t* D,
    const mlas_sve_fp16_t* A,
    size_t lda,
    size_t CountY,
    size_t CountX
    );

//
// C = A * B for a single row of A (M == 1). ZeroMode overwrites C, otherwise
// C is accumulated into.
//
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

}  // extern "C"
