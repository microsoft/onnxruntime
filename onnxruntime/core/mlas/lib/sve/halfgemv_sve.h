/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemv_sve.h

Abstract:

    Interface to the SVE FP16 matrix-vector kernel, for the N == 1 case that
    the general HGEMM path handles badly.

    The general path packs B into PACKED_B_BLOCK_WIDTH_FP16 (32) wide blocks,
    so a K x 1 operand pays for a full block: measured on Graviton3, N = 1
    through N = 16 all cost the same (~404 us at M = K = 1023), i.e. 21x the
    per-column rate reached at N >= 32. This kernel skips the pack entirely
    and streams B once per row group.

    As with the HGEMM kernels there are two interchangeable implementations --
    the SVE intrinsics reference in sve/halfgemv_kernel_sve.cpp and the frozen
    machine code in aarch64/halfgemv_sve_asm.S -- so the symbol is extern "C".

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
// C[m] = alpha * dot(A[m, 0..CountK), B[0..CountK)) for m in [0, CountM).
//
//   A       CountM x CountK, row-major, row stride lda
//   B       length-CountK vector, contiguous. SVE has no 16-bit gather, so a
//           strided B is left to the general path (the driver checks).
//   C       length-CountM output, element stride ldc_elem
//   ZeroMode  true  => C is overwritten
//             false => the product is accumulated into C
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
