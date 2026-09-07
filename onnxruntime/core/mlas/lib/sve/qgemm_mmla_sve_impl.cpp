/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_mmla_sve_impl.cpp

Abstract:

    SVE intrinsics reference implementation of the int8 svmmla GEMM compute
    kernels, and the regeneration source for the portable machine-code
    variant (aarch64/qgemm_mmla_sve_asm.S, script: sve/gen_sve_asm.py).

    This is the only QGEMM translation unit that requires an SVE-capable
    compiler (-march=...+sve+i8mm); the driver/pack/dispatch code is plain
    C++. The exported extern "C" symbols match the generated assembly
    exactly, so the build can link either implementation.

    Self-containment contract (verified by the generator): the kernels make
    no calls, reference no global data and use no literal pools — every
    input arrives through the argument registers/stack, so the frozen
    machine code is fully position-independent.

--*/

#include "qgemm_mmla_sve.h"

#include <arm_sve.h>

// Deliberately self-sufficient (no mlasi.h): the smaller the translation
// unit, the simpler the frozen object is to verify. This TU is only ever
// compiled by an SVE-capable gcc/clang.
#if !defined(MLAS_FORCEINLINE)
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif

#include "qgemm_mmla_kernel_sve.h"

extern "C" {

size_t
MlasGemmS8S8KernelSmmlaSveImpl(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    // S8S8: signed A x signed B -> svmmla_s32.
    return MlasQGemmMmlaKernelSve</*AUnsigned=*/false>(
        A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

size_t
MlasGemmU8X8KernelUmmlaSveImpl(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    // U8S8: unsigned A x signed B -> svmmla_u32 on offset-adjusted operands.
    return MlasQGemmMmlaKernelSve</*AUnsigned=*/true>(
        A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

}
