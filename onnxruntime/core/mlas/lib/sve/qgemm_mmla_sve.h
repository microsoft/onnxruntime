/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_mmla_sve.h

Abstract:

    Prototypes for the SVE i8mm (svmmla) int8 GEMM compute kernels.

    The symbols are extern "C" so the two interchangeable implementations
    link identically: the SVE intrinsics reference translation unit
    (qgemm_mmla_sve_impl.cpp) and the generated KleidiAI-style machine-code
    variant (aarch64/qgemm_mmla_sve_asm.S). The build links exactly one of
    the two. This header is includable from translation units compiled
    WITHOUT SVE support (the QGEMM driver code), on any platform.

--*/

#pragma once

#include <cstddef>
#include <cstdint>

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
    );

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
    );

}
