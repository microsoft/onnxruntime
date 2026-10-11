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

//
// The 12-row tile. The driver translation units use this to pick Strides.M, and
// the compute core uses it to pick the Rows == 12 path; the two are compiled
// separately, so the default lives in the one header both include rather than
// in per-target compile flags, where they could drift apart. An explicit
// -DMLAS_SVE_QGEMM_TILE_12X8=0 still overrides.
//
// Turning it off is correct, only slower: the kernel returns the row count it
// handled and the driver advances packed A linearly by it, so a 12-row group
// packed as [8-group][4-group] is simply consumed as 8 then 4.
//
#ifndef MLAS_SVE_QGEMM_TILE_12X8
#define MLAS_SVE_QGEMM_TILE_12X8 1
#endif

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
