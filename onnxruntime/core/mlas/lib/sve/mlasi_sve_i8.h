/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

    mlasi_sve_i8.h

Abstract:

    SVE intrinsic wrappers specific to int8 / QGEMM kernels (FEAT_I8MM).
    Contains only the instructions that operate on uint8 data or the
    svmmla_u32 matrix-multiply-accumulate (UMMLA) instruction.

    All general-purpose integer SVE helpers (predicates, reinterprets,
    arithmetic, permutes) live in mlasi_sve.h, which is included here.

--*/

#pragma once

#include "mlasi_sve.h"

#ifdef __clang__
#undef MLAS_SVE_TARGET
#define MLAS_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve+i8mm")))
#endif

#ifndef __clang__
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+i8mm")
#endif

// ---------------------------------------------------------------------------
// uint8 reinterpret
// ---------------------------------------------------------------------------

/// Reinterpret svuint8_t as svuint32_t.
MLAS_SVE_TARGET
MLAS_FORCEINLINE
svuint32_t
MlasSveReinterpretU32FromU8(svuint8_t Vector)
{
    return svreinterpret_u32_u8(Vector);
}

// ---------------------------------------------------------------------------
// uint8 loads
// ---------------------------------------------------------------------------

/// Load 16 bytes and replicate across the full SVE vector (svld1rq_u8).
/// Used to broadcast an A tile across the accumulator width.
MLAS_SVE_TARGET
MLAS_FORCEINLINE
svuint8_t
MlasSveLoadReplicateU8(MLAS_SVBOOL Pred, const uint8_t* Buffer)
{
    return svld1rq_u8(Pred, Buffer);
}

/// Load a full SVE vector of uint8 elements (svld1_u8).
/// Used to load B tiles.
MLAS_SVE_TARGET
MLAS_FORCEINLINE
svuint8_t
MlasSveLoadU8(MLAS_SVBOOL Pred, const uint8_t* Buffer)
{
    return svld1_u8(Pred, Buffer);
}

// ---------------------------------------------------------------------------
// Matrix-multiply-accumulate (FEAT_I8MM)
// ---------------------------------------------------------------------------

/// Unsigned 8-bit matrix multiply-accumulate into uint32 (svmmla_u32 / UMMLA).
/// Computes: Acc += A_tile * B_tile
MLAS_SVE_TARGET
MLAS_FORCEINLINE
svuint32_t
MlasSveMatMulAddU32(svuint32_t Acc, svuint8_t A, svuint8_t B)
{
    return svmmla_u32(Acc, A, B);
}

#ifndef __clang__
#pragma GCC pop_options
#endif