/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   Gelu.cpp

Abstract:

    This module contains  Gelu helper functions .

--*/

#include "fp16_common.h"
#if defined(MLAS_NEON_INTRINSICS)
#include "erf_neon_fp16.h"
#endif

#ifdef MLAS_USE_SVE
#include "sve/mlasi_sve.h"
#endif

void
MLASCALL
MlasNeonGeluF16Kernel(
    const MLAS_FP16* input,
    MLAS_FP16* output,
    MLAS_FP16* temp,
    int64_t count,
    const std::string& algo
);