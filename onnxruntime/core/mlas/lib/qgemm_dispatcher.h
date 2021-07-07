/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_dispatcher.h

Abstract:

    This module defines dispatcher structure and logic for quantized integer
    matrix/matrix multiply operation (QGEMM).

--*/

#pragma once

#include "mlasi.h"

//
// Quantized integer matrix/matrix dispatch structure.
//

typedef
void
(MLAS_GEMM_U8X8_OPERATION)(
    const MLAS_GEMM_U8X8_SHAPE_PARAMS* Shape,
    const MLAS_GEMM_U8X8_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );

typedef
void
(MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE)(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    );

struct MLAS_GEMM_U8X8_DISPATCH {
    MLAS_GEMM_U8X8_OPERATION* Operation;
    MLAS_GEMM_U8X8_OPERATION* PackedOperation;
    MLAS_GEMM_U8X8_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t PackedK;
    size_t PackedStrideK;
};

MLAS_FORCEINLINE
const MLAS_GEMM_U8X8_DISPATCH*
MlasGemmU8X8GetDispatch(
    bool BIsSigned
)
{
    const MLAS_GEMM_U8X8_DISPATCH* GemmU8X8Dispatch;

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

#if defined(MLAS_TARGET_AMD64)
    if (BIsSigned) {
        GemmU8X8Dispatch = MlasPlatform.GemmU8S8Dispatch;
    }
    else {
        GemmU8X8Dispatch = MlasPlatform.GemmU8U8Dispatch;
    }
#elif defined(MLAS_SSE2_INTRINSICS)
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchSse;
#elif defined(MLAS_NEON64_INTRINSICS)
    GemmU8X8Dispatch = MlasPlatform.GemmU8X8Dispatch;
#elif defined(MLAS_NEON32_INTRINSICS) && !defined(_MSC_VER)
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchNeon;
#else
    GemmU8X8Dispatch = &MlasGemmU8X8DispatchDefault;
#endif

    return GemmU8X8Dispatch;
}
