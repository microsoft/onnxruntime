//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../mlasi.h"

// Fix to ensure compatibility with MSVC build
#if defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif
namespace ArmKleidiAI {
// By default we should try for SME2 first before falling back to SME.
inline const bool UseSME2 = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2();

//
// Buffer packing routines.
//

size_t
MLASCALL
MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    );

bool
MLASCALL
MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    );

bool
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    );

size_t
MLASCALL
MlasDynamicQgemmPackBSize(
    size_t N,
    size_t K
);

void
MLASCALL
MlasDynamicQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
);

//pack symmetric quantized B and dynamic quantized A
void
MLASCALL
MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    );

bool
MLASCALL
MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
                size_t Dimensions,
                size_t BatchCount,
                size_t GroupCount,
                size_t InputChannels,
                const int64_t* InputShape,
                const int64_t* KernelShape,
                const int64_t* DilationShape,
                const int64_t* Padding,
                const int64_t* StrideShape,
                const int64_t* OutputShape,
                size_t FilterCount,
                const MLAS_ACTIVATION* Activation,
                size_t* WorkingBufferSize,
                float Beta,
                MLAS_THREADPOOL* ThreadPool);

bool
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    );
}

/*++

Routine Description:

    This routine determines if a wraparound will occur when multiplying two size_t variables
    Uses __builtin_mul_overflow if available on the current system and if not falls back
    to a default implementation to check this wraparound.

Arguments:

    a - Supplies the first number to be muliplied.

    b - Supplies the second number to be muliplied.

    out - pointer to a size_t which acts as the return value in success cases.

Return Value:

    Returns false if the operation was successful
    Returns true if wraparound of size_t was detected

--*/
inline bool mul_overflow_size_t_builtin(size_t a, size_t b, size_t* out) {
#if defined(__has_builtin)
#  if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#  endif
#endif
    // Fallback to manual check if builtin not available
    if (b != 0 && a > SIZE_MAX / b) return true;
    if (out) *out = a * b;
    return false;
}
