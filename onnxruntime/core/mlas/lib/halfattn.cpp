/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn.cpp

Abstract:

    This module implements the attention products over fp16 key/value caches
    that accumulate in fp32 (MlasHalfAttnQK, MlasHalfAttnPV), with portable
    loops for platforms without a vector kernel.

--*/

#include "halfattn.h"

bool
MLASCALL
MlasHalfAttnAccelerationSupported(
    void
    )
{
    return GetMlasPlatform().HalfAttnDispatch != nullptr;
}

void
MLASCALL
MlasHalfAttnQK(
    size_t M,
    size_t N,
    size_t K,
    float Alpha,
    const MLAS_FP16* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
    )
{
    if (M == 0 || N == 0) {
        return;
    }

    const auto* dispatch = GetMlasPlatform().HalfAttnDispatch;
    if (dispatch != nullptr) {
        dispatch->QK(M, N, K, Alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * lda + k].ToFloat() * B[n * ldb + k].ToFloat();
            }
            C[m * ldc + n] = Alpha * sum;
        }
    }
}

void
MLASCALL
MlasHalfAttnPV(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const MLAS_FP16* B,
    size_t ldb,
    float* C,
    size_t ldc
    )
{
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    const auto* dispatch = GetMlasPlatform().HalfAttnDispatch;
    if (dispatch != nullptr) {
        dispatch->PV(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }

    for (size_t m = 0; m < M; ++m) {
        float* c = C + m * ldc;
        for (size_t k = 0; k < K; ++k) {
            const float a = A[m * lda + k];
            const MLAS_FP16* b = B + k * ldb;
            for (size_t n = 0; n < N; ++n) {
                c[n] += a * b[n].ToFloat();
            }
        }
    }
}
