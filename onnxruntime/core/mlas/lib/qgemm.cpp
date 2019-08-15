/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"

//
// Define the default strides to step through slices of the input matrices.
//

#define MLAS_GEMM_U8U8_STRIDEM              12
#define MLAS_GEMM_U8U8_STRIDEN              128
#define MLAS_GEMM_U8U8_STRIDEK              128

void
MLASCALL
MlasQgemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const uint8_t* B,
    size_t ldb,
    uint8_t offb,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_DECLSPEC_ALIGN(int16_t PanelA[MLAS_GEMM_U8U8_STRIDEM * MLAS_GEMM_U8U8_STRIDEK], 64);
    MLAS_DECLSPEC_ALIGN(uint8_t PanelB[MLAS_GEMM_U8U8_STRIDEN * MLAS_GEMM_U8U8_STRIDEK], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumVector[MLAS_GEMM_U8U8_STRIDEM], 16);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumVector[MLAS_GEMM_U8U8_STRIDEN], 16);

    size_t StrideM = MLAS_GEMM_U8U8_STRIDEM;
    size_t StrideN = MLAS_GEMM_U8U8_STRIDEN;
    size_t StrideK = MLAS_GEMM_U8U8_STRIDEK;

    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = StrideK;

        if (CountK > (K - k)) {
            CountK = K - k;
        }

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = StrideN;

            if (CountN > (N - n)) {
                CountN = N - n;
            }

            MlasPlatform.GemmU8U8CopyPackBRoutine(PanelB, B + n + k * ldb, ldb, CountN, CountK, ColumnSumVector, offa);

            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = StrideM;

                if (CountM > (M - m)) {
                    CountM = M - m;
                }

                MlasPlatform.GemmU8U8CopyPackARoutine(PanelA, A + k + m * lda, lda, CountM, CountK, RowSumVector, offb);

                int16_t* pa = PanelA;
                int32_t* c = C + n + m * ldc;

                int32_t* RowSums = RowSumVector;

                size_t RowsRemaining = CountM;
                size_t RowsHandled;

                size_t KK = (CountK + 1) & (~1U);

                while (RowsRemaining > 0) {

                    RowsHandled = MlasPlatform.GemmU8U8Kernel(pa, PanelB, c, CountK, RowsRemaining, CountN, ldc, RowSums, ColumnSumVector, int32_t(CountK) * offa * offb, k == 0);

                    RowsRemaining -= RowsHandled;
                    c += ldc * RowsHandled;
                    pa += KK * RowsHandled;
                    RowSums += RowsHandled;
                }
            }
        }
    }
}
