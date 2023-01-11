/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm.h

Abstract:

    This module defines the set of template functions to implement a kernel of
    half precision matrix/matrix multiply operation (QGEMM).

    To implement a new kernel, template functions below need to be specialized:
        MlasHalfGemmCopyPackB
        MlasHalfGemmKernel
    Specialization of MlasHalfGemmTryGemvKernel is optional.

    MlasHalfGemmOperation and MlasHalfGemmPackedOperation are shared kernel drivers.


--*/

#pragma once

#include <cstdlib>
#include <string>

#include "mlasi.h"


/**
 * @brief Define the default striding parameters for
 *        the half precision gemm operation
 */
struct MLAS_HALF_GEMM_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

template<typename KernelType>
MLAS_FORCEINLINE
bool
MlasHalfGemmTryGemvKernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    size_t ldb,
    MLAS_FP16* C,
    size_t CountK,
    size_t CountN
)
{
    MLAS_UNREFERENCED_PARAMETER(A);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(C);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    MLAS_UNREFERENCED_PARAMETER(CountN);

    return false;
}

template<typename KernelType>
void
MlasHalfGemmCopyPackB(
    MLAS_FP16* D,
    const MLAS_FP16* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
);

template<typename KernelType>
void
MlasHalfGemmKernel(
    const size_t CountM,
    const size_t CountN,
    const size_t CountK,
    const MLAS_FP16* A,
    const size_t lda,
    const MLAS_FP16* B,
    const size_t ldb,
    MLAS_FP16* C,
    size_t ldc,
    const MLAS_FP16* Bias,
    const bool ZeroMode
);


template<typename KernelType>
MLAS_FORCEINLINE
void 
MlasHalfGemmThreadInit()
{
    if (!KernelType::PackNeeded) {
        return;
    } 
    constexpr MLAS_HALF_GEMM_STRIDES Strides = KernelType::Strides;
    constexpr size_t packBSize = UpAlignSize(Strides.N * Strides.K * sizeof(MLAS_FP16));

    MlasThreadedBufAlloc(packBSize);
}


template<typename KernelType>
void
MlasHalfGemmOperation(
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
{
    MlasHalfGemmThreadInit<KernelType>();

    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    const MLAS_FP16* A = Data->A + RangeStartM * lda;
    const MLAS_FP16* B = Data->B + RangeStartN;
    const MLAS_FP16* Bias = Data->Bias + RangeStartN;
    MLAS_FP16* C = Data->C + RangeStartM * ldc + RangeStartN;

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((RangeCountM == 1) && (Data->OutputProcessor == nullptr)) {
        if (MlasHalfGemmTryGemvKernel<KernelType>(A, B, ldb, C, K, RangeCountN)) {
            return;
        }
    }

    if (!KernelType::PackNeeded) {
        // We are not restricted by packing panel size, so simpler tiling

        auto pa = A;
        auto c = C;
        size_t RowsRemaining = RangeCountM;

        while (RowsRemaining > 0) {
            MlasHalfGemmKernel<KernelType>(
                RowsRemaining,
                RangeCountN,
                K,
                pa,
                lda,
                B,
                ldb,
                c, 
                ldc,
                Bias,
                true);

            size_t RowsHandled = std::min(RowsRemaining, KernelType::KernelMaxM);

            if (Data->OutputProcessor != nullptr) {
                Data->OutputProcessor->Process(
                    Data->C,
                    RangeStartM + RangeCountM - RowsRemaining,
                    RangeStartN,
                    RowsHandled,
                    RangeCountN,
                    Data->ldc);
            }

            c += ldc * RowsHandled;
            pa += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }

        return;
    } 

    //
    // Three dimensional tiling due to limited packing panel size
    //

    constexpr MLAS_HALF_GEMM_STRIDES Strides = KernelType::Strides;
    MLAS_FP16* PanelB = reinterpret_cast<MLAS_FP16*>(ThreadedBufHolder.get());

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < RangeCountN; n += CountN) {

            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            MlasHalfGemmCopyPackB<KernelType>(
                PanelB,
                B + n,
                ldb,
                CountN,
                CountK);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            MLAS_FP16* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < RangeCountM; m += CountM) {

                CountM = std::min(RangeCountM - m, Strides.M);

                const MLAS_FP16* pa = A + m * lda;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {
                    MlasHalfGemmKernel<KernelType>(
                        RowsRemaining,
                        CountN,
                        CountK,
                        pa,
                        lda,
                        PanelB,
                        0, // ldb not needed for packed B
                        c,
                        ldc,
                        Bias,
                        ZeroMode);

                    size_t RowsHandled = std::min(RowsRemaining, KernelType::KernelMaxM);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc);
                    }

                    c += ldc * RowsHandled;
                    pa += lda * RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}


template<typename KernelType>
void
MlasHalfGemmPackedOperation(
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
{
    const size_t lda = Data->lda;
    const size_t ldc = Data->ldc;

    auto pa = (Data->A) + RangeStartM * lda;
    const size_t PackedCountK = (K + KernelType::PackedK - 1) / KernelType::PackedK;
    const MLAS_FP16* b = Data->B + RangeStartN * KernelType::PackedK * PackedCountK;
    const MLAS_FP16* Bias = Data->Bias + RangeStartN;
    auto* c = C;

    size_t RowsRemaining = RangeCountM;
    while (RowsRemaining > 0) {
        MlasHalfGemmKernel<KernelType>(
            RowsRemaining,
            RangeCountN,
            K,
            pa,
            lda,
            b,
            0, // packed B ldb not needed
            c, 
            ldc,
            Bias,
            true);

        size_t RowsHandled = std::min(RowsRemaining, KernelType::KernelMaxM);

        if (Data->OutputProcessor != nullptr) {
            Data->OutputProcessor->Process(
                Data->C,
                RangeStartM + RangeCountM - RowsRemaining,
                RangeStartN + n,
                RowsHandled,
                RangeCountN,
                Data->ldc);
        }

        c += ldc * RowsHandled;
        pa += lda * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
}




//
// dispatch structure.
//

typedef
void
(MLAS_HALF_GEMM_OPERATION)(
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    );


typedef
void
(MLAS_HALF_GEMM_COPY_PACKB_ROUTINE)(
    MLAS_FP16* D,
    const MLAS_FP16* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    );

struct MLAS_HALF_GEMM_DISPATCH {
    MLAS_HALF_GEMM_OPERATION* Operation;
    MLAS_HALF_GEMM_OPERATION* PackedOperation;
    MLAS_HALF_GEMM_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    size_t StrideM;
};

extern const MLAS_HALF_GEMM_DISPATCH MlasHalfGemmDispatchDefault;

MLAS_FORCEINLINE
const MLAS_HALF_GEMM_DISPATCH*
MlasHalfGemmGetDispatch()
{
    return &MlasHalfGemmDispatchDefault;
}
