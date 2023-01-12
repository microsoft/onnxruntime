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
       MlasHalfGemmConvertPackA
       MlasHalfGemmConvertPackB
       MlasHalfGemmPackedBOffset
       MlasHalfGemmPackedBLeadingDim
       MlasHalfGemmKernel

    MlasHalfGemmOperation is the shared kernel driver.

    A kernel type should define the following constants:
        bool PackNeeded;         Whether fp16 B needs to be packed
        size_t KernelMaxM;       Max # rows the vectorized kernel can process
        size_t PackedK;          Packed alignment on the K dim (power of 2)
        MLAS_HALF_GEMM_STRIDES Strides{128, 128, 128};
--*/

#pragma once

#include <cstdlib>
#include <cassert>
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

/**
 * @brief Packing function for fp16 B matrix
 * 
 * @tparam KernelType 
 * @param[out] D         Address of packing buffer
 * @param[in]  B         Address of source matrix B
 * @param[in]  ldb       Leading dimension of B 
 * @param[in]  CountN    # of column to pack 
 * @param[in]  CountK    # of rows to pack
*/
template<typename KernelType>
MLAS_FORCEINLINE 
void
MlasHalfGemmCopyPackB(
    MLAS_FP16* D,
    const MLAS_FP16* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    MLAS_UNREFERENCED_PARAMETER(D);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    // No packing needed by default
}

/**
 * @brief Convert fp32 matrix A to fp16 and pack the data
 * 
 * @tparam KernelType 
 * @param[out] D        Address of the packing buffer
 * @param[in]  A        Address of fp32 matrix A
 * @param[in]  lda      leading dimension of A
 * @param[in]  CountM   # of rows to pack 
 * @param[in]  CountK   # of columns to pack
*/
template<typename KernelType>
void
MlasHalfGemmConvertPackA(
    MLAS_FP16* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
);

/**
 * @brief Convert fp32 matrix B to fp16 and pack the data
 *
 * @tparam KernelType
 * @param[out] D         Address of packing buffer
 * @param[in]  B         Address of source matrix B in fp32
 * @param[in]  ldb       Leading dimension of B
 * @param[in]  CountN    # of column to pack
 * @param[in]  CountK    # of rows to pack
 */
template <typename KernelType>
void
MlasHalfGemmConvertPackB(
    MLAS_FP16* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
);

/**
 * @brief Find the location of [StartK, StartN] in packed B buffer
 * 
 * @tparam KernelType 
 * @param PackedB 
 * @param DimN 
 * @param DimK 
 * @param StartN 
 * @param StartK 
 * @return  Address of PackedB[StartK, StartN]
*/
template <typename KernelType>
MLAS_FORCEINLINE
const MLAS_FP16*
MlasHalfGemmPackedBOffset(
    const MLAS_FP16* PackedB,
    size_t DimN,
    size_t DimK,
    size_t StartN,
    size_t StartK)
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return PackedB + StartK * DimN + StartN;
}

/**
 * @brief leading dimension of the packed B buffer
 *        Related to how B is packed
 * @tparam KernelType 
 * @param DimN 
 * @param DimK 
 * @return leading dimension of the packed B buffer
*/
template <typename KernelType>
MLAS_FORCEINLINE
size_t
MlasHalfGemmPackedBLeadingDim(
    size_t DimN,
    size_t DimK)
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return DimN;
}


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
MlasHalfGemmNoPackOperation(
    const size_t N,
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
{
    //
    // Optimize for the special case where no packing is needed.
    // Simpler tiling as we are not restricted by packing panel size
    //

    const size_t lda = Data->lda;
    size_t ldb = Data->ldb;  // 0 if prepacked
    const size_t ldc = Data->ldc;

    const MLAS_FP16* pa = reinterpret_cast<const MLAS_FP16*>(Data->A) + RangeStartM * lda;
    const MLAS_FP16* B;
    if (ldb == 0) {
        B = MlasHalfGemmPackedBOffset<KernelType>(
            reinterpret_cast<const MLAS_FP16*>(Data->B),
            N,
            K,
            RangeStartN,
            0);
        ldb = MlasHalfGemmPackedBLeadingDim<KernelType>(N, K);
    } else {
        B = reinterpret_cast<const MLAS_FP16*>(Data->B) + RangeStartN;
    }

    const MLAS_FP16* Bias = Data->Bias + RangeStartN;
    MLAS_FP16* c = Data->C + RangeStartM * ldc + RangeStartN;

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
}


template<typename KernelType>
void
MlasHalfGemmOperation(
    const size_t N,
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
    )
{
    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    if (!Data->AIsfp32 && (ldb == 0 || !KernelType::PackNeeded && !Data->BIsfp32)) {
        // No packing needed, use a simpler driver instead
        MlasHalfGemmNoPackOperation<KernelType>(
            N,
            K,
            Data,
            RangeStartM,
            RangeCountM,
            RangeStartN,
            RangeCountN);
        return;
    }

    const MLAS_FP16* Bias = Data->Bias + RangeStartN;
    MLAS_FP16* C = Data->C + RangeStartM * ldc + RangeStartN;

    //
    // Three dimensional tiling due to limited packing panel size
    //
    constexpr MLAS_HALF_GEMM_STRIDES Strides = KernelType::Strides;
    constexpr size_t packASize = UpAlignSize(Strides.M * Strides.K * sizeof(MLAS_FP16));
    constexpr size_t packBSize = UpAlignSize(Strides.N * Strides.K * sizeof(MLAS_FP16));
    MlasThreadedBufAlloc(packASize + packBSize);

    uint8_t* p = ThreadedBufHolder.get();
    MLAS_FP16* PanelA = reinterpret_cast<MLAS_FP16*>(p);
    p += packASize;
    MLAS_FP16* PanelB = reinterpret_cast<MLAS_FP16*>(p);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;
    for (size_t k = 0; k < K; k += CountK) {
        CountK = std::min(K - k, Strides.K);
        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) / KernelType::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //
            size_t ld_pb;
            const MLAS_FP16* pb;
            if (ldb == 0) {
                // Already packed
                pb = MlasHalfGemmPackedBOffset<KernelType>(
                    reinterpret_cast<const MLAS_FP16*>(Data->B),
                    N,
                    K,
                    RangeStartN + n,
                    k);
                ld_pb = MlasHalfGemmPackedBLeadingDim<KernelType>(N, K);
            } else if (Data->BIsfp32) {
                MlasHalfGemmConvertPackB<KernelType>(
                    PanelB,
                    reinterpret_cast<const float*>(Data->B) + ldb * k + RangeStartN + n,
                    ldb,
                    CountN,
                    CountK);
                pb = PanelB;
                ld_pb = MlasHalfGemmPackedBLeadingDim<KernelType>(CountN, CountK);
            } else if (KernelType::PackNeeded) {
                MlasHalfGemmCopyPackB<KernelType>(
                    PanelB,
                    reinterpret_cast<const MLAS_FP16*>(Data->B) + ldb * k + RangeStartN + n,
                    ldb,
                    CountN,
                    CountK);
                pb = PanelB;
                ld_pb = MlasHalfGemmPackedBLeadingDim<KernelType>(CountN, CountK);
            } else {
                // fp16, and no packing needed
                pb = reinterpret_cast<const MLAS_FP16*>(Data->B) + ldb * k + RangeStartN + n;
                ld_pb = ldb;
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            MLAS_FP16* c = C + n;
            size_t CountM;
            for (size_t m = 0; m < RangeCountM; m += CountM) {
                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //
                const MLAS_FP16* pa;
                size_t ld_pa;
                if (Data->AIsfp32) {
                    MlasHalfGemmConvertPackA<KernelType>(
                        PanelA,
                        reinterpret_cast<const float*>(Data->A) + (RangeStartM + m) * lda + k,
                        lda,
                        CountM,
                        CountK);
                    pa = PanelA;
                    ld_pa = KernelType::PackedK * PackedCountK;
                } else {
                    pa = reinterpret_cast<const MLAS_FP16*>(Data->A) + (RangeStartM + m) * lda + k;
                    ld_pa = lda;
                }

                size_t RowsRemaining = CountM;
                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {
                    MlasHalfGemmKernel<KernelType>(
                        RowsRemaining,
                        CountN,
                        CountK,
                        pa,
                        ld_pa,
                        pb,
                        ld_pb,
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
                    pa += ld_pa * RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }
    }
}


//
// dispatch structure.
//

typedef
void
(MLAS_HALF_GEMM_OPERATION)(
    const size_t N,
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

typedef
void
(MLAS_HALF_GEMM_CONVERT_PACKB_ROUTINE)(
    MLAS_FP16* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    );

struct MLAS_HALF_GEMM_DISPATCH {
    MLAS_HALF_GEMM_OPERATION* Operation;
    MLAS_HALF_GEMM_COPY_PACKB_ROUTINE* CopyPackBRoutine;
    MLAS_HALF_GEMM_CONVERT_PACKB_ROUTINE* ConvertPackBRoutine;
    size_t PackededK;
    size_t StrideM;
};

extern const MLAS_HALF_GEMM_DISPATCH MlasHalfGemmDispatchDefault;

MLAS_FORCEINLINE
const MLAS_HALF_GEMM_DISPATCH*
MlasHalfGemmGetDispatch()
{
    return &MlasHalfGemmDispatchDefault;
}
