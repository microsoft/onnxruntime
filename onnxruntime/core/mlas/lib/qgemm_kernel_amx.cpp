/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_amx.cpp

Abstract:

    This module implements QGEMM kernels for amx.

--*/

#include "mlasi.h"
#include "qgemm.h"

#define M_ACC 2              // Number of C accumulators spanning the M dimension
#define N_ACC 2              // Number of C accumulators spanning the N dimension
#define TILE_M 16            // Number of rows in an A or C tile
#define TILE_N 16            // Number of columns in a B or C tile
#define TILE_K 64            // Number of columns in an A tile or rows in a B tile
typedef int8_t type_t;       // The type of the data being operated on
typedef int32_t res_type_t;  // The data type of the result
#define _tdp _tile_dpbuud    // Multiplication operator

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

#define KPACK (4 / sizeof(type_t))  // Vertical K packing into Dword

struct MLAS_GEMM_U8U8_KERNEL_AMX {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = KPACK;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{256, 256, 256};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{256, 256, 256};
};

constexpr size_t MLAS_GEMM_U8U8_KERNEL_AMX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8U8_KERNEL_AMX::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8U8_KERNEL_AMX::PackedStrides;

#if 0
template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    }

    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_DEFAULT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_DEFAULT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}
#endif

template <>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8U8_KERNEL_AMX>(MLAS_GEMM_U8U8_KERNEL_AMX::PackedAType* D,
                                                  const uint8_t* A,
                                                  size_t lda,
                                                  size_t CountM,
                                                  size_t CountK,
                                                  int32_t* RowSumBuffer,
                                                  bool AIsSigned)
{
    printf("Enter MlasGemmQuantCopyPackA<MLAS_GEMM_U8U8_KERNEL_AMX>\n");

    MLAS_UNREFERENCED_PARAMETER(AIsSigned);

    for (size_t m = 0; m < CountM; ++m) {
        RowSumBuffer[m] = 0;
        for (size_t k = 0; k < CountK; ++k) {
            D[m * CountK + k] = A[m * lda + k];
            RowSumBuffer[m] += A[m * lda + k];
            // printf("%d ", B_mem[k/KPACK][n][k%KPACK]);
        }
    }
}

template <>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8U8_KERNEL_AMX>(MLAS_GEMM_U8U8_KERNEL_AMX::PackedBType* D,
                                                  const uint8_t* B,
                                                  size_t ldb,
                                                  size_t CountN,
                                                  size_t CountK,
                                                  int32_t* ColumnSumBuffer,
                                                  bool BIsSigned)
{
    printf("Enter MlasGemmQuantCopyPackB<MLAS_GEMM_U8U8_KERNEL_AMX>\n");

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    for (size_t n = 0; n < CountN; ++n) {
        ColumnSumBuffer[n] = 0;
        for (size_t k = 0; k < CountK; ++k) {
            ColumnSumBuffer[n] += B[k * ldb + n];
        }
    }

    for (size_t k = 0; k < CountK; ++k) {
        for (size_t n = 0; n < CountN; ++n) {
            *(D + (k / KPACK) * CountN * KPACK + n * KPACK + k % KPACK) = B[k * ldb + n];
            // printf("%d ", B_mem[k/KPACK][n][k%KPACK]);
        }
    }
}

template <>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8U8_KERNEL_AMX>(const MLAS_GEMM_U8U8_KERNEL_AMX::PackedAType* A,
                                               const MLAS_GEMM_U8U8_KERNEL_AMX::PackedBType* B,
                                               int32_t* C,
                                               size_t PackedCountK,
                                               size_t CountM,
                                               size_t CountN,
                                               size_t ldc,
                                               const int32_t* RowSumBuffer,
                                               const int32_t* ColumnSumBuffer,
                                               const int32_t* ZeroPointB,
                                               bool ZeroMode)
{
    printf("Enter MlasGemmQuantKernel<MLAS_GEMM_U8U8_KERNEL_AMX>\n");
    MLAS_UNREFERENCED_PARAMETER(RowSumBuffer);
    MLAS_UNREFERENCED_PARAMETER(ColumnSumBuffer);
    MLAS_UNREFERENCED_PARAMETER(ZeroPointB);
    MLAS_UNREFERENCED_PARAMETER(ZeroMode);

    // A_mem and B_mem are blocks within the original matrix, C_mem is the original matrix
    size_t K = PackedCountK * KPACK;
    // size_k M_ACC, N_ACC;
    // M_ACC = N_ACC = 2;
    // typedef int8_t type_t;      // The type of the data being operated on
    // typedef int32_t res_type_t; // The data type of the result
    //#define _tdp _tile_dpbssd   // Multiplication operator

    for (size_t n = 0; n < CountN; n += N_ACC * TILE_N) {
        for (size_t m = 0; m < CountM; m += M_ACC * TILE_M) {
            /*for (int n_acc = 0; n_acc < N_ACC; ++n_acc)
              for (int m_acc = 0; m_acc < M_ACC; ++m_acc)
                _tile_zero(tC(m_acc,n_acc));*/
            _tile_zero(TMM0);
            _tile_zero(TMM1);
            _tile_zero(TMM2);
            _tile_zero(TMM3);

            // const int tB = TMM6;
            // const int tA = TMM4;

            for (size_t k = 0; k < K; k += TILE_K) {
                // Preloading A tiles
                //_tile_loadd(TMM4, &A_mem[m + 0*TILE_M][k], K*sizeof(type_t));
                //_tile_loadd(TMM5, &A_mem[m + 1*TILE_M][k], K*sizeof(type_t));

                for (int n_acc = 0; n_acc < N_ACC; ++n_acc) {
                    //_tile_loadd(TMM6, B_mem[k/KPACK][n + n_acc*TILE_N],
                    // N*sizeof(type_t)*KPACK);
                    _tile_loadd(
                        TMM6,
                        (void*)(B + (k / KPACK) * (CountN * KPACK) + (n + n_acc * TILE_N) * KPACK),
                        static_cast<int>(CountN * sizeof(type_t) * KPACK));

                    for (int m_acc = 0; m_acc < M_ACC; ++m_acc) {
                        if (n_acc == 0 && m_acc == 0)
                            //_tile_loadd(TMM4, &A_mem[m + m_acc*TILE_M][k], K*sizeof(type_t));
                            _tile_loadd(TMM4, (void*)(A + (m + 0 * TILE_M) * K + k),
                                        static_cast<int>(K * sizeof(type_t)));
                        else if (n_acc == 0 && m_acc == 1)
                            //_tile_loadd(TMM5, &A_mem[m + m_acc*TILE_M][k], K*sizeof(type_t));
                            _tile_loadd(TMM5, (void*)(A + (m + 1 * TILE_M) * K + k),
                                        static_cast<int>(K * sizeof(type_t)));

                        int tC = m_acc + n_acc * N_ACC;
                        if (tC == TMM0) {
                            if (m_acc)
                                _tdp(TMM0, TMM5, TMM6);
                            else
                                _tdp(TMM0, TMM4, TMM6);
                        }
                        else if (tC == TMM1) {
                            if (m_acc)
                                _tdp(TMM1, TMM5, TMM6);
                            else
                                _tdp(TMM1, TMM4, TMM6);
                        }
                        else if (tC == TMM2) {
                            if (m_acc)
                                _tdp(TMM2, TMM5, TMM6);
                            else
                                _tdp(TMM2, TMM4, TMM6);
                        }
                        else if (tC == TMM3) {
                            if (m_acc)
                                _tdp(TMM3, TMM5, TMM6);
                            else
                                _tdp(TMM3, TMM4, TMM6);
                        }

                        // Store in the same loop instead of doing it at the end
                        if (k == K - TILE_K) {
                            int mc = m + m_acc * TILE_M, nc = n + n_acc * TILE_N;
                            // int tC = m_acc + n_acc * N_ACC;
                            if (tC == TMM0)
                                _tile_stored(TMM0, (void*)(C + mc * ldc + nc),
                                             static_cast<int>(ldc * sizeof(res_type_t)));
                            else if (tC == TMM1)
                                _tile_stored(TMM1, (void*)(C + mc * ldc + nc),
                                             static_cast<int>(ldc * sizeof(res_type_t)));
                            else if (tC == TMM2)
                                _tile_stored(TMM2, (void*)(C + mc * ldc + nc),
                                             static_cast<int>(ldc * sizeof(res_type_t)));
                            else if (tC == TMM3)
                                _tile_stored(TMM3, (void*)(C + mc * ldc + nc),
                                             static_cast<int>(ldc * sizeof(res_type_t)));
                        }
                    }
                }
            }
        }
    }
    return CountM;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8U8DispatchAmx = {
    MlasGemmQuantOperation<MLAS_GEMM_U8U8_KERNEL_AMX>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U8U8_KERNEL_AMX>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8U8_KERNEL_AMX>,
    MLAS_GEMM_U8U8_KERNEL_AMX::PackedK,
    MLAS_GEMM_U8U8_KERNEL_AMX::PackedStrides.K,
    8  // temp
};
