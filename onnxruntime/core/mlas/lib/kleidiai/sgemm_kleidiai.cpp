//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#include "mlasi_kleidiai.h"

bool UseSME2 = MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2();

size_t
MLASCALL
ArmKleidiAI::MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
)
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    TransA - Supplies the transpose operation on A matrix

    TransB - Supplies the transpose operation on B matrix

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    if (TransA != CblasNoTrans ||  N == 0  || K == 0) {
        return 0;
    }
    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    size_t bytes = 0;

    if (TransA == CblasNoTrans) {
        switch (TransB) {
            case CblasNoTrans:
                bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
                break;
            case CblasTrans:
                bytes = kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(N, K);
                break;
            default:
                return 0;
        }
    } else {
        return 0;
    }

    return bytes;
}

bool
MLASCALL
ArmKleidiAI::MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
)
/*++

Routine Description:

    This routine packs the contents of matrix B to the destination buffer. The
    destination buffer should be sized based on MlasGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    Returns true if the packing operation was handled by KleidiAI.
    Returns false if the configuration requires a fallback to the default MLAS implementation.

--*/
{
    if (N == 0 || K == 0) {
        return false;
    }

    if (TransA == CblasNoTrans) {
        const size_t nr = UseSME2 ? kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                                  : kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
        const size_t kr = UseSME2 ? kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                                  : kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
        const size_t sr = UseSME2 ? kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                                  : kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();

        // pass zeroed bias values
        const std::vector<float> bias(N);

        switch (TransB) {
            case CblasNoTrans:
                kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, bias.data(), nullptr, PackedB, 0, nullptr);
                break;
            case CblasTrans:
                kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, bias.data(), nullptr, PackedB, 0, nullptr);
                break;
            default:
                return false;
        }
        return true;
    }
    else{
        return false;
    }
}

bool
MLASCALL
ArmKleidiAI::MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine performs a batched matrix multiplication (GEMM) operation using KleidiAI kernels.
    It handles both packed and unpacked inputs and manages tiling and kernel selection depending on
    SME2 availability. If packing is needed, it prepares the required buffers and invokes the
    appropriate left-hand side (LHS) and right-hand side (RHS) pack functions.

    The function also applies alpha and beta scaling to the result, supports efficient memcpy
    paths where possible, and dispatches tile-level GEMM work using multithreading.

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and rows of matrix B.

    Data - Supplies a pointer to the MLAS_SGEMM_DATA_PARAMS array containing per-batch input/output pointers and parameters.

    BatchSize - Supplies the number of independent GEMM computations to perform in the batch.

    ThreadPool - Supplies the thread pool to parallelize computation across batches and tiles.

Return Value:

    Returns true if the GEMM operation was handled by KleidiAI.
    Returns false if the configuration requires a fallback to the default MLAS implementation.

--*/

{
    if (M == 0 || N == 0) {
        return true;
    }

    if (Data->alpha == 0.0f) {
        if (Data->beta == 0.0f) {
            for (size_t i = 0; i < M; ++i) {
                std::fill_n(Data->C + i * Data->ldc, N, 0.0f);
            }
        } else if (Data->beta != 1.0f) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    Data->C[i * Data->ldc + j] *= Data->beta;
                }
            }
        }
        return true;
    }

    if (K == 0) {
        if (Data->beta == 0.0f) {
            for (size_t i = 0; i < M; ++i) {
                std::fill_n(Data->C + i * Data->ldc, N, 0.0f);
            }
        } else if (Data->beta != 1.0f) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    Data->C[i * Data->ldc + j] *= Data->beta;
                }
            }
        }
        return true;
    }

    const size_t mr = UseSME2 ? kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                              : kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t kr = UseSME2 ? kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                              : kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t sr = UseSME2 ? kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                              : kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();

    size_t m_step = UseSME2 ? kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                            : kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    size_t n_step = UseSME2 ? kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa()
                            : kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();

    if ((M < m_step && N < n_step) && !Data->BIsPacked) {
        // Fallback to MLAS
        return false;
    }

    std::vector<MLAS_SGEMM_DATA_PARAMS> KaiPackedData;
    KaiPackedData.resize(BatchSize);

    size_t LhsPackedStride = 0;
    std::byte* LhsPackedData = nullptr;

    LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
    auto LhsPacked = std::make_unique<std::byte[]>(LhsPackedStride * BatchSize);
    LhsPackedData = LhsPacked.get();

    std::unique_ptr<std::byte[]> RhsPacked{nullptr};

    // It is assumed all B batches require packing or not
    if (Data[0].BIsPacked) {
        // We have already decided the matmul variant we are using, before having values for M,N,K
        MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {
            std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);
            kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);
            KaiPackedData[batch_idx].A = reinterpret_cast<const float*>(LhsPackedPtr);
            KaiPackedData[batch_idx].B = Data[batch_idx].B;
        });
    } else {
        // Multithread pack lhs and rhs
        size_t RhsPackedStride = 0;
        std::byte* RhsPackedData = nullptr;

        RhsPackedStride = ArmKleidiAI::MlasGemmPackBSize(TransA, TransB, N, K);
        RhsPacked = std::make_unique<std::byte[]>(RhsPackedStride * BatchSize);
        RhsPackedData = RhsPacked.get();

        MlasTrySimpleParallel(ThreadPool, BatchSize * 2, [&](ptrdiff_t batch_idx) {
            // lhs odd, rhs even
            if (batch_idx & 0x1) {
                batch_idx >>= 1;

                std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);

                kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);

                KaiPackedData[batch_idx].A = reinterpret_cast<const float*>(LhsPackedPtr);
            } else {
                batch_idx >>= 1;

                std::byte* RhsPackedPtr = &(RhsPackedData[RhsPackedStride * batch_idx]);

                ArmKleidiAI::MlasGemmPackB(TransA, TransB, N, K, reinterpret_cast<const float*>(Data[batch_idx].B), Data[batch_idx].ldb, RhsPackedPtr);

                KaiPackedData[batch_idx].B = reinterpret_cast<const float*>(RhsPackedPtr);
            }
        });
    }

    // tile iteration dimensions
    std::array<size_t, 3> dim;
    dim[0] = BatchSize;                  // B
    dim[1] = MlasDivRoundup(M, m_step);  // M
    dim[2] = MlasDivRoundup(N, n_step);  // N

    // Minimize the kernel call count for the number of available threads
    auto RequiredTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0] * dim[1] * dim[2]);

    // scale required tiles over available tile processors
    dim[1] = MlasDivRoundup(RequiredTiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(RequiredTiles * dim[2], dim[1] * dim[2]);

    // compute new step sizes
    m_step *= MlasDivRoundup(MlasDivRoundup(M, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(N, dim[2]), n_step);

    // update tile iterations
    dim[1] = MlasDivRoundup(M, m_step);
    dim[2] = MlasDivRoundup(N, n_step);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [=](ptrdiff_t tid) {
        // compute B,M,N index from iteration index
        ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
        ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
        ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

        // Get rhs tile, B
        const size_t rhs_packed_offset =
            UseSME2 ? kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(NIdx * n_step, K)
                    : kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(NIdx * n_step, K);

        auto BTile = reinterpret_cast<const void*>(
            reinterpret_cast<const std::byte*>(KaiPackedData[BIdx].B) + rhs_packed_offset
        );

        // Get lhs tile, A
        const size_t lhs_packed_offset =
            UseSME2 ? kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(MIdx * m_step, K)
                    : kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(MIdx * m_step, K);

        auto ATile = reinterpret_cast<const float*>(
            reinterpret_cast<const std::byte*>(KaiPackedData[BIdx].A) + lhs_packed_offset
        );

        auto TileSizeM = (MIdx + 1) * m_step > M ? (M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > N ? (N - NIdx * n_step) : n_step;

        // Get result tile, C
        auto CTile = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(Data[BIdx].C) +
            MIdx * m_step * Data[BIdx].ldc * sizeof(float) +
            NIdx * n_step * sizeof(float)
        );
        // Allocate temporary buffer for raw A*B result
        std::vector<float> OutputTile(TileSizeM * TileSizeN, 0.0f);
        float* temp_tile = OutputTile.data();

        if (UseSME2) {
        kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
            TileSizeM,
            TileSizeN,
            K,
            ATile, BTile, temp_tile,
            TileSizeN * sizeof(float), sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );
        } else {
        kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
            TileSizeM,
            TileSizeN,
            K,
            ATile, BTile, temp_tile,
            TileSizeN * sizeof(float), sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );
        }

        // Final output tile pointer
        float* dst_tile = reinterpret_cast<float*>(CTile);

        // quick copy of data in cases where we are not scaling or accumulating anything
        // with bounds checking on tile sizing to ensure the data fits in the memory block
        bool can_memcpy = (
            Data[BIdx].alpha == 1.0f &&
            Data[BIdx].beta == 0.0f &&
            Data[BIdx].ldc == TileSizeN &&
            MIdx * m_step + TileSizeM <= M &&
            NIdx * n_step + TileSizeN <= N &&
            TileSizeM != 0 &&
            TileSizeN != 0);
        if (can_memcpy) {
            std::memcpy(dst_tile, temp_tile, TileSizeM * TileSizeN * sizeof(float));
            return true;
        }

        float alpha = Data[BIdx].alpha;
        float beta = Data[BIdx].beta;
        size_t ldc = Data[BIdx].ldc;

        for (size_t i = 0; i < TileSizeM; ++i) {
            for (size_t j = 0; j < TileSizeN; ++j) {
                const size_t temp_idx = i * TileSizeN + j;
                const size_t dst_idx = i * ldc + j;

                float ab = temp_tile[temp_idx];
                float c_orig = dst_tile[dst_idx];

                if (alpha == 1.0f && beta == 0.0f) {
                    dst_tile[dst_idx] = ab;
                } else if (alpha == 1.0f) {
                    dst_tile[dst_idx] = ab + beta * c_orig;
                } else if (beta == 0.0f) {
                    dst_tile[dst_idx] = alpha * ab;
                } else {
                    dst_tile[dst_idx] = alpha * ab + beta * c_orig;
                }
            }
        }
        return true;
    });
    return true;
}
