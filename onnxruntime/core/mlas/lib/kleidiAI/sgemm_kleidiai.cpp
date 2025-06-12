// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#include "mlasi_kleidiai.h"

size_t
MLASCALL
ARMKleidiAI::MlasGemmPackBSize(
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
    if (N == 0 || K == 0) {
        // no computation to do
        return 0;
    }

    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    size_t bytes = 0;

    if (TransA == CblasNoTrans && MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        switch (TransB) {
            case CblasNoTrans:
                bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
                break;
            case CblasTrans:
                bytes = kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(N, K);
                break;
            default:
                throw ARMKleidiAI::NotSupported();
                break;
        }
    } else {
        throw ARMKleidiAI::NotSupported();
    }

    return bytes;
}

void
MLASCALL
ARMKleidiAI::MlasGemmPackB(
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

    None.

--*/
{
    if (N == 0 || K == 0) {
        // no computation to do
        return;
    }

    if (TransA == CblasNoTrans && MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        const size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

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
                throw ARMKleidiAI::NotSupported();
                break;
        }
    } else {
        throw ARMKleidiAI::NotSupported();
    }
}

void
MLASCALL
ARMKleidiAI::MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (M == 0 || N == 0 || K == 0) {
        //no computation to do
        return;
    }

    if (TransA == CblasNoTrans && MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

        auto m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
        auto n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

        std::vector<MLAS_SGEMM_DATA_PARAMS> KaiPackedData;
        KaiPackedData.resize(BatchSize);

        size_t LhsPackedStride = 0;
        std::byte* LhsPackedData = nullptr;

        LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
        auto LhsPacked = std::make_unique_for_overwrite<std::byte[]>(LhsPackedStride * BatchSize);
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

            RhsPackedStride = ARMKleidiAI::MlasGemmPackBSize(TransA, TransB, N, K);
            RhsPacked = std::make_unique_for_overwrite<std::byte[]>(RhsPackedStride * BatchSize);
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

                    ARMKleidiAI::MlasGemmPackB(TransA, TransB, N, K, reinterpret_cast<const float*>(Data[batch_idx].B), Data[batch_idx].ldb, RhsPackedPtr);

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
                kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(NIdx * n_step, K);

            auto BTile = reinterpret_cast<const void*>(
                reinterpret_cast<const std::byte*>(KaiPackedData[BIdx].B) + rhs_packed_offset
            );

            // Get lhs tile, A
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(MIdx * m_step, K);

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

            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
                TileSizeM,
                TileSizeN,
                K,
                ATile, BTile, CTile,
                Data[BIdx].ldc * sizeof(float), sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
            );
        });

    } else {
        throw ARMKleidiAI::NotSupported();
    }
}
