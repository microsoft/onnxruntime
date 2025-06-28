//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

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
    if (N == 0 || K == 0) {
        // no computation to do
        return 0;
    }

    if (TransA != CblasNoTrans || !MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        ::MlasGemmPackBSize(TransA, TransB, N, K); // fallback
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
                ::MlasGemmPackBSize(TransA, TransB, N, K); // fallback
        }
    } else {
        ::MlasGemmPackBSize(TransA, TransB, N, K); // fallback
    }

    return bytes;
}

void
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
                ::MlasGemmPackB(TransA, TransB, N, K, B, ldb, PackedB); // fallback
        }
    }
}

void
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
{
    // Guard against unsupported cases
    extern void MLASCALL MlasGemmBatch(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, size_t, size_t, size_t,const MLAS_SGEMM_DATA_PARAMS*, size_t, MLAS_THREADPOOL*);
    if (M == 0 || N == 0 || K == 0 ||
        TransA != CblasNoTrans ||
        (TransB != CblasNoTrans && !Data[0].BIsPacked) ||
        !MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME())
    {
        // If unsupported case, explicitly call the fallback to default via function pointer
        ::MlasGemmBatch(TransA, TransB, M, N, K, Data, BatchSize, ThreadPool);
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    size_t m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    size_t n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

    if (M < m_step || N < n_step) {
        if (GetMlasPlatform().MlasGemmBatch != ArmKleidiAI::MlasGemmBatch){
            //Fallback to MLAS
            ::MlasGemmBatch(TransA, TransB, M, N, K, Data, BatchSize, ThreadPool);
        }
    }

    // Allocate packed buffer
    const size_t LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
    std::unique_ptr<std::byte[]> LhsPacked(new std::byte[LhsPackedStride * BatchSize]);
    std::vector<MLAS_SGEMM_DATA_PARAMS> KaiPackedData(BatchSize);

    std::unique_ptr<std::byte[]> RhsPacked;
    size_t RhsPackedStride = 0;

    // Serialize packing to avoid nested thread pool contention
    for (size_t i = 0; i < BatchSize; ++i) {
        std::byte* lhs_ptr = LhsPacked.get() + i * LhsPackedStride;
        kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, Data[i].A, Data[i].lda * sizeof(float), lhs_ptr);
        KaiPackedData[i].A = reinterpret_cast<const float*>(lhs_ptr);
        KaiPackedData[i].C = Data[i].C;
        KaiPackedData[i].ldc = Data[i].ldc;

        if (Data[0].BIsPacked) {
            KaiPackedData[i].B = Data[i].B;
        }
    }

    if (!Data[0].BIsPacked) {
        RhsPackedStride = ArmKleidiAI::MlasGemmPackBSize(TransA, TransB, N, K);
        RhsPacked.reset(new std::byte[RhsPackedStride * BatchSize]);

        for (size_t i = 0; i < BatchSize; ++i) {
            std::byte* rhs_ptr = RhsPacked.get() + i * RhsPackedStride;
            ArmKleidiAI::MlasGemmPackB(TransA, TransB, N, K, reinterpret_cast<const float*>(Data[i].B), Data[i].ldb, rhs_ptr);
            KaiPackedData[i].B = reinterpret_cast<const float*>(rhs_ptr);
        }
    }

    // Prepare tile loop bounds
    const size_t MTiles = MlasDivRoundup(M, m_step);
    const size_t NTiles = MlasDivRoundup(N, n_step);
    const size_t TileCount = BatchSize * MTiles * NTiles;

    // Adjust m_step/n_step to spread tile load better
    const size_t ThreadBudget = std::min<size_t>(TileCount, MlasGetMaximumThreadCount(ThreadPool));
    m_step *= MlasDivRoundup(MlasDivRoundup(M, ThreadBudget), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(N, ThreadBudget), n_step);

    // Recalculate tile counts with adjusted step sizes
    const size_t MTileFinal = MlasDivRoundup(M, m_step);
    const size_t NTileFinal = MlasDivRoundup(N, n_step);

    // Parallelize outer tile loop
    MlasTrySimpleParallel(ThreadPool, BatchSize * MTileFinal * NTileFinal, [&](ptrdiff_t tid) {
        const size_t BIdx = tid / (MTileFinal * NTileFinal);
        const size_t MIdx = (tid / NTileFinal) % MTileFinal;
        const size_t NIdx = tid % NTileFinal;

        const size_t MBase = MIdx * m_step;
        const size_t NBase = NIdx * n_step;
        const size_t TileM = std::min(m_step, M - MBase);
        const size_t TileN = std::min(n_step, N - NBase);

        const size_t lhs_offset = kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(MBase, K);
        const size_t rhs_offset = kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(NBase, K);

        const float* ATile = reinterpret_cast<const float*>(
            reinterpret_cast<const std::byte*>(KaiPackedData[BIdx].A) + lhs_offset);

        const void* BTile = reinterpret_cast<const void*>(
            reinterpret_cast<const std::byte*>(KaiPackedData[BIdx].B) + rhs_offset);

        void* CTile = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(KaiPackedData[BIdx].C) +
            MBase * KaiPackedData[BIdx].ldc * sizeof(float) + NBase * sizeof(float));

        kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
            TileM, TileN, K,
            ATile, BTile, CTile,
            KaiPackedData[BIdx].ldc * sizeof(float), sizeof(float),
            -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()
        );
    });
}
