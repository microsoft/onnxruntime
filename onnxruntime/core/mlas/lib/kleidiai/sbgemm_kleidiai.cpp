//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#if defined(__aarch64__) && defined(__linux__)

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"

#include "mlas.h"

#include "mlasi_kleidiai.h"
#include "kai_ukernel_interface.h"

// Thread-local reusable buffers to reduce allocation overhead across tiles.
struct KaiTlsBuffers {
    std::vector<float> output_tile;
    std::vector<float> bias_zero;
    std::vector<std::byte> rhs_packed;
    std::vector<std::byte> lhs_packed;
    std::vector<float> gemv_lhs_row_tmp;
};
static thread_local KaiTlsBuffers g_kai_tls;

kai_matmul_clamp_f32_bf16p_bf16p_ukernel sbgemm_gemm = GetKleidiAISBGemmUKernel();

/*++
Routine Description:
    Apply bias to a 2-D tile (rows x cols).

Arguments:
    src       - Pointer to the temporary A*B results (row-major, rows x cols).
    rows      - Number of rows in the tile.
    cols      - Number of columns in the tile.
    bias      - Pointer to the bias vector or nullptr if no bias.
    dst       - Pointer to the destination tile in C (row-major with leading dimension ldc).
    ldc       - Leading dimension of C (in elements).
    start_col - Starting column index of the tile (NIdx * n_step).

Notes:
    Uses a row by row memcpy path when no bias.
--*/
static inline void ApplyBias2D(const float* src,
                               size_t rows, 
                               size_t cols,
                               const float* bias, 
                               float* dst,
                               size_t ldc, 
                               size_t start_col) {
    for (size_t i = 0; i < rows; ++i) {
        const float* src_row = src + i * cols;
        float* dst_row = dst + i * ldc;

        if (bias != nullptr) {
            for (size_t j = 0; j < cols; ++j) {
                dst_row[j] = src_row[j] + bias[start_col + j];
            }
        } else {
            // No bias but can't memcpy whole so needs to be done row by row.
            memcpy(dst_row, src_row, cols * sizeof(float));
        }
    }
}

size_t
MLASCALL
ArmKleidiAI::MlasSBGemmPackBSize(
    size_t N,
    size_t K
)
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    if (N == 0  || K == 0) {
        KLEIDIAI_DEBUG_LOG("MlasSBGemmPackBSize returning 0 size. N=" << N << " K=" << K);
        return 0;
    }
    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    size_t bytes = 0;
    bytes = kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(N, K);

    return bytes;
}

bool
MLASCALL
ArmKleidiAI::MlasSBGemmPackB(
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
)
/*++

Routine Description:

    This routine packs the contents of matrix B to the destination buffer. The
    destination buffer should be sized based on MlasSBGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

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
        KLEIDIAI_DEBUG_LOG("MlasSBGemmPackB one of N or K is 0, falling back to MLAS.");
        return false;
    }

    const size_t nr = sbgemm_gemm.get_nr();
    const size_t kr = sbgemm_gemm.get_kr();
    const size_t sr = sbgemm_gemm.get_sr();

    // Ensure size and zero the used span.
    g_kai_tls.bias_zero.resize(N, 0.0f);

    kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, g_kai_tls.bias_zero.data(), nullptr, PackedB, 0, nullptr);

    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasSBGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SBGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine performs a bfloat16 batched matrix multiplication (SBGEMM) operation using KleidiAI kernels.
    If packing is needed, it prepares the required buffers and invokes the
    appropriate left-hand side (LHS) and right-hand side (RHS) pack functions.

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and rows of matrix B.

    Data - Supplies a pointer to the MLAS_SBGEMM_DATA_PARAMS array containing per-batch input/output pointers and parameters.

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

    size_t m_step = sbgemm_gemm.get_m_step();
    size_t n_step = sbgemm_gemm.get_n_step();

    if ((M < m_step || N < n_step) && !Data->BIsPacked) {
        // Fallback
        return false;
    }

    const size_t mr = sbgemm_gemm.get_mr();
    const size_t kr = sbgemm_gemm.get_kr();
    const size_t sr = sbgemm_gemm.get_sr();

    size_t LhsPackedStride = 0;
    std::byte* LhsPackedData = nullptr;

    LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(M, K, mr, kr, sr);

    size_t lhs_resize = 0;
    if(mul_overflow_size_t_builtin(LhsPackedStride, BatchSize, &lhs_resize))
    {
        // size_t wraparound detected for LhsPackedStride, fallback to MLAS
        return false;
    }

    g_kai_tls.lhs_packed.resize(lhs_resize);
    LhsPackedData = g_kai_tls.lhs_packed.data();

    // RHS packed buffer: use TLS reusable vector to minimize allocations
    size_t RhsPackedStride = 0;
    std::byte* RhsPackedData = nullptr;

    // It is assumed all B batches require packing or not
    if (Data[0].BIsPacked) {
        // We have already decided the matmul variant we are using, before having values for M,N,K
        MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {
            std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);
            KLEIDIAI_KERNEL_LOG("kai_run_lhs_pack_bf16p2vlx2_f32_sme" << " M=" << M << " K=" << K << " mr=" << mr << " kr=" << kr << " sr=" << sr);
            kai_run_lhs_pack_bf16p2vlx2_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);
        });
    } else {
        // Multithread pack lhs and rhs
        RhsPackedStride = ArmKleidiAI::MlasSBGemmPackBSize(N, K);
        size_t rhs_resize = 0;
        if (mul_overflow_size_t_builtin(RhsPackedStride, BatchSize, &rhs_resize))
        {
            // size_t wraparound detected for RhsPackedStride, fallback to MLAS
            return false;
        }

        g_kai_tls.rhs_packed.resize(rhs_resize);
        RhsPackedData = g_kai_tls.rhs_packed.data();

        MlasTrySimpleParallel(ThreadPool, BatchSize * 2, [&](ptrdiff_t batch_idx) {
            if (batch_idx & 0x1) {
                batch_idx >>= 1;
                std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);
                KLEIDIAI_KERNEL_LOG("kai_run_lhs_pack_bf16p2vlx2_f32_sme"
                                    << " M=" << M << " K=" << K << " mr=" << mr << " kr=" << kr << " sr=" << sr);
                kai_run_lhs_pack_bf16p2vlx2_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);
            } else {
                batch_idx >>= 1;
                std::byte* RhsPackedPtr = &(RhsPackedData[RhsPackedStride * batch_idx]);
                ArmKleidiAI::MlasSBGemmPackB(N, K,
                                             reinterpret_cast<const float*>(Data[batch_idx].B),
                                             Data[batch_idx].ldb, RhsPackedPtr);
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

    // Pre-check maximum tile size to avoid per-iteration overflow inside the parallel loop.
    // Any TileSizeM/TileSizeN used below will be <= m_step/n_step respectively.
    size_t max_tile_elems = 0;
    if (mul_overflow_size_t_builtin(m_step, n_step, &max_tile_elems)) {
        // size_t wraparound detected for tile size, fallback to MLAS
        return false;
    }

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [=](ptrdiff_t tid) {
        // compute B,M,N index from iteration index
        ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
        ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
        ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

        // Get rhs tile, B
        const size_t rhs_packed_offset = sbgemm_gemm.get_rhs_packed_offset(NIdx * n_step, K);

        const std::byte* B_base = Data[0].BIsPacked
            ? reinterpret_cast<const std::byte*>(Data[BIdx].B)
            : (RhsPackedData + RhsPackedStride * BIdx);
        auto BTile = reinterpret_cast<const void*>(B_base + rhs_packed_offset);

        // Get lhs tile, A
        const size_t lhs_packed_offset = sbgemm_gemm.get_lhs_packed_offset(MIdx * m_step, K);

        const std::byte* A_base = LhsPackedData + LhsPackedStride * BIdx;
        auto ATile = reinterpret_cast<const float*>(A_base + lhs_packed_offset);

        auto TileSizeM = (MIdx + 1) * m_step > M ? (M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > N ? (N - NIdx * n_step) : n_step;

        // Get result tile, C
        auto CTile = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(Data[BIdx].C) +
            MIdx * m_step * Data[BIdx].ldc * sizeof(float) +
            NIdx * n_step * sizeof(float)
        );
        // Allocate temporary buffer for raw A*B result (TLS reusable buffer)
        size_t tile_elems = TileSizeM * TileSizeN;

        // resize the tile to the required size
        g_kai_tls.output_tile.resize(tile_elems);

        float* temp_tile = g_kai_tls.output_tile.data();
        std::fill_n(temp_tile, tile_elems, 0.0f);

        sbgemm_gemm.run_matmul(
                TileSizeM,
                TileSizeN,
                K,
                ATile, BTile, temp_tile,
                TileSizeN * sizeof(float), sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );

        // Final output tile pointer
        float* dst_tile = reinterpret_cast<float*>(CTile);
        const float* bias = Data[BIdx].Bias;

        // quick copy of data in cases where we are not applying bias
        // with bounds checking on tile sizing to ensure the data fits in the memory block
        bool can_memcpy = (
            bias == nullptr &&
            Data[BIdx].ldc == TileSizeN &&
            MIdx * m_step + TileSizeM <= M &&
            NIdx * n_step + TileSizeN <= N &&
            TileSizeM != 0 &&
            TileSizeN != 0);

        if (can_memcpy) {
            std::memcpy(dst_tile, temp_tile, TileSizeM * TileSizeN * sizeof(float));
            return;
        }

        ApplyBias2D(temp_tile, TileSizeM, TileSizeN, bias, dst_tile, Data[BIdx].ldc, NIdx * n_step);
        return;
    });
    return true;
}
#endif
