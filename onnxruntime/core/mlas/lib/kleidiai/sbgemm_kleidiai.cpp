//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#if defined(__aarch64__) && defined(__linux__)

#include <vector>
#include <cstring>
#include <cstddef>
#include <array>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"

#include "mlas.h"

#include "mlasi_kleidiai.h"
#include "kai_ukernel_interface.h"

// Thread-local reusable buffers to reduce allocation overhead across tiles.
struct KaiTlsBuffersSbgemm {
    std::vector<float> output_tile;
    std::vector<float> bias_zero;
    std::vector<std::byte> rhs_packed;
    std::vector<std::byte> lhs_packed;
};
static thread_local KaiTlsBuffersSbgemm g_kai_tls_sbgemm;

const KaiBF16SBgemmKernel& sbgemm_gemm = GetKleidiAISBGemmUKernel();

/*++
Routine Description:
    Accumulate src into dst: dst[i,j] += src[i,j], respecting ldc.

Arguments:
    src       - Pointer to the temporary A*B results (row-major, rows x cols).
    rows      - Number of rows in the tile.
    cols      - Number of columns in the tile.
    dst       - Pointer to the destination tile in C (row-major with leading dimension ldc).
    ldc       - Leading dimension of C (in elements).

Notes:
    Implements the accumulation path for SBGEMM when ZeroMode == false.
--*/
static inline void AccumulateTile(const float* src,
                                  size_t rows,
                                  size_t cols,
                                  float* dst,
                                  size_t ldc) {
    if (ldc == cols) {
        // contiguous block in memory: add elementwise across whole block
        size_t elems = rows * cols;
        for (size_t i = 0; i < elems; ++i) {
            dst[i] += src[i];
        }
    } else {
        // general case with row stride and a column offset
        for (size_t i = 0; i < rows; ++i) {
            const float* src_row = src + i * cols;
            float* dst_row = dst + i * ldc;
            for (size_t j = 0; j < cols; ++j) {
                dst_row[j] += src_row[j];
            }
        }
    }
}

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
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
)
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    TransA - Supplies the transpose operation on A matrix.

    TransB - Supplies the transpose operation on B matrix.

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    if (TransA != CblasNoTrans || TransB != CblasNoTrans || N == 0  || K == 0) {
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
    destination buffer should be sized based on MlasSBGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

    TransA - Supplies the transpose operation on A matrix.

    TransB - Supplies the transpose operation on B matrix.

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
    if (TransA != CblasNoTrans || TransB != CblasNoTrans || N == 0 || K == 0) {
        KLEIDIAI_DEBUG_LOG("MlasSBGemmPackB one of N or K is 0, falling back to MLAS.");
        return false;
    }

    const size_t nr = sbgemm_gemm.ukernel.get_nr();
    const size_t kr = sbgemm_gemm.ukernel.get_kr();
    const size_t sr = sbgemm_gemm.ukernel.get_sr();

    // Ensure size and zero the used span.
    g_kai_tls_sbgemm.bias_zero.resize(N, 0.0f);

    kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, g_kai_tls_sbgemm.bias_zero.data(), nullptr, PackedB, 0, nullptr);

    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasSBGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
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

    TransA - Supplies the transpose operation on A matrix.

    TransB - Supplies the transpose operation on B matrix.

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
    if (TransA != CblasNoTrans || TransB != CblasNoTrans || K == 0) {
        return false;
    }

    if (M == 0 || N == 0 || BatchSize == 0) {
        return true;
    }

    size_t m_step = sbgemm_gemm.ukernel.get_m_step();
    size_t n_step = sbgemm_gemm.ukernel.get_n_step();

    if ((M < m_step || N < n_step) && !Data->BIsPacked) {
        // Fallback
        return false;
    }

    const size_t mr = sbgemm_gemm.ukernel.get_mr();
    const size_t kr = sbgemm_gemm.ukernel.get_kr();
    const size_t sr = sbgemm_gemm.ukernel.get_sr();

    size_t LhsPackedStride = 0;
    std::byte* LhsPackedData = nullptr;

    LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(M, K, mr, kr, sr);

    size_t lhs_resize = 0;
    if (mul_overflow_size_t_builtin(LhsPackedStride, BatchSize, &lhs_resize))
    {
        // size_t wraparound detected for LhsPackedStride, fallback to MLAS
        return false;
    }

    g_kai_tls_sbgemm.lhs_packed.resize(lhs_resize);
    LhsPackedData = g_kai_tls_sbgemm.lhs_packed.data();

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
        RhsPackedStride = ArmKleidiAI::MlasSBGemmPackBSize(TransA, TransB, N, K);
        size_t rhs_resize = 0;
        if (mul_overflow_size_t_builtin(RhsPackedStride, BatchSize, &rhs_resize))
        {
            // size_t wraparound detected for RhsPackedStride, fallback to MLAS
            return false;
        }

        g_kai_tls_sbgemm.rhs_packed.resize(rhs_resize);
        RhsPackedData = g_kai_tls_sbgemm.rhs_packed.data();

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
                ArmKleidiAI::MlasSBGemmPackB(TransA, TransB, N, K,
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
        const size_t rhs_packed_offset = sbgemm_gemm.ukernel.get_rhs_packed_offset(NIdx * n_step, K);

        const std::byte* B_base = Data[0].BIsPacked
            ? reinterpret_cast<const std::byte*>(Data[BIdx].B)
            : (RhsPackedData + RhsPackedStride * BIdx);
        auto BTile = reinterpret_cast<const void*>(B_base + rhs_packed_offset);

        // Get lhs tile, A
        const size_t lhs_packed_offset = sbgemm_gemm.ukernel.get_lhs_packed_offset(MIdx * m_step, K);

        const std::byte* A_base = LhsPackedData + LhsPackedStride * BIdx;
        auto ATile = reinterpret_cast<const void*>(A_base + lhs_packed_offset);

        auto TileSizeM = (MIdx + 1) * m_step > M ? (M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > N ? (N - NIdx * n_step) : n_step;

        // Get result tile, C
        auto CTile = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(Data[BIdx].C) +
            MIdx * m_step * Data[BIdx].ldc * sizeof(float) +
            NIdx * n_step * sizeof(float)
        );

        // Final output tile and bias pointers
        float* dst_tile = reinterpret_cast<float*>(CTile);
        const float* bias = Data[BIdx].Bias;
        const size_t ldc = Data[BIdx].ldc;

        // Select output destination and strides once, then run_matmul exactly once.
        const bool direct_to_c = (
            bias == nullptr &&
            Data[BIdx].ZeroMode &&
            TileSizeM != 0 &&
            TileSizeN != 0);

        float* out_tile = nullptr;
        size_t out_row_stride_bytes = 0;

        if (direct_to_c) {
            out_tile = dst_tile;
            out_row_stride_bytes = ldc * sizeof(float);
        } else {
            // Compute into a temporary buffer for raw A*B result (TLS reusable buffer)
            const size_t tile_elems = TileSizeM * TileSizeN;
            g_kai_tls_sbgemm.output_tile.resize(tile_elems);
            out_tile = g_kai_tls_sbgemm.output_tile.data();
            out_row_stride_bytes = TileSizeN * sizeof(float);
        }

        KLEIDIAI_KERNEL_LOG(sbgemm_gemm.name
                            << " M=" << TileSizeM << " N=" << TileSizeN << " K=" << K);
        sbgemm_gemm.ukernel.run_matmul(
            TileSizeM,
            TileSizeN,
            K,
            ATile, BTile, out_tile,
            out_row_stride_bytes, sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );

        if (!direct_to_c) {
            if (Data[BIdx].ZeroMode) {
                ApplyBias2D(out_tile, TileSizeM, TileSizeN, bias, dst_tile, ldc, NIdx * n_step);
            } else {
                AccumulateTile(out_tile, TileSizeM, TileSizeN, dst_tile, ldc);
            }
        }

        if (Data[BIdx].OutputProcessor != nullptr) {
            Data[BIdx].OutputProcessor->Process(
                Data[BIdx].C,
                MIdx * m_step,
                NIdx * n_step,
                TileSizeM,
                TileSizeN,
                Data[BIdx].ldc);
        }
    });
    return true;
}
#endif
