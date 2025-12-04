//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <map>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"

#include "mlasi_kleidiai.h"

// Thread-local reusable buffers to reduce allocation overhead across tiles.
struct KaiTlsBuffersQgemm {
    std::vector<std::byte> lhs_packed;
    std::vector<const std::byte*> lhs_base_table;
};
static thread_local KaiTlsBuffersQgemm g_kai_tls_qgemm;

//Matmul with float output of dynamic quantized A and symmetric quantized B.

size_t
MLASCALL
ArmKleidiAI::MlasDynamicQgemmPackBSize(
    size_t N,
    size_t K
) {
    //Default to sme2_mopa but this may not awalys be the most optimal kernel variant to use
    auto nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();

    //regardless of kernel variant use neon packing variant
    KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon Groups=1"
                        << " N="<< N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr);
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
) {
    // Default to sme2_mopa but this may not awalys be the most optimal kernel variant to use
    auto nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();

    // y - float output
    // scale_factor_lhs - lhs scaling factor
    // scale_factor_rhs - rhs scaling factor
    // lhs_q - lhs quantized (asymmetric, so has zero point)
    // rhs_q - rhs quantized (symmetric so no zero point)
    // lhs_zp - lhs zero point
    // y = (1/(scale_factor_lhs * scale_factor_rhs) * sum( (lhs_q + lhs_zp)*rhs_q )) + bias

    // rhs packing requires lhs_zp because it will perform lhs_zp*rhs_q during rhs packing
    // because lhs quantization is hidden from us, by lhs quant packing, we don't have a value for lhs_zp it is
    // lhs dynamic quantization

    kai_rhs_pack_qsi8cx_params params{
        1,  // lhs_zp - set to 1 so it becomes sum((lhs_q + 1)*rhs_q )),
            // the actual lhs_zp is applied during the matmul
        1.f  // it is not used
    };

    //regardless of kernel variant use neon packing variant
    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1, N, K, nr, kr, sr, B,
                                             // N bias values
                                             Bias,
                                             // N scale values
                                             Scales, PackedB, 0, &params);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
) {

    const size_t mr = UseSME2 ? kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa()
                              : kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa();
    const size_t kr = UseSME2 ? kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa()
                              : kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa();
    const size_t sr = UseSME2 ? kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa()
                              : kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa();

    size_t m_step = UseSME2 ? kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa()
                            : kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa();
    size_t n_step = UseSME2 ? kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa()
                            : kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa();

    if (BatchSize == 0 || Shape.M == 0 || Shape.N == 0 ) {
        return;
    }

    //We are required to enforce errors when we reach this stage as we will not be able
    //to reverse the packing decision that was made for RHS.

    ORT_ENFORCE(DataParams != nullptr, "Dynamic QGEMM requires valid DataParams.");
    ORT_ENFORCE(Shape.K > 0, "Dynamic QGEMM requires Shape.K to be non-zero.");

    for (size_t batch_idx = 0; batch_idx < BatchSize; ++batch_idx) {
        const auto& params = DataParams[batch_idx];
        ORT_ENFORCE(params.A != nullptr, "Dynamic QGEMM requires non-null A pointer for batch ", batch_idx);
        ORT_ENFORCE(params.C != nullptr, "Dynamic QGEMM requires non-null C pointer for batch ", batch_idx);
        ORT_ENFORCE(params.PackedB != nullptr, "Dynamic QGEMM requires non-null PackedB pointer for batch ", batch_idx);
        const size_t lda = params.lda != 0 ? params.lda : Shape.K;
        const size_t ldc = params.ldc != 0 ? params.ldc : Shape.N;
        ORT_ENFORCE(lda >= Shape.K, "lda (", lda, ") must be >= Shape.K (", Shape.K, ") for batch ", batch_idx);
        ORT_ENFORCE(ldc >= Shape.N, "ldc (", ldc, ") must be >= Shape.N (", Shape.N, ") for batch ", batch_idx);
    }

    //Dynamic Quantize A - lhs
    const size_t LhsPackedStride = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr);
    std::byte* LhsPackedData = nullptr;

    if (g_kai_tls_qgemm.lhs_packed.capacity() < LhsPackedStride * BatchSize) {

        g_kai_tls_qgemm.lhs_packed.reserve(LhsPackedStride * BatchSize);
    }
    g_kai_tls_qgemm.lhs_packed.resize(LhsPackedStride * BatchSize);
    LhsPackedData = g_kai_tls_qgemm.lhs_packed.data();

    //Per-batch table of lhs
    if (g_kai_tls_qgemm.lhs_base_table.capacity() < BatchSize) {

        g_kai_tls_qgemm.lhs_base_table.reserve(BatchSize);
    }
    g_kai_tls_qgemm.lhs_base_table.resize(BatchSize);
    // Capture the shared batch table pointer so worker threads use the same backing storage.
    const std::byte** tls_lhs_base = g_kai_tls_qgemm.lhs_base_table.data();
    // B batches require no packing
    // We have already decided the matmul variant we are using, before having values for M,N,K
    MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {

        std::byte* lhs = nullptr;
        if (DataParams[batch_idx].Workspace && DataParams[batch_idx].WorkspaceSize >= LhsPackedStride) {
            lhs = static_cast<std::byte*>(DataParams[batch_idx].Workspace);
        } else {
            lhs = &(LhsPackedData[LhsPackedStride * batch_idx]);
        }

        kai_run_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr, 0, DataParams[batch_idx].A, DataParams[batch_idx].lda*sizeof(float), lhs);
        tls_lhs_base[batch_idx] = lhs;
    });

    // tile iteration dimensions
    std::array<size_t, 3> dim;
    dim[0] = BatchSize;                  // B
    dim[1] = MlasDivRoundup(Shape.M, m_step);  // M
    dim[2] = MlasDivRoundup(Shape.N, n_step);  // N

    // Minimize the kernel call count for the number of available threads
    auto RequiredTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0] * dim[1] * dim[2]);

    // scale required tiles over available tile processors
    dim[1] = MlasDivRoundup(RequiredTiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(RequiredTiles * dim[2], dim[1] * dim[2]);

    // compute new step sizes
    m_step *= MlasDivRoundup(MlasDivRoundup(Shape.M, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(Shape.N, dim[2]), n_step);

    // update tile iterations
    dim[1] = MlasDivRoundup(Shape.M, m_step);
    dim[2] = MlasDivRoundup(Shape.N, n_step);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [=](ptrdiff_t tid) {

        // compute B,M,N index from iteration index
        ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
        ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
        ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

        // Get rhs tile, B
        const size_t rhs_packed_offset =
           UseSME2 ? kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(NIdx * n_step, Shape.K)
                   : kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa(NIdx * n_step, Shape.K);

        const std::byte* B_base = reinterpret_cast<const std::byte*>(DataParams[BIdx].PackedB);
        auto BTile = reinterpret_cast<const void*>(B_base + rhs_packed_offset);

        // Get lhs tile, A
        const size_t lhs_packed_offset =
            UseSME2 ? kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(MIdx * m_step, Shape.K)
                    : kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa(MIdx * m_step, Shape.K);

        const std::byte* A_base = tls_lhs_base[BIdx]; // LhsPackedData + LhsPackedStride * BIdx; OR DataParams[batch_idx].Workspace;
        auto ATile = reinterpret_cast<const std::byte*>(A_base + lhs_packed_offset);

        auto TileSizeM = (MIdx + 1) * m_step > Shape.M ? (Shape.M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > Shape.N ? (Shape.N - NIdx * n_step) : n_step;

        float* dst_tile = reinterpret_cast<float*>(
        reinterpret_cast<std::byte*>(DataParams[BIdx].C) +
        MIdx * m_step * DataParams[BIdx].ldc * sizeof(float) +
        NIdx * n_step * sizeof(float)
        );

        if (UseSME2) {
            kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
                TileSizeM, TileSizeN, Shape.K, ATile, BTile,
                dst_tile,
                DataParams[BIdx].ldc * sizeof(float),
                sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
                );
            }
        else {
            kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa(
                TileSizeM, TileSizeN, Shape.K, ATile, BTile,
                dst_tile,
                DataParams[BIdx].ldc * sizeof(float),
                sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
                );
        }
    });
}
