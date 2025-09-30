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
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"

#include "mlasi_kleidiai.h"

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
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
) {
    for (auto b = BatchN; b > 0; --b,++DataParams) {
        auto mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
        auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
        auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();


        //TODO enable multi-threading for lhs packing and matmul
        MLAS_UNREFERENCED_PARAMETER(ThreadPool);

        //Dynamic Quantize A - lhs
        auto lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr);
        std::byte* lhs = nullptr;
        std::unique_ptr<std::byte[]> fallback;

        if (DataParams->Workspace && DataParams->WorkspaceSize >= lhs_size) {
            lhs = static_cast<std::byte*>(DataParams->Workspace);
        } else {
            fallback = std::make_unique<std::byte[]>(lhs_size);
            lhs = fallback.get();
        }

        kai_run_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr, 0, DataParams->A,
                                           Shape.K*sizeof(float), lhs);

        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
            Shape.M, Shape.N, Shape.K, lhs, DataParams->PackedB,
            DataParams->C,
            Shape.N * sizeof(float),
            sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );
    }
}
