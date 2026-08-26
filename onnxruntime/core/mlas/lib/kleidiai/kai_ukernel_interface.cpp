//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//


#include "../mlasi.h"

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

// NEON / NEON+dotprod / i8mm kernels
//   GEMM/QGEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot.h"
//   GEMV
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"

// SME kernels
//   GEMM/QGEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
//   GEMV
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"
//   IMATMUL
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa.h"

// SME2 kernels
//   GEMM/QGEMM/SBGEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"

//   IMATMUL
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"

// FP32 legacy packers. The legacy SME/QMX kernels use these directly; their
// next-generation counterparts are provided by the common packer API.
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"

// FP16 HGEMM kernels
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"

#if defined(ENABLE_QMX_KERNELS)
// QMX kernels (optional)
//   GEMM/QGEMM
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa.h"
//   IMATMUL
#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_qmx_mopa.h"
#endif // ENABLE_QMX_KERNELS

namespace {

const kai_matmul_uker_config kMatmulConfig{};
const kai_matmul_pack_lhs_uker_config kLhsPackConfig{};
const kai_matmul_pack_rhs_uker_config kRhsPackConfig{};

}  // namespace

KaiF32SgemmUkernel::KaiF32SgemmUkernel(kai_matmul_clamp_f32_f32p_f32p_ukernel ukernel)
    : KaiMatmulUkernel(ukernel) {}

KaiF32SgemmUkernel::KaiF32SgemmUkernel(kai_matmul_uker_api ukernel,
                                       kai_matmul_pack_lhs_uker_api lhs_packer,
                                       kai_matmul_pack_rhs_uker_api rhs_kxn_packer,
                                       kai_matmul_pack_rhs_uker_api rhs_nxk_packer)
    : KaiMatmulUkernel(ukernel),
      lhs_packer_(lhs_packer),
      rhs_kxn_packer_(rhs_kxn_packer),
      rhs_nxk_packer_(rhs_nxk_packer) {}

size_t KaiF32SgemmUkernel::get_m_step() const {
    if (!UsesUkernelApi()) {
        return LegacyUkernelInterface().get_m_step();
    }

    return UkernelApi().get_step(&kMatmulConfig).m;
}

size_t KaiF32SgemmUkernel::get_n_step() const {
    if (!UsesUkernelApi()) {
        return LegacyUkernelInterface().get_n_step();
    }

    return UkernelApi().get_step(&kMatmulConfig).n;
}

size_t KaiF32SgemmUkernel::get_lhs_packed_offset(size_t m_idx, size_t k) const {
    if (!UsesUkernelApi()) {
        return LegacyUkernelInterface().get_lhs_packed_offset(m_idx, k);
    }

    const kai_matmul_uker_lhs_dim_args shape{0, k};
    const auto stride = UkernelApi().get_lhs_stride(&kMatmulConfig, &shape);
    const kai_matmul_uker_lhs_dim_args index{m_idx, 0};
    return UkernelApi().get_lhs_offset(&kMatmulConfig, &index, &stride);
}

size_t KaiF32SgemmUkernel::get_rhs_packed_offset(size_t n_idx, size_t k) const {
    if (!UsesUkernelApi()) {
        return LegacyUkernelInterface().get_rhs_packed_offset(n_idx, k);
    }

    const kai_matmul_uker_rhs_dim_args shape{0, k};
    const auto stride = UkernelApi().get_rhs_stride(&kMatmulConfig, &shape);
    const kai_matmul_uker_rhs_dim_args index{n_idx, 0};
    return UkernelApi().get_rhs_offset(&kMatmulConfig, &index, &stride);
}

void KaiF32SgemmUkernel::run_matmul(size_t m, size_t n, size_t k,
                                    const void* lhs_packed, const void* rhs_packed, void* dst,
                                    size_t dst_stride_row, size_t dst_stride_col,
                                    float clamp_min, float clamp_max) const {
    if (!UsesUkernelApi()) {
        LegacyUkernelInterface().run_matmul(m, n, k, lhs_packed, rhs_packed, dst,
                                            dst_stride_row, dst_stride_col, clamp_min, clamp_max);
        return;
    }

    const kai_matmul_uker_lhs_dim_args lhs_shape{m, k};
    const kai_matmul_uker_rhs_dim_args rhs_shape{n, k};
    kai_matmul_uker_args args{};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape = {m, n, k};
    args.operand.lhs.ptr = lhs_packed;
    args.operand.lhs.stride = UkernelApi().get_lhs_stride(&kMatmulConfig, &lhs_shape);
    args.operand.rhs.ptr = rhs_packed;
    args.operand.rhs.stride = UkernelApi().get_rhs_stride(&kMatmulConfig, &rhs_shape);
    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = dst_stride_row;
    args.activation.clamp.min_ptr = &clamp_min;
    args.activation.clamp.max_ptr = &clamp_max;
    UkernelApi().run(&kMatmulConfig, &args);
    MLAS_UNREFERENCED_PARAMETER(dst_stride_col);
}

size_t KaiF32SgemmUkernel::GetLhsPackedSize(size_t m, size_t k) const {
    if (!UsesUkernelApi()) {
        const auto& ukernel = LegacyUkernelInterface();
        return kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(
            m, k, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
    }

    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape{m, k};
    const auto stride = lhs_packer_.get_lhs_packed_stride(&kLhsPackConfig, &shape);
    return lhs_packer_.get_lhs_packed_size(&kLhsPackConfig, &shape, &stride);
}

const char* KaiF32SgemmUkernel::GetLhsPackerName() const {
    return UsesUkernelApi()
               ? "kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme"
               : "kai_run_lhs_pack_f32p2vlx1_f32_sme";
}

void KaiF32SgemmUkernel::PackLhs(size_t m, size_t k, const float* lhs,
                                 size_t lhs_stride, void* lhs_packed) const {
    if (!UsesUkernelApi()) {
        const auto& ukernel = LegacyUkernelInterface();
        kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr(),
                                            0, lhs, lhs_stride * sizeof(float), lhs_packed);
        return;
    }

    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args packed_shape{m, k};
    kai_matmul_pack_lhs_uker_args args{};
    args.shape = {m, k};
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = lhs_stride * sizeof(float);
    args.operand.lhs_packed.ptr = lhs_packed;
    args.operand.lhs_packed.stride =
        lhs_packer_.get_lhs_packed_stride(&kLhsPackConfig, &packed_shape);
    lhs_packer_.run(&kLhsPackConfig, &args);
}

const kai_matmul_pack_rhs_uker_api& KaiF32SgemmUkernel::RhsPacker(KaiF32RhsLayout layout) const {
    return layout == KaiF32RhsLayout::KxN ? rhs_kxn_packer_ : rhs_nxk_packer_;
}

size_t KaiF32SgemmUkernel::GetRhsPackedSize(KaiF32RhsLayout layout, size_t n, size_t k) const {
    if (!UsesUkernelApi()) {
        return layout == KaiF32RhsLayout::KxN
                   ? kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k)
                   : kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k);
    }

    const auto& packer = RhsPacker(layout);
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape{n, k};
    const auto stride = packer.get_rhs_packed_stride(&kRhsPackConfig, &shape);
    return packer.get_rhs_packed_size(&kRhsPackConfig, &shape, &stride);
}

const char* KaiF32SgemmUkernel::GetRhsPackerName(KaiF32RhsLayout layout) const {
    if (UsesUkernelApi()) {
        return layout == KaiF32RhsLayout::KxN
                   ? "kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme"
                   : "kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme";
    }

    return layout == KaiF32RhsLayout::KxN
               ? "kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme"
               : "kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme";
}

void KaiF32SgemmUkernel::PackRhs(KaiF32RhsLayout layout, size_t n, size_t k,
                                 const float* rhs, size_t rhs_stride,
                                 const float* bias, void* rhs_packed) const {
    if (!UsesUkernelApi()) {
        const auto& ukernel = LegacyUkernelInterface();
        if (layout == KaiF32RhsLayout::KxN) {
            kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                1, n, k, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(), rhs_stride * sizeof(float),
                rhs, bias, nullptr, rhs_packed, 0, nullptr);
        } else {
            kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(
                1, n, k, ukernel.get_nr(), ukernel.get_kr(), ukernel.get_sr(), rhs_stride * sizeof(float),
                rhs, bias, nullptr, rhs_packed, 0, nullptr);
        }
        return;
    }

    const auto& packer = RhsPacker(layout);
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape{n, k};
    kai_matmul_pack_rhs_uker_args args{};
    args.shape = {n, k};
    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = layout == KaiF32RhsLayout::KxN
                                  ? kai_matmul_pack_rhs_uker_rhs_stride_args{sizeof(float),
                                                                            rhs_stride * sizeof(float)}
                                  : kai_matmul_pack_rhs_uker_rhs_stride_args{rhs_stride * sizeof(float),
                                                                            sizeof(float)};
    args.operand.rhs_packed.ptr = rhs_packed;
    args.operand.rhs_packed.stride =
        packer.get_rhs_packed_stride(&kRhsPackConfig, &packed_shape);
    args.operand.bias_n.ptr = bias;
    packer.run(&kRhsPackConfig, &args);
}

KaiF32SgemvUkernel::KaiF32SgemvUkernel(kai_matmul_clamp_f32_f32_f32p_ukernel ukernel)
    : KaiMatmulUkernel(ukernel) {}

KaiF32SgemvUkernel::KaiF32SgemvUkernel(kai_matmul_uker_api ukernel)
    : KaiMatmulUkernel(ukernel) {}

void KaiF32SgemvUkernel::run_matmul(size_t m, size_t n, size_t k,
                                    const void* lhs, size_t lhs_stride,
                                    const void* rhs_packed, void* dst,
                                    size_t dst_stride_row, size_t dst_stride_col,
                                    float clamp_min, float clamp_max) const {
    if (!UsesUkernelApi()) {
        LegacyUkernelInterface().run_matmul(m, n, k, lhs, lhs_stride, rhs_packed, dst,
                                            dst_stride_row, dst_stride_col, clamp_min, clamp_max);
        return;
    }

    const kai_matmul_uker_rhs_dim_args rhs_shape{n, k};
    kai_matmul_uker_args args{};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape = {m, n, k};
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = lhs_stride;
    args.operand.rhs.ptr = rhs_packed;
    args.operand.rhs.stride = UkernelApi().get_rhs_stride(&kMatmulConfig, &rhs_shape);
    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = dst_stride_row;
    args.activation.clamp.min_ptr = &clamp_min;
    args.activation.clamp.max_ptr = &clamp_max;
    UkernelApi().run(&kMatmulConfig, &args);
    MLAS_UNREFERENCED_PARAMETER(dst_stride_col);
}

// -------------------------------------------------------------------------------------------------
// KleidiAI ukernel wrapper macros
//
// These macros exist solely to reduce boilerplate when constructing the various `Kai*Kernel` info
// structs in this file. The names are field-name sequence based as per the typedef interface.h files.
//
// Pass the ukernel "stem" (the suffix shared by all exported functions), e.g.
//   matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa
//
// Each macro derives the full symbol names like:
//   kai_get_m_step_<stem>, ... , kai_run_<stem>
//
// IMPORTANT:
// - Only use a macro if the target ukernel exports *exactly* the expected helper/core symbols.
// - Some ukernel families use different interface shapes; those must use the matching macro (or be
//   instantiated manually).
// -------------------------------------------------------------------------------------------------

// 11-slot `run_matmul` interface shape.
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h
//     struct kai_matmul_clamp_f32_f32p_f32p_ukernel
// - kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h
//     struct kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel
// - kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h
//     struct kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_n_step, get_mr, get_nr, get_kr, get_sr,
//   get_lhs_packed_offset, get_rhs_packed_offset, get_dst_offset, get_dst_size, run_matmul.
#define KAI_WRAP_UKERNEL_RUN_MATMUL_11(STEM)                                                             \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        {kai_get_m_step_##STEM,                                                                          \
         kai_get_n_step_##STEM,                                                                          \
         kai_get_mr_##STEM,                                                                              \
         kai_get_nr_##STEM,                                                                              \
         kai_get_kr_##STEM,                                                                              \
         kai_get_sr_##STEM,                                                                              \
         kai_get_lhs_packed_offset_##STEM,                                                               \
         kai_get_rhs_packed_offset_##STEM,                                                               \
         kai_get_dst_offset_##STEM,                                                                      \
         kai_get_dst_size_##STEM,                                                                        \
         kai_run_##STEM}                                                                                 \
    }

#define KAI_WRAP_F32_SGEMM_LEGACY(STEM)                                                                  \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        KaiF32SgemmUkernel {                                                                             \
            kai_matmul_clamp_f32_f32p_f32p_ukernel {                                                    \
                kai_get_m_step_##STEM,                                                                   \
                kai_get_n_step_##STEM,                                                                   \
                kai_get_mr_##STEM,                                                                       \
                kai_get_nr_##STEM,                                                                       \
                kai_get_kr_##STEM,                                                                       \
                kai_get_sr_##STEM,                                                                       \
                kai_get_lhs_packed_offset_##STEM,                                                        \
                kai_get_rhs_packed_offset_##STEM,                                                        \
                kai_get_dst_offset_##STEM,                                                               \
                kai_get_dst_size_##STEM,                                                                 \
                kai_run_##STEM                                                                           \
            }                                                                                            \
        }                                                                                                \
    }

#define KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(STEM, RHS_LAYOUT)                                              \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        {kai_get_m_step_##STEM,                                                                          \
         kai_get_n_step_##STEM,                                                                          \
         kai_get_mr_##STEM,                                                                              \
         kai_get_nr_##STEM,                                                                              \
         kai_get_kr_##STEM,                                                                              \
         kai_get_sr_##STEM,                                                                              \
         kai_get_lhs_packed_offset_##STEM,                                                               \
         kai_get_rhs_packed_offset_##STEM,                                                               \
         kai_get_dst_offset_##STEM,                                                                      \
         kai_get_dst_size_##STEM,                                                                        \
         kai_run_##STEM},                                                                                \
        RHS_LAYOUT                                                                                       \
    }

// 7-slot packed `run_imatmul` interface shape.
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p_f32p_interface.h
//     struct kai_imatmul_clamp_f32_f32p_f32p_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_n_step, get_lhs_packed_offset, get_rhs_packed_offset, get_dst_offset, get_dst_size, run_imatmul.
#define KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(STEM)                                                      \
    {                                                                                                    \
        "kai_run_" #STEM,                                                                                \
        {kai_get_m_step_##STEM,                                                                          \
         kai_get_n_step_##STEM,                                                                          \
         kai_get_lhs_packed_offset_##STEM,                                                               \
         kai_get_rhs_packed_offset_##STEM,                                                               \
         kai_get_dst_offset_##STEM,                                                                      \
         kai_get_dst_size_##STEM,                                                                        \
         kai_run_##STEM}                                                                                 \
    }

// 10-slot `run_matmul` interface shape with un-packed LHS offset helper.
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h
//     struct kai_matmul_clamp_f32_f32_f32p_ukernel
// - kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp_interface.h
//     struct kai_matmul_clamp_qai8_qai8p_qsi8cxp_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_n_step, get_nr, get_kr, get_sr, get_lhs_offset,
//   get_rhs_packed_offset, get_dst_offset, get_dst_size, run_matmul.
//
// Note: This corresponds to the "GEMV-style" layout currently instantiated manually below.
#define KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_OFFSET(STEM)                                                   \
    {                                                                                                     \
        "kai_run_" #STEM,                                                                                 \
        {kai_get_m_step_##STEM,                                                                           \
         kai_get_n_step_##STEM,                                                                           \
         kai_get_nr_##STEM,                                                                               \
         kai_get_kr_##STEM,                                                                               \
         kai_get_sr_##STEM,                                                                               \
         kai_get_lhs_offset_##STEM,                                                                       \
         kai_get_rhs_packed_offset_##STEM,                                                                \
         kai_get_dst_offset_##STEM,                                                                       \
         kai_get_dst_size_##STEM,                                                                         \
         kai_run_##STEM}                                                                                  \
    }

// 10-slot `run_matmul` interface shape with packed LHS offset helper (no MR field).
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h
//     struct kai_matmul_clamp_f16_f16_f16p_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_n_step, get_nr, get_kr, get_sr, get_lhs_packed_offset,
//   get_rhs_packed_offset, get_dst_offset, get_dst_size, run_matmul.
#define KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_PACKED_OFFSET(STEM)                                             \
    {                                                                                                     \
        "kai_run_" #STEM,                                                                                 \
        {kai_get_m_step_##STEM,                                                                           \
         kai_get_n_step_##STEM,                                                                           \
         kai_get_nr_##STEM,                                                                               \
         kai_get_kr_##STEM,                                                                               \
         kai_get_sr_##STEM,                                                                               \
         kai_get_lhs_packed_offset_##STEM,                                                                \
         kai_get_rhs_packed_offset_##STEM,                                                                \
         kai_get_dst_offset_##STEM,                                                                       \
         kai_get_dst_size_##STEM,                                                                         \
         kai_run_##STEM}                                                                                  \
    }

// 6-slot `run_imatmul` interface shape without LHS packed-offset helper.
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/matmul/imatmul_clamp_f32_f32_f32p/kai_imatmul_clamp_f32_f32_f32p_interface.h
//     struct kai_imatmul_clamp_f32_f32_f32p_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_n_step, get_rhs_packed_offset, get_dst_offset, get_dst_size, run_imatmul.
#define KAI_WRAP_UKERNEL_RUN_IMATMUL_6_NO_LHS_PACKED_OFFSET(STEM)                                          \
    {                                                                                                     \
        "kai_run_" #STEM,                                                                                 \
        {kai_get_m_step_##STEM,                                                                           \
         kai_get_n_step_##STEM,                                                                           \
         kai_get_rhs_packed_offset_##STEM,                                                                \
         kai_get_dst_offset_##STEM,                                                                       \
         kai_get_dst_size_##STEM,                                                                         \
         kai_run_##STEM}                                                                                  \
    }

// 4-slot planar `run_dwconv` interface shape.
//
// Applies to KleidiAI ukernel interface headers/structs such as:
// - kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p_interface.h
//     struct kai_dwconv_clamp_f32_f32_f32p_planar_ukernel
//
// Field sequence (must match exactly):
//   get_m_step, get_dst_offset, get_dst_size, run_dwconv.
#define KAI_WRAP_UKERNEL_RUN_DWCONV_PLANAR_4(STEM)                                                         \
    {                                                                                                     \
        "kai_run_" #STEM,                                                                                 \
        {kai_get_m_step_##STEM,                                                                           \
         kai_get_dst_offset_##STEM,                                                                       \
         kai_get_dst_size_##STEM,                                                                         \
         kai_run_##STEM}                                                                                  \
    }



const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
                                      KaiQ4RhsPackLayout::SymmetricNxK);

const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
                                      KaiQ4RhsPackLayout::SymmetricNxK);

const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
                                      KaiQ4RhsPackLayout::SymmetricNxK);

const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
                                      KaiQ4RhsPackLayout::SymmetricNxK);

const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
                                      KaiQ4RhsPackLayout::SymmetricNxKInterleavedNrx4);

const KaiQnbitGemmKernel kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot,
                                      KaiQ4RhsPackLayout::SymmetricNxKInterleavedNrx4);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod,
                                      KaiQ4RhsPackLayout::AsymmetricNxK);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod,
                                      KaiQ4RhsPackLayout::AsymmetricNxK);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod,
                                      KaiQ4RhsPackLayout::AsymmetricNxK);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm,
                                      KaiQ4RhsPackLayout::AsymmetricNxK);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa,
                                      KaiQ4RhsPackLayout::AsymmetricNxKInterleavedNrx4);

const KaiQnbitAsymGemmKernel kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot =
    KAI_WRAP_Q4_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot,
                                      KaiQ4RhsPackLayout::AsymmetricNxKInterleavedNrx4);

const KaiF32SgemmKernel sgemm_gemm_sme =
    KAI_WRAP_F32_SGEMM_LEGACY(matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa);

// IMATMUL kernels used by KleidiAI convolution. These are packed-imatmul (7-slot) interfaces.
const KaiF32IMatmulKernel imatmul_conv_sme =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa);

const KaiF32IMatmulKernel imatmul_conv_sme2 =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa);

const KaiF16IMatmulKernel imatmul_f16_conv_sme =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa);

const KaiF16IMatmulKernel imatmul_f16_conv_sme2 =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa);

const KaiBF16SBgemmKernel sbgemm_gemm_sme2 =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa);

#if defined(ENABLE_QMX_KERNELS)
const KaiF32IMatmulKernel imatmul_conv_qmx =
    KAI_WRAP_UKERNEL_RUN_IMATMUL_PACKED_7(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_qmx_mopa);
#endif // ENABLE_QMX_KERNELS

const KaiF32SgemmKernel sgemm_gemm_sme2 =
    {
        "kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa",
        KaiF32SgemmUkernel{
            kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(),
            kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(),
            kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(),
            kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(),
        },
    };

const KaiDynamicQGemmKernel qgemm_gemm_sme =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa);

const KaiDynamicQGemmKernel qgemm_gemm_sme2 =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa);

const KaiF16HgemmKernel hgemm_sme =
    KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_OFFSET(matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla);

const KaiF16HgemmKernel hgemm_sme2 =
    KAI_WRAP_UKERNEL_RUN_MATMUL_10_LHS_PACKED_OFFSET(matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot);


#if defined(ENABLE_QMX_KERNELS)

const KaiDynamicQGemmKernel qgemm_gemm_qmx =
    KAI_WRAP_UKERNEL_RUN_MATMUL_11(matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa);

const KaiF32SgemmKernel sgemm_gemm_qmx =
    KAI_WRAP_F32_SGEMM_LEGACY(matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa);
#endif // ENABLE_QMX_KERNELS

// Gemv kernels do not conform to the same ukernel interface layout
// Manual instantiation of this as per below is required
const KaiF32SgemvKernel sgemm_gemv_sme =
    {
        "kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla",
        KaiF32SgemvUkernel{
            kai_matmul_clamp_f32_f32_f32p_ukernel{
                kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
                kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
            },
        },
    };

const KaiF32SgemvKernel sgemm_gemv_sme2 =
    {
        "kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla",
        KaiF32SgemvUkernel{kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla()},
    };



const KaiQnbitGemmKernel& GetKleidiAIGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm;
    } else {
        return kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod;
    }
}

const KaiQnbitGemmKernel& GetKleidiAIGemvUKernel() {
    // The `sme2_dot` kernel uses SME2 SDOT instructions; `dot` here does not refer to
    // the separate FEAT_DotProd feature.
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
    } else {
        return kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod;
    }
}

const KaiQnbitAsymGemmKernel& GetKleidiAIQai4GemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme2_mopa;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm;
    } else {
        return kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod;
    }
}

const KaiQnbitAsymGemmKernel& GetKleidiAIQai4GemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme2_dot;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod;
    } else {
        return kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4x4_1x4_neon_dotprod;
    }
}

const KaiF32SgemmKernel& GetKleidiAISGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemm_sme2;
    } else {
#if defined(ENABLE_QMX_KERNELS)
        if (ArmKleidiAI::vendor_name.compare("Qualcomm") == 0)
        {
            KLEIDIAI_KERNEL_LOG("SGEMM: Using QMX Kernel");
            return sgemm_gemm_qmx;
        } else {
            return sgemm_gemm_sme;
        }
#else
        return sgemm_gemm_sme;
#endif // ENABLE_QMX_KERNELS
    }
}

const KaiF32SgemvKernel& GetKleidiAISGemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemv_sme2;
    } else {
        return sgemm_gemv_sme;
    }
}

const KaiF32IMatmulKernel& GetKleidiAIF32IMatmulUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return imatmul_conv_sme2;
    } else {
#if defined(ENABLE_QMX_KERNELS)
        if (ArmKleidiAI::vendor_name.compare("Qualcomm") == 0)
        {
            KLEIDIAI_KERNEL_LOG("IMATMUL: Using QMX Kernel");
            return imatmul_conv_qmx;
        } else {
            return imatmul_conv_sme;
        }
#else
        return imatmul_conv_sme;
#endif // ENABLE_QMX_KERNELS
    }
}

const KaiF16IMatmulKernel& GetKleidiAIF16IMatmulUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return imatmul_f16_conv_sme2;
    } else {
        return imatmul_f16_conv_sme;
    }
}

const KaiDynamicQGemmKernel& GetKleidiAIQGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return qgemm_gemm_sme2;
    } else {
#if defined(ENABLE_QMX_KERNELS)
        if (ArmKleidiAI::vendor_name.compare("Qualcomm") == 0)
        {
            KLEIDIAI_KERNEL_LOG("QGEMM: Using QMX Kernel");
            return qgemm_gemm_qmx;
        } else {
            return qgemm_gemm_sme;
        }
#else
        return qgemm_gemm_sme;
#endif // ENABLE_QMX_KERNELS
    }
}

const KaiBF16SBgemmKernel& GetKleidiAISBGemmUKernel() {
    // Currently only SME2 variant exists for bfloat16/SBGEMM kernel
    return sbgemm_gemm_sme2;
}

const KaiF16HgemmKernel& GetKleidiAIHgemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return hgemm_sme2;
    } else {
        return hgemm_sme;
    }
}
