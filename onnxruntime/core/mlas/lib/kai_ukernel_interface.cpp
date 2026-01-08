//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "kai_ukernel_interface.h"
#include "mlasi.h"
#include "kleidiai/mlasi_kleidiai.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod =
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod};

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod =
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod};

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod =
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod};

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm =
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm};

const kai_matmul_clamp_f32_f32_f32p_ukernel sgemm_gemv_sme =
    {kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla,
    kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla};

const kai_matmul_clamp_f32_f32_f32p_ukernel sgemm_gemv_sme2 =
    {kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_get_dst_size_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
    kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla};

const kai_matmul_clamp_f32_f32p_f32p_ukernel sgemm_gemm_sme =
    {kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa};

const kai_matmul_clamp_f32_f32p_f32p_ukernel sgemm_gemm_sme2 =
    {kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
    kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa};

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm;
    } else {
        return kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod;
    }
}

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& GetKleidiAIGemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
    } else {
        return kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod;
    }
}

const kai_matmul_clamp_f32_f32p_f32p_ukernel& GetKleidiAISGemmUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemm_sme2;
    } else {
        return sgemm_gemm_sme;
    }
}

const kai_matmul_clamp_f32_f32_f32p_ukernel& GetKleidiAISGemvUKernel() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2()) {
        return sgemm_gemv_sme2;
    } else {
        return sgemm_gemv_sme;
    }
}
