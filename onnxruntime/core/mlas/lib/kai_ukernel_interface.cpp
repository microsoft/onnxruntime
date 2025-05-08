//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "kai_ukernel_interface.h"
#include "mlasi.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"

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
