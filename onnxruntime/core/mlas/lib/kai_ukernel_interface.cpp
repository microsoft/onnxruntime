//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "kai_ukernel_interface.h"
#include "mlasi.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"

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

const kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa =
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_get_dst_size_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
      kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa};

class KaiQai8dxpQsi4c32pStrategy: public KleidiAIQ4BitGemmStrategy {
    const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel;

public:
    KaiQai8dxpQsi4c32pStrategy(const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel &ukernel):
        ukernel(ukernel) {}

    size_t GetMStep() const override {
        return ukernel.get_m_step();
    }

    size_t GetNStep() const override {
        return ukernel.get_n_step();
    }

    size_t GetRHSPackedSize(size_t N, size_t K, size_t BlkLen) const override {
        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, BlkLen, kai_dt_bf16);
    }

    size_t GetLHSPackedSize(size_t M, size_t K) const override {
        const size_t mr = ukernel.get_mr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();
        return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    }

    void RunRHSPack(size_t N, size_t K, size_t BlkLen, const std::byte* QuantBData,
            const float* QuantBScale, std::byte* PackedQuantBData) const override {
        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = kai_dt_bf16;

        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        const size_t scales_len = N * BlockCountK;
        std::vector<uint16_t> scales(scales_len);
        for (size_t i = 0; i < scales_len; i++) {
            const uint32_t* i32 = reinterpret_cast<const uint32_t*>(&QuantBScale[i]);
            scales[i] = *i32 >> 16;
        }

        kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1, N, K, nr, kr, sr, BlkLen,
                reinterpret_cast<const uint8_t*>(QuantBData), BlockCountK * BlkLen / 2,
                nullptr, scales.data(), BlockCountK * sizeof(uint16_t),
                PackedQuantBData, 0, &params);
    }

    void RunLHSPack(size_t M, size_t K, const float* A, std::byte* QuantA) const override {
        const size_t mr = ukernel.get_mr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        const size_t src_stride = K * sizeof(float);
        const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(0, src_stride);
        const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                                0, K, mr, kr, sr);

        const float* src_ptr = reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(A) + lhs_offset);
        void* dst_ptr = QuantA + lhs_packed_offset;

        kai_run_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr, 0, src_ptr, src_stride, dst_ptr);
    }

    void RunMatMul(
        size_t BlkLen,
        const std::byte* QuantA,
        const std::byte* PackedQuantBData,
        float* C,
        const size_t RangeStartM,
        const size_t RangeCountM,
        const size_t RangeStartN,
        const size_t RangeCountN,
        size_t CountK,
        size_t ldc
    ) const override {
        const size_t dst_stride = ldc * sizeof(float);

        const size_t lhs_packed_offset = ukernel.get_lhs_packed_offset(RangeStartM, CountK);
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(RangeStartN, CountK, BlkLen);
        const size_t dst_offset = ukernel.get_dst_offset(RangeStartM, RangeStartN, dst_stride);

        const void* lhs_ptr = QuantA + lhs_packed_offset;
        const void* rhs_ptr = PackedQuantBData + rhs_packed_offset;
        float* dst_ptr = reinterpret_cast<float*>(reinterpret_cast<std::byte*>(C) + dst_offset);

        ukernel.run_matmul(
            RangeCountM, RangeCountN, CountK, BlkLen, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }
};

class KaiQsi8d32pQsi4c32pStrategy: public KleidiAIQ4BitGemmStrategy {
    const kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;

public:
    KaiQsi8d32pQsi4c32pStrategy(const kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel &ukernel):
        ukernel(ukernel) {}

    size_t GetMStep() const override {
        return ukernel.get_m_step();
    }

    size_t GetNStep() const override {
        return ukernel.get_n_step();
    }

    size_t GetRHSPackedSize(size_t N, size_t K, size_t) const override {
        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(N, K, nr, kr, 32);
    }

    size_t GetLHSPackedSize(size_t M, size_t K) const override {
        const size_t mr = ukernel.get_mr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();
        return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(M, K, 32, mr, kr, sr);
    }

    void RunRHSPack(size_t N, size_t K, size_t BlkLen, const std::byte* QuantBData,
            const float* QuantBScale, std::byte* PackedQuantBData) const override {
        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        kai_rhs_pack_qs4cxs1s0_param params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;

        const size_t PackedBlkLen = 32;
        const size_t BlkRatio = BlkLen / PackedBlkLen;
        const size_t BlockBytes = PackedBlkLen / 2 + 2;
        const size_t BlockCountK = MlasDivRoundup(K, PackedBlkLen);
        const size_t scales_len = N * BlockCountK;
        std::vector<uint8_t> rhs(scales_len * (PackedBlkLen / 2 + 2));
        for (size_t i = 0; i < scales_len; i++) {
            uint16_t *scale = reinterpret_cast<uint16_t *>(&rhs[i * BlockBytes]);
            *scale = kai_cast_f16_f32(QuantBScale[i / BlkRatio]);

            uint8_t block[PackedBlkLen];
            for (size_t j = 0; j < PackedBlkLen / 2; j++) {
                const uint8_t v = static_cast<uint8_t>(QuantBData[i * PackedBlkLen / 2 + j]);
                const uint8_t v0 = v & 0xF;
                const uint8_t v1 = v >> 4;
                block[j * 2] = v0;
                block[j * 2 + 1] = v1;
            }

            for (size_t j = 0; j < PackedBlkLen / 2; j++) {
                const uint8_t v0 = block[j];
                const uint8_t v1 = block[j + PackedBlkLen / 2];
                rhs[i * BlockBytes + 2 + j] = v0 | (v1 << 4);
            }
        }

        kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(1, N, K, nr, kr, sr, PackedBlkLen,
                rhs.data(), nullptr, PackedQuantBData, 0, &params);
    }

    void RunLHSPack(size_t M, size_t K, const float* A, std::byte* QuantA) const override {
        const size_t mr = ukernel.get_mr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        const size_t src_stride = K * sizeof(float);
        const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32_neon(0, src_stride);
        const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32_neon(
                                0, K, 32, mr, kr, sr);

        const float* src_ptr = reinterpret_cast<const float*>(
                                reinterpret_cast<const uint8_t*>(A) + lhs_offset);
        void*        dst_ptr = reinterpret_cast<void *>(
                                reinterpret_cast<uint8_t*>(QuantA) + lhs_packed_offset);

        kai_run_lhs_quant_pack_qsi8d32p_f32_neon(M, K, 32, mr, kr, sr, 0, src_ptr, src_stride, dst_ptr);
    }

    void RunMatMul(
        size_t,
        const std::byte* QuantA,
        const std::byte* PackedQuantBData,
        float* C,
        const size_t RangeStartM,
        const size_t RangeCountM,
        const size_t RangeStartN,
        const size_t RangeCountN,
        size_t CountK,
        size_t ldc
    ) const override {
        const size_t dst_stride = ldc * sizeof(float);

        const size_t lhs_packed_offset = ukernel.get_lhs_packed_offset(RangeStartM, CountK, 32);
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(RangeStartN, CountK, 32);
        const size_t dst_offset = ukernel.get_dst_offset(RangeStartM, RangeStartN, dst_stride);

        const void* lhs_ptr = reinterpret_cast<const void*>(
                reinterpret_cast<const char *>(QuantA) + lhs_packed_offset);
        const void* rhs_ptr = reinterpret_cast<const void*>(
                reinterpret_cast<const char *>(PackedQuantBData) + rhs_packed_offset);
        float* dst_ptr = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(C) + dst_offset);

        ukernel.run_matmul(
            RangeCountM, RangeCountN, CountK, 32, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }
};

const KaiQai8dxpQsi4c32pStrategy kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_strategy
    (kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod);

const KaiQai8dxpQsi4c32pStrategy kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_strategy
    (kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod);

const KaiQai8dxpQsi4c32pStrategy kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_strategy
    (kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod);

const KaiQai8dxpQsi4c32pStrategy kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_strategy
    (kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm);

const KaiQsi8d32pQsi4c32pStrategy kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_strategy
    (kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa);

const KleidiAIQ4BitGemmStrategy& GetKleidiAIGemmStrategy() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        return kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_strategy;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm_strategy;
    } else {
        return kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod_strategy;
    }
}

const KleidiAIQ4BitGemmStrategy& GetKleidiAIGemvStrategy() {
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME()) {
        return kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa_strategy;
    } else if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeon_I8MM()) {
        return kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod_strategy;
    } else {
        return kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod_strategy;
    }
}
