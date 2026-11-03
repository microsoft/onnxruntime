// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

/*++

Module Name:

    qnbitgemm_kleidiai.cpp

Abstract:

    This module implements the KleidiAI QNBitGemm backend overrides.

--*/

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../qnbitgemm.h"
#include "../sqnbitgemm_q8_block.h"
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"
#include "kai_ukernel_interface.h"
#include "mlasi_kleidiai.h"

namespace
{

// Maps ORT's unsigned Q4 range [0, 15] to KleidiAI's signed-int4 origin [-8, 7].
constexpr int32_t kKaiQ4SignedZeroPointOffset = 8;

enum class KleidiAIQ4Backend {
    None,
    Qai8dxpQsi4c32p,   // 4-bit symmetric block-quantized RHS
    Qsi8d32pQai4c32p,  // 4-bit asymmetric block-quantized RHS
};

bool
IsKleidiAIQ4ShapeSupported(
    size_t K,
    size_t BlkLen,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    if (BackendKernelSelectorConfig != nullptr && !BackendKernelSelectorConfig->use_kleidiai) {
        return false;
    }

    if (K == 0 || BlkLen == 0) {
        return false;
    }

    return (BlkLen % 32) == 0 && (K % BlkLen) == 0;
}

KleidiAIQ4Backend
SelectKleidiAIQ4Backend(
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    if (!IsKleidiAIQ4ShapeSupported(K, BlkLen, BackendKernelSelectorConfig)) {
        return KleidiAIQ4Backend::None;
    }

    const auto& cpuid = MLAS_CPUIDINFO::GetCPUIDInfo();
    const bool has_kleidiai_q4 =
        cpuid.HasArm_SME2() || cpuid.HasArmNeon_I8MM() || cpuid.HasArmNeonDot();
    if (!has_kleidiai_q4) {
        return KleidiAIQ4Backend::None;
    }

    // Zero-point inputs are handled exclusively by the dedicated asymmetric kernels.
    return HasZeroPoint ? KleidiAIQ4Backend::Qsi8d32pQai4c32p
                        : KleidiAIQ4Backend::Qai8dxpQsi4c32p;
}

size_t
GetKleidiAIQ4PackedQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p: {
            const auto& kernel = GetKleidiAIGemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t nr = ukernel.get_nr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            size_t packed_size;
            switch (kernel.rhs_layout) {
                case KaiQ4RhsPackLayout::SymmetricNxK:
                    packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                        N, K, nr, kr, sr, BlkLen, kai_dt_bf16
                    );
                    break;
                case KaiQ4RhsPackLayout::SymmetricNxKInterleavedNrx4:
                    packed_size =
                        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
                            N, K, nr, kr, sr, BlkLen, kai_dt_bf16
                        );
                    break;
                default:
                    MLAS_THROW_EX(std::runtime_error,
                                  "Unexpected RHS layout for symmetric Q4 backend.");
            }

            return packed_size;
        }
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p: {
            // Packed B is shared by GEMM and GEMV, so both kernels must use this RHS layout.
            const auto& kernel = GetKleidiAIQai4GemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t nr = ukernel.get_nr();
            const size_t kr = ukernel.get_kr();

            switch (kernel.rhs_layout) {
                case KaiQ4RhsPackLayout::AsymmetricNxK:
                    return kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
                        N, K, nr, kr, BlkLen
                    );
                case KaiQ4RhsPackLayout::AsymmetricNxKInterleavedNrx4:
                    return kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
                        N, K, nr, kr, BlkLen
                    );
                default:
                    MLAS_THROW_EX(std::runtime_error,
                                  "Unexpected RHS layout for asymmetric Q4 backend.");
            }
        }
        case KleidiAIQ4Backend::None:
            assert(false);
            return 0;
    }

    return 0;
}

void
PackKleidiAIQ4QuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    const std::byte* QuantBData,
    const float* QuantBScale,
    bool HasZeroPoint,
    const std::byte* QuantBZeroPoint,
    std::byte* PackedQuantBData,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    if (QuantBData == nullptr) {
        return;
    }

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p: {
            const auto& kernel = GetKleidiAIGemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t nr = ukernel.get_nr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            assert(QuantBScale != nullptr);
            kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
            params.lhs_zero_point = 1;
            params.rhs_zero_point = kKaiQ4SignedZeroPointOffset;
            params.scale_dt = kai_dt_bf16;

            size_t scales_len;
            if (MlasMultiplyOverflowsSizeT(N, BlockCountK, &scales_len)) {
                MLAS_THROW_EX(std::overflow_error, "KleidiAI QNBitGemm scale count overflow.");
            }
            std::vector<uint16_t> scales(scales_len);
            for (size_t i = 0; i < scales_len; ++i) {
                uint32_t bits;
                static_assert(sizeof(bits) == sizeof(QuantBScale[i]), "Unexpected float size");
                std::memcpy(&bits, &QuantBScale[i], sizeof(bits));
                scales[i] = static_cast<uint16_t>(bits >> 16);
            }

            const auto* rhs = reinterpret_cast<const uint8_t*>(QuantBData);
            const size_t rhs_stride = BlockCountK * BlkLen / 2;
            const size_t scale_stride = BlockCountK * sizeof(uint16_t);
            switch (kernel.rhs_layout) {
                case KaiQ4RhsPackLayout::SymmetricNxK:
                    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                        1, N, K, nr, kr, sr, BlkLen, rhs, rhs_stride, nullptr, scales.data(), scale_stride,
                        PackedQuantBData, 0, &params
                    );
                    return;
                case KaiQ4RhsPackLayout::SymmetricNxKInterleavedNrx4:
                    kai_run_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
                        1, N, K, nr, kr, sr, BlkLen, rhs, rhs_stride, nullptr, scales.data(), scale_stride,
                        PackedQuantBData, 0, &params
                    );
                    return;
                default:
                    MLAS_THROW_EX(std::runtime_error,
                                  "Unexpected RHS layout for symmetric Q4 backend.");
            }
        }
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p: {
            // Packed B is shared by GEMM and GEMV, so both kernels must use this RHS layout.
            const auto& kernel = GetKleidiAIQai4GemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t nr = ukernel.get_nr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            assert(QuantBScale != nullptr);
            assert(QuantBZeroPoint != nullptr);
            kai_rhs_pack_nxk_qai4c32p_params params{};
            params.lhs_zero_point = 1;
            params.rhs_zero_point = kKaiQ4SignedZeroPointOffset;

            const size_t zp_stride = MlasDivRoundup(BlockCountK, size_t{2});
            const size_t rhs_stride = K / 2;
            size_t zero_offsets_size;
            size_t rhs_for_kai_size;
            if (MlasMultiplyOverflowsSizeT(nr, BlockCountK, &zero_offsets_size) ||
                MlasMultiplyOverflowsSizeT(nr, rhs_stride, &rhs_for_kai_size)) {
                MLAS_THROW_EX(std::overflow_error, "KleidiAI QNBitGemm scratch size overflow.");
            }
            // Reuse one kernel-width panel to keep scratch memory independent of N.
            std::vector<float> zero_offsets(zero_offsets_size);
            std::vector<uint8_t> rhs_for_kai(rhs_for_kai_size);
            const auto* rhs = reinterpret_cast<const uint8_t*>(QuantBData);

            for (size_t panel_start = 0; panel_start < N; panel_start += nr) {
                const size_t panel_rows = std::min(nr, N - panel_start);

                for (size_t panel_n = 0; panel_n < panel_rows; ++panel_n) {
                    const size_t n = panel_start + panel_n;
                    for (size_t block = 0; block < BlockCountK; ++block) {
                        const uint8_t zp_byte =
                            static_cast<uint8_t>(QuantBZeroPoint[n * zp_stride + block / 2]);
                        const uint8_t zp = (block & 1) == 0 ? (zp_byte & 0x0F) : (zp_byte >> 4);
                        const size_t source_index = n * BlockCountK + block;
                        const size_t panel_index = panel_n * BlockCountK + block;
                        const float kai_zp_offset =
                            static_cast<float>(params.rhs_zero_point) - static_cast<float>(zp);
                        zero_offsets[panel_index] = kai_zp_offset * QuantBScale[source_index];
                    }
                }

                const uint8_t* rhs_panel = rhs + panel_start * rhs_stride;
                const size_t rhs_panel_size = panel_rows * rhs_stride;
                for (size_t i = 0; i < rhs_panel_size; ++i) {
                    rhs_for_kai[i] =
                        static_cast<uint8_t>(((rhs_panel[i] & 0x0F) << 4) | ((rhs_panel[i] & 0xF0) >> 4));
                }

                const float* scales_panel = QuantBScale + panel_start * BlockCountK;

                switch (kernel.rhs_layout) {
                    case KaiQ4RhsPackLayout::AsymmetricNxK: {
                        const size_t packed_offset =
                            kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
                                panel_start, K, nr, kr, BlkLen
                            );
                        kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
                            1, panel_rows, K, nr, kr, sr, BlkLen, rhs_for_kai.data(), zero_offsets.data(),
                            nullptr, scales_panel, PackedQuantBData + packed_offset, 0, &params
                        );
                        break;
                    }
                    case KaiQ4RhsPackLayout::AsymmetricNxKInterleavedNrx4: {
                        const size_t packed_offset =
                            kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
                                panel_start, K, nr, kr, BlkLen
                            );
                        kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
                            1, panel_rows, K, nr, kr, sr, BlkLen, rhs_for_kai.data(), zero_offsets.data(),
                            nullptr, scales_panel, PackedQuantBData + packed_offset, 0, &params
                        );
                        break;
                    }
                    default:
                        MLAS_THROW_EX(std::runtime_error,
                                      "Unexpected RHS layout for asymmetric Q4 backend.");
                }
            }
            return;
        }
        case KleidiAIQ4Backend::None:
            assert(false);
            return;
    }
}

size_t
GetKleidiAIQ4PerGemmWorkspaceSize(
    size_t M,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p: {
            const auto& kernel = (M == 1) ? GetKleidiAIGemvUKernel() : GetKleidiAIGemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t mr = ukernel.get_mr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
        }
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p: {
            const auto& kernel = (M == 1) ? GetKleidiAIQai4GemvUKernel() : GetKleidiAIQai4GemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t mr = ukernel.get_mr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
                M, K, BlkLen, mr, kr, sr
            );
        }
        case KleidiAIQ4Backend::None:
            assert(false);
            return 0;
    }

    return 0;
}

size_t
GetKleidiAIQ4PerGemmWorkspaceStride(
    size_t M,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    const size_t workspace_size =
        GetKleidiAIQ4PerGemmWorkspaceSize(M, K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig);
    const size_t alignment = Q8BlkAlignment();
    return MlasDivRoundup(workspace_size, alignment) * alignment;
}

template <typename KleidiAIKernel>
size_t
GetKleidiAIQ4NAlignment(const KleidiAIKernel& kernel)
{
    const auto& ukernel = kernel.ukernel;
    const size_t n_step = ukernel.get_n_step();
    const size_t nr = ukernel.get_nr();
    const size_t n_alignment = std::max(n_step, nr);

    assert(n_step != 0 && nr != 0);
    assert((n_alignment % n_step) == 0 && (n_alignment % nr) == 0);

    return n_alignment;
}

size_t
GetPackedQ4BitGemmNAlignment(
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p:
            return std::max(
                GetKleidiAIQ4NAlignment(GetKleidiAIGemmUKernel()),
                GetKleidiAIQ4NAlignment(GetKleidiAIGemvUKernel())
            );
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p:
            return std::max(
                GetKleidiAIQ4NAlignment(GetKleidiAIQai4GemmUKernel()),
                GetKleidiAIQ4NAlignment(GetKleidiAIQai4GemvUKernel())
            );
        case KleidiAIQ4Backend::None:
            assert(false);
            return MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    }

    return MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
}

void
PackKleidiAIQ4Lhs(
    size_t BlkLen,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    bool HasZeroPoint,
    std::byte* QuantA,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p: {
            const auto& kernel = (M == 1) ? GetKleidiAIGemvUKernel() : GetKleidiAIGemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t mr = ukernel.get_mr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            const size_t source_stride = lda * sizeof(float);
            const size_t source_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(0, source_stride);
            const size_t packed_offset =
                kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(0, K, mr, kr, sr);

            const float* source =
                reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(A) + source_offset);
            void* destination = QuantA + packed_offset;

            kai_run_lhs_quant_pack_qai8dxp_f32(
                M, K, mr, kr, sr, 0, source, source_stride, destination
            );
            return;
        }
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p: {
            const auto& kernel = (M == 1) ? GetKleidiAIQai4GemvUKernel() : GetKleidiAIQai4GemmUKernel();
            const auto& ukernel = kernel.ukernel;
            const size_t mr = ukernel.get_mr();
            const size_t kr = ukernel.get_kr();
            const size_t sr = ukernel.get_sr();

            const size_t source_stride = lda * sizeof(float);
            const size_t source_offset =
                kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(0, source_stride);
            const size_t packed_offset =
                kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
                    0, K, BlkLen, mr, kr, sr
                );

            const float* source =
                reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(A) + source_offset);
            void* destination = QuantA + packed_offset;

            kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
                M, K, BlkLen, mr, kr, sr, 0, source, source_stride, destination
            );
            return;
        }
        case KleidiAIQ4Backend::None:
            assert(false);
            return;
    }
}

size_t
GetKleidiAIQ4LhsPackedOffset(
    const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel,
    size_t m,
    size_t k,
    size_t /* BlkLen */
)
{
    return ukernel.get_lhs_packed_offset(m, k);
}

size_t
GetKleidiAIQ4LhsPackedOffset(
    const kai_matmul_clamp_f32_qsi8d32p_qai4c32p_ukernel& ukernel,
    size_t m,
    size_t k,
    size_t BlkLen
)
{
    return ukernel.get_lhs_packed_offset(m, k, BlkLen);
}

template <typename KleidiAIKernel>
void
RunKleidiAIQ4Packed(
    const KleidiAIKernel& kernel,
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* PackedQuantBData,
    float* C,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN,
    size_t K,
    size_t ldc,
    const float* Bias
)
{
    const auto& ukernel = kernel.ukernel;
    const size_t destination_stride = ldc * sizeof(float);

    const size_t lhs_packed_offset =
        GetKleidiAIQ4LhsPackedOffset(ukernel, RangeStartM, K, BlkLen);
    const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(RangeStartN, K, BlkLen);
    const size_t destination_offset =
        ukernel.get_dst_offset(RangeStartM, RangeStartN, destination_stride);

    const void* lhs = QuantA + lhs_packed_offset;
    const void* rhs = PackedQuantBData + rhs_packed_offset;
    float* destination = reinterpret_cast<float*>(reinterpret_cast<std::byte*>(C) + destination_offset);

    ukernel.run_matmul(
        RangeCountM, RangeCountN, K, BlkLen, lhs, rhs, destination, destination_stride, sizeof(float),
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
    );

    if (Bias != nullptr) {
        for (size_t m = RangeStartM; m < RangeStartM + RangeCountM; ++m) {
            for (size_t n = RangeStartN; n < RangeStartN + RangeCountN; ++n) {
                C[m * ldc + n] += Bias[n];
            }
        }
    }
}

void
RunKleidiAIQ4Tile(
    size_t BlkLen,
    size_t K,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* DataParams,
    const std::byte* QuantA,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
)
{
    const bool HasZeroPoint = DataParams->QuantBZeroPoint != nullptr;
    switch (SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)) {
        case KleidiAIQ4Backend::Qai8dxpQsi4c32p: {
            const auto& kernel = (RangeCountM == 1 && RangeStartM == 0)
                                     ? GetKleidiAIGemvUKernel()
                                     : GetKleidiAIGemmUKernel();
            RunKleidiAIQ4Packed(
                kernel, BlkLen, QuantA, DataParams->PackedQuantBData, DataParams->C, RangeStartM,
                RangeCountM, RangeStartN, RangeCountN, K, DataParams->ldc, DataParams->Bias
            );
            break;
        }
        case KleidiAIQ4Backend::Qsi8d32pQai4c32p: {
            const auto& kernel = (RangeCountM == 1 && RangeStartM == 0)
                                     ? GetKleidiAIQai4GemvUKernel()
                                     : GetKleidiAIQai4GemmUKernel();
            RunKleidiAIQ4Packed(
                kernel, BlkLen, QuantA, DataParams->PackedQuantBData, DataParams->C, RangeStartM,
                RangeCountM, RangeStartN, RangeCountN, K, DataParams->ldc, DataParams->Bias
            );
            break;
        }
        case KleidiAIQ4Backend::None:
            assert(false);
            return;
    }

    if (DataParams->PostProcessor != nullptr) {
        DataParams->PostProcessor->Process(
            DataParams->C, RangeStartM, RangeStartN, RangeCountM, RangeCountN, DataParams->ldc
        );
    }
}

}  // namespace

bool
    MLASCALL
    ArmKleidiAI::MlasQNBitGemmIsSupported(
        size_t K,
        size_t BlkBitWidth,
        size_t BlkLen,
        bool HasZeroPoint,
        MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    const bool IsSupportedBlockLength =
        BlkLen == 32 || BlkLen == 64 || BlkLen == 128 || BlkLen == 256;
    return BlkBitWidth == 4 && IsSupportedBlockLength && ComputeType == SQNBIT_CompInt8 &&
           SelectKleidiAIQ4Backend(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig) !=
               KleidiAIQ4Backend::None;
}

size_t
    MLASCALL
    ArmKleidiAI::MlasQNBitGemmPackQuantBDataSize(
        size_t N,
        size_t K,
        size_t BlkBitWidth,
        size_t BlkLen,
        bool HasZeroPoint,
        MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(BlkBitWidth);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    assert(MlasQNBitGemmIsSupported(
        K, BlkBitWidth, BlkLen, HasZeroPoint, ComputeType, BackendKernelSelectorConfig
    ));
    return GetKleidiAIQ4PackedQuantBDataSize(N, K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig);
}

void
    MLASCALL
    ArmKleidiAI::MlasQNBitGemmPackQuantBData(
        size_t N,
        size_t K,
        size_t BlkBitWidth,
        size_t BlkLen,
        MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const void* QuantBData,
        void* PackedQuantBData,
        const void* QuantBScale,
        bool HasZeroPoint,
        const void* QuantBZeroPoint,
        MLAS_THREADPOOL* ThreadPool,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(BlkBitWidth);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
    assert(MlasQNBitGemmIsSupported(
        K, BlkBitWidth, BlkLen, HasZeroPoint, ComputeType, BackendKernelSelectorConfig
    ));
    PackKleidiAIQ4QuantBData(
        N, K, BlkLen, static_cast<const std::byte*>(QuantBData), static_cast<const float*>(QuantBScale),
        HasZeroPoint, static_cast<const std::byte*>(QuantBZeroPoint), static_cast<std::byte*>(PackedQuantBData),
        BackendKernelSelectorConfig
    );
}

size_t
    MLASCALL
    ArmKleidiAI::MlasQNBitGemmBatchWorkspaceSize(
        size_t M,
        size_t N,
        size_t K,
        size_t BatchN,
        size_t BlkBitWidth,
        size_t BlkLen,
        bool HasZeroPoint,
        MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(BlkBitWidth);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    assert(MlasQNBitGemmIsSupported(
        K, BlkBitWidth, BlkLen, HasZeroPoint, ComputeType, BackendKernelSelectorConfig
    ));

    const size_t PerGemmWorkspaceStride =
        GetKleidiAIQ4PerGemmWorkspaceStride(M, K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig);
    if (PerGemmWorkspaceStride == 0) {
        return 0;
    }

    size_t WorkspaceSize;
    if (MlasMultiplyOverflowsSizeT(BatchN, PerGemmWorkspaceStride, &WorkspaceSize) ||
        MlasAddOverflowsSizeT(WorkspaceSize, Q8BlkAlignment() - 1, &WorkspaceSize)) {
        MLAS_THROW_EX(std::overflow_error, "KleidiAI QNBitGemm batch workspace size overflow.");
    }

    return WorkspaceSize;
}

void
    MLASCALL
    ArmKleidiAI::MlasQNBitGemmBatch(
        size_t M,
        size_t N,
        size_t K,
        size_t BatchN,
        size_t BlkBitWidth,
        size_t BlkLen,
        MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* DataParams,
        void* Workspace,
        MLAS_THREADPOOL* ThreadPool,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(BlkBitWidth);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);

    // Empty outputs are no-ops; K == 0 falls back because bias may still be required.
    if (BatchN == 0 || M == 0 || N == 0) {
        return;
    }

    const bool HasZeroPoint = DataParams->QuantBZeroPoint != nullptr;
    // Workspace layout and backend selection are batch-wide.
    for (size_t gemm_index = 1; gemm_index < BatchN; ++gemm_index) {
        if ((DataParams[gemm_index].QuantBZeroPoint != nullptr) != HasZeroPoint) {
            MLAS_THROW_EX(
                std::invalid_argument,
                "KleidiAI QNBitGemm does not support mixing zero-point presence within a batch."
            );
        }
    }

    assert(MlasQNBitGemmIsSupported(
        K, BlkBitWidth, BlkLen, HasZeroPoint, ComputeType, BackendKernelSelectorConfig
    ));

    if (Workspace != nullptr) {
        const uintptr_t WorkspaceAddress = reinterpret_cast<uintptr_t>(Workspace);
        const size_t Alignment = Q8BlkAlignment();
        Workspace = reinterpret_cast<void*>((WorkspaceAddress + Alignment - 1) & ~(Alignment - 1));
    }

    const size_t PerGemmWorkspaceStride =
        GetKleidiAIQ4PerGemmWorkspaceStride(M, K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig);

    MlasTrySimpleParallel(ThreadPool, BatchN, [&](ptrdiff_t gemm_index) {
        const auto& data = DataParams[gemm_index];
        std::byte* quant_a =
            static_cast<std::byte*>(Workspace) + gemm_index * PerGemmWorkspaceStride;
        const bool has_zero_point = data.QuantBZeroPoint != nullptr;
        PackKleidiAIQ4Lhs(BlkLen, data.A, M, K, data.lda, has_zero_point, quant_a, BackendKernelSelectorConfig);
    });

    if (ThreadPool == nullptr) {
        for (size_t gemm_index = 0; gemm_index < BatchN; ++gemm_index) {
            const auto* data = &DataParams[gemm_index];
            const auto* quant_a =
                static_cast<const std::byte*>(Workspace) + gemm_index * PerGemmWorkspaceStride;
            RunKleidiAIQ4Tile(
                BlkLen, K, data, quant_a, 0, M, 0, N, BackendKernelSelectorConfig
            );
        }
        return;
    }

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);
    ptrdiff_t TargetThreadCount =
        ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;
    const ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;
    TargetThreadCount = std::min(TargetThreadCount, MaximumThreadCount);

    ptrdiff_t ThreadsPerGemm =
        std::max(TargetThreadCount / static_cast<ptrdiff_t>(BatchN), ptrdiff_t{1});
    constexpr size_t StrideM = 128;
    const size_t StrideNThreadAlignment = std::max(
        size_t{MLAS_QGEMM_STRIDEN_THREAD_ALIGN},
        GetPackedQ4BitGemmNAlignment(K, BlkLen, HasZeroPoint, BackendKernelSelectorConfig)
    );

    size_t columns_per_thread = N;
    if (ThreadsPerGemm > 1) {
        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t maximum_columns = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (maximum_columns < columns_per_thread) {
            columns_per_thread = std::min(
                columns_per_thread,
                MlasDivRoundup(maximum_columns, StrideNThreadAlignment) * StrideNThreadAlignment
            );
        }
    }

    const size_t StrideN = columns_per_thread;
    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t thread_id) {
        const size_t gemm_index = thread_id / ThreadsPerGemm;
        const size_t block_index = thread_id % ThreadsPerGemm;
        const auto* data = &DataParams[gemm_index];

        const size_t ThreadIdN = block_index / ThreadCountM;
        const size_t ThreadIdM = block_index % ThreadCountM;
        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, StrideM);
        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, StrideN);
        const auto* quant_a =
            static_cast<const std::byte*>(Workspace) + gemm_index * PerGemmWorkspaceStride;

        RunKleidiAIQ4Tile(
            BlkLen, K, data, quant_a, RangeStartM, RangeCountM, RangeStartN, RangeCountN,
            BackendKernelSelectorConfig
        );
    });
}

#if defined(MLAS_ENABLE_TEST_HOOKS) && defined(USE_KLEIDIAI)
const char*
ArmKleidiAI::GetKleidiAIQ4GemmKernelNameForTesting()
{
    return GetKleidiAIGemmUKernel().name;
}

const char*
ArmKleidiAI::GetKleidiAIQ4GemvKernelNameForTesting()
{
    return GetKleidiAIGemvUKernel().name;
}
#endif
