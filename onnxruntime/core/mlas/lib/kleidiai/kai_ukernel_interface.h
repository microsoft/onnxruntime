//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p_qai4c32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"

#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p_f32p_interface.h"

#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p_f16p_interface.h"

#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h"

// Wrapper type that carries a stable "name" alongside the KAI ukernel interface.
// This avoids needing to infer which underlying microkernel was selected from a function pointer.
template <typename UkernelFn>
struct KaiMatmulKernel {
    const char* name;
    UkernelFn ukernel;
};

// Holds either a family-specific legacy interface or the common matmul ukernel API.
// Type-specific adapters derive from this and expose the operations required by their caller.
template <typename LegacyUkernel>
class KaiMatmulUkernel {
  protected:
    explicit KaiMatmulUkernel(LegacyUkernel ukernel) : legacy_ukernel_(ukernel) {}
    explicit KaiMatmulUkernel(kai_matmul_uker_api ukernel)
        : ukernel_api_(ukernel), uses_ukernel_api_(true) {}

    bool UsesUkernelApi() const { return uses_ukernel_api_; }
    const LegacyUkernel& LegacyUkernelInterface() const { return legacy_ukernel_; }
    const kai_matmul_uker_api& UkernelApi() const { return ukernel_api_; }

  private:
    LegacyUkernel legacy_ukernel_{};
    kai_matmul_uker_api ukernel_api_{};
    bool uses_ukernel_api_{false};
};

enum class KaiF32RhsLayout {
    KxN,
    NxK,
};

class KaiF32SgemmUkernel final : public KaiMatmulUkernel<kai_matmul_clamp_f32_f32p_f32p_ukernel> {
  public:
    explicit KaiF32SgemmUkernel(kai_matmul_clamp_f32_f32p_f32p_ukernel ukernel);
    KaiF32SgemmUkernel(kai_matmul_uker_api ukernel,
                       kai_matmul_pack_lhs_uker_api lhs_packer,
                       kai_matmul_pack_rhs_uker_api rhs_kxn_packer,
                       kai_matmul_pack_rhs_uker_api rhs_nxk_packer);

    size_t get_m_step() const;
    size_t get_n_step() const;
    size_t get_lhs_packed_offset(size_t m_idx, size_t k) const;
    size_t get_rhs_packed_offset(size_t n_idx, size_t k) const;
    void run_matmul(size_t m, size_t n, size_t k,
                    const void* lhs_packed, const void* rhs_packed, void* dst,
                    size_t dst_stride_row, size_t dst_stride_col,
                    float clamp_min, float clamp_max) const;

    size_t GetLhsPackedSize(size_t m, size_t k) const;
    const char* GetLhsPackerName() const;
    void PackLhs(size_t m, size_t k, const float* lhs, size_t lhs_stride, void* lhs_packed) const;
    size_t GetRhsPackedSize(KaiF32RhsLayout layout, size_t n, size_t k) const;
    const char* GetRhsPackerName(KaiF32RhsLayout layout) const;
    void PackRhs(KaiF32RhsLayout layout, size_t n, size_t k,
                 const float* rhs, size_t rhs_stride, const float* bias, void* rhs_packed) const;

  private:
    const kai_matmul_pack_rhs_uker_api& RhsPacker(KaiF32RhsLayout layout) const;

    kai_matmul_pack_lhs_uker_api lhs_packer_{};
    kai_matmul_pack_rhs_uker_api rhs_kxn_packer_{};
    kai_matmul_pack_rhs_uker_api rhs_nxk_packer_{};
};

class KaiF32SgemvUkernel final : public KaiMatmulUkernel<kai_matmul_clamp_f32_f32_f32p_ukernel> {
  public:
    explicit KaiF32SgemvUkernel(kai_matmul_clamp_f32_f32_f32p_ukernel ukernel);
    explicit KaiF32SgemvUkernel(kai_matmul_uker_api ukernel);

    void run_matmul(size_t m, size_t n, size_t k,
                    const void* lhs, size_t lhs_stride, const void* rhs_packed, void* dst,
                    size_t dst_stride_row, size_t dst_stride_col,
                    float clamp_min, float clamp_max) const;
};

enum class KaiQ4RhsPackLayout {
    SymmetricNxK,
    SymmetricNxKInterleavedNrx4,
    AsymmetricNxK,
    AsymmetricNxKInterleavedNrx4,
};

template <typename UkernelFn>
struct KaiQ4MatmulKernel {
    const char* name;
    UkernelFn ukernel;
    KaiQ4RhsPackLayout rhs_layout;
};

// Wrapper for FP32 GEMM kernels where both LHS and RHS are pre-packed (common SGEMM path).
using KaiF32SgemmKernel = KaiMatmulKernel<KaiF32SgemmUkernel>;

// Wrapper for FP32 kernels used for GEMV-style workloads (typically a single-row/skinny-M use case).
using KaiF32SgemvKernel = KaiMatmulKernel<KaiF32SgemvUkernel>;

// Wrapper for Qnbit GEMM kernels producing FP32 output.
using KaiQnbitGemmKernel = KaiQ4MatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>;

// Wrapper for Qnbit Asymmetric-quantized GEMM kernels producing FP32 output.
using KaiQnbitAsymGemmKernel = KaiQ4MatmulKernel<kai_matmul_clamp_f32_qsi8d32p_qai4c32p_ukernel>;

// Wrapper for dynamic-quantized GEMM kernels producing FP32 output.
using KaiDynamicQGemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>;

// Wrapper for FP32 IMATMUL kernels used by the KleidiAI convolution implementation.
using KaiF32IMatmulKernel = KaiMatmulKernel<kai_imatmul_clamp_f32_f32p_f32p_ukernel>;

// Wrapper for FP16 IMATMUL kernels used by the KleidiAI convolution implementation.
using KaiF16IMatmulKernel = KaiMatmulKernel<kai_imatmul_clamp_f16_f16p_f16p_ukernel>;

using KaiBF16SBgemmKernel = KaiMatmulKernel<kai_matmul_clamp_f32_bf16p_bf16p_ukernel>;

// Wrapper for FP16 HGEMM kernels producing FP16 output.
using KaiF16HgemmKernel = KaiMatmulKernel<kai_matmul_clamp_f16_f16_f16p_ukernel>;

// Returns the selected Qnbit GEMM ukernel based on runtime CPU capabilities.
const KaiQnbitGemmKernel& GetKleidiAIGemmUKernel();

// Returns the selected Qnbit kernel used for GEMV-style workloads based on runtime CPU capabilities.
const KaiQnbitGemmKernel& GetKleidiAIGemvUKernel();

// Returns the selected Qnbit Asymmetric-quantized GEMM ukernel.
const KaiQnbitAsymGemmKernel& GetKleidiAIQai4GemmUKernel();

// Returns the selected Qnbit Asymmetric-quantized kernel used for GEMV-style workloads.
const KaiQnbitAsymGemmKernel& GetKleidiAIQai4GemvUKernel();

// Returns the selected dynamic-quantized GEMM ukernel based on runtime CPU capabilities and optional vendor selection.
const KaiDynamicQGemmKernel& GetKleidiAIQGemmUKernel();

// Returns the selected FP32 SGEMM ukernel based on runtime CPU capabilities and optional vendor selection.
const KaiF32SgemmKernel& GetKleidiAISGemmUKernel();

// Returns the selected FP32 kernel used for GEMV-style workloads based on runtime CPU capabilities.
const KaiF32SgemvKernel& GetKleidiAISGemvUKernel();

// Returns the selected FP32 IMATMUL ukernel used by the KleidiAI convolution implementation.
const KaiF32IMatmulKernel& GetKleidiAIF32IMatmulUKernel();

// Returns the selected FP16 IMATMUL ukernel used by the KleidiAI convolution implementation.
const KaiF16IMatmulKernel& GetKleidiAIF16IMatmulUKernel();

// Returns the selected BF16 SBGEMM ukernel used by the KleidiAI based on runtime CPU capabilities.
const KaiBF16SBgemmKernel& GetKleidiAISBGemmUKernel();

// Returns the selected FP16 HGEMM ukernel based on runtime CPU capabilities.
const KaiF16HgemmKernel& GetKleidiAIHgemmUKernel();
