//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include "mlas.h"

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

namespace {
struct KaiHalfTlsBuffers {
    std::vector<MLAS_FP16> lhs_converted;
    std::vector<MLAS_FP16> rhs_converted;
    std::vector<MLAS_FP16> bias_zero;
    std::vector<std::byte> rhs_packed;

    void ReleaseLargeBuffers() {
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(lhs_converted);
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(rhs_converted);
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(bias_zero);
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(rhs_packed);
    }
};

struct ScopedKaiHalfTlsCleanup {
    KaiHalfTlsBuffers& buffers;

    ~ScopedKaiHalfTlsCleanup() {
        buffers.ReleaseLargeBuffers();
    }
};

thread_local KaiHalfTlsBuffers g_kai_half_tls;

template <typename T>
bool TryResizeVector(std::vector<T>& buffer, size_t size) {
    if (size > buffer.max_size()) {
        return false;
    }
    buffer.resize(size);
    return true;
}

static inline void ConvertFloatMatrixToHalf(
    const float* src,
    MLAS_FP16* dst,
    size_t rows,
    size_t cols,
    size_t src_ld) {
    for (size_t r = 0; r < rows; ++r) {
        MlasConvertFloatToHalfBuffer(src + r * src_ld, dst + r * cols, cols);
    }
}
}  // namespace

size_t
MLASCALL
ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
) {
    if (TransA != CblasNoTrans || TransB != CblasNoTrans || N == 0 || K == 0) {
        return 0;
    }

    return kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(N, K);
}

bool
MLASCALL
ArmKleidiAI::MlasHalfGemmKleidiAIPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const MLAS_FP16* B,
    size_t ldb,
    void* PackedB
) {
    if (TransA != CblasNoTrans || TransB != CblasNoTrans) {
        return false;
    }

    if (PackedB == nullptr || B == nullptr || N == 0 || K == 0 || ldb < N) {
        return false;
    }

    const size_t packed_rhs_size = ArmKleidiAI::MlasHalfGemmKleidiAIPackBSize(TransA, TransB, N, K);
    if (packed_rhs_size == 0) {
        return false;
    }

    std::vector<MLAS_FP16> zero_bias(N, MLAS_FP16::FromBits(0));

    size_t ldb_bytes = 0;
    if (MlasMultiplyOverflowsSizeT(ldb, sizeof(MLAS_FP16), &ldb_bytes)) {
        return false;
    }

    const auto& hgemm = GetKleidiAIHgemmUKernel();
    kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
        1, N, K, hgemm.ukernel.get_nr(), hgemm.ukernel.get_kr(), hgemm.ukernel.get_sr(), ldb_bytes,
        B,
        zero_bias.data(),
        nullptr,
        PackedB,
        0,
        nullptr);

    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasHalfGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    const MLAS_HALF_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
) {
    if (BackendKernelSelectorConfig != nullptr && !BackendKernelSelectorConfig->use_kleidiai) {
        return false;
    }
    if (BatchN == 0 || M == 0 || N == 0) {
        return true;
    }
    if (K == 0) {
        return false;
    }
    if (DataParams == nullptr) {
        return false;
    }

    ScopedKaiHalfTlsCleanup cleanup{g_kai_half_tls};

    // Validate all batch entries up front so we never partially execute and then
    // fall back (which would corrupt results for the already-written outputs).
    bool needs_rhs_packing = false;
    for (size_t b = 0; b < BatchN; ++b) {
        const auto& data = DataParams[b];
        if (data.OutputProcessor != nullptr) {
            return false;
        }
        if (data.BIsBackendNativePacked && data.Bias != nullptr) {
            return false;
        }
        if (data.BIsBackendNativePacked && data.ldb != 0) {
            return false;
        }
        // Native-packed RHS is consumed directly below. Only allocate the
        // runtime RHS packing scratch when at least one batch entry needs it.
        needs_rhs_packing = needs_rhs_packing || !data.BIsBackendNativePacked;
    }

    const auto& hgemm = GetKleidiAIHgemmUKernel();
    const size_t n_step = hgemm.ukernel.get_n_step();
    const size_t nr = hgemm.ukernel.get_nr();
    const size_t kr = hgemm.ukernel.get_kr();
    const size_t sr = hgemm.ukernel.get_sr();
    KLEIDIAI_KERNEL_LOG(hgemm.name);

    const size_t packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(N, K);
    if (packed_rhs_size == 0) {
        return false;
    }

    // TODO: Plumb MLAS_ACTIVATION through this call site if MLAS_HALF_GEMM_DATA_PARAMS
    // grows fused activation support.
    const float clamp_min = -std::numeric_limits<float>::infinity();
    const float clamp_max = std::numeric_limits<float>::infinity();

    if (needs_rhs_packing && !TryResizeVector(g_kai_half_tls.rhs_packed, packed_rhs_size)) {
        return false;
    }

    for (size_t b = 0; b < BatchN; ++b) {
        const auto& data = DataParams[b];

        const MLAS_FP16* lhs_base = reinterpret_cast<const MLAS_FP16*>(data.A);
        const MLAS_FP16* rhs_base = reinterpret_cast<const MLAS_FP16*>(data.B);
        const std::byte* rhs_packed = nullptr;
        size_t lhs_ld = data.lda;
        size_t rhs_ld = data.ldb;

        if (data.AIsfp32) {
            size_t lhs_elements = 0;
            if (MlasMultiplyOverflowsSizeT(M, K, &lhs_elements) ||
                !TryResizeVector(g_kai_half_tls.lhs_converted, lhs_elements)) {
                return false;
            }
            ConvertFloatMatrixToHalf(
                reinterpret_cast<const float*>(data.A),
                g_kai_half_tls.lhs_converted.data(),
                M, K, data.lda);
            lhs_base = g_kai_half_tls.lhs_converted.data();
            lhs_ld = K;
        }

        if (data.BIsBackendNativePacked) {
            rhs_packed = reinterpret_cast<const std::byte*>(data.B);
        } else if (data.ldb == 0) {
            // Prepacked B from MlasHalfGemmPackB/MlasHalfGemmConvertPackB.
            // For the current default halfgemm dispatch this is a row-major
            // fp16 KxN buffer with leading dimension N. It is not the native
            // KleidiAI RHS-packed layout, so this path falls back to packing
            // it into KleidiAI format before execution.
            rhs_ld = N;
        } else if (data.BIsfp32) {
            size_t rhs_elements = 0;
            if (MlasMultiplyOverflowsSizeT(K, N, &rhs_elements) ||
                !TryResizeVector(g_kai_half_tls.rhs_converted, rhs_elements)) {
                return false;
            }
            ConvertFloatMatrixToHalf(
                reinterpret_cast<const float*>(data.B),
                g_kai_half_tls.rhs_converted.data(),
                K, N, data.ldb);
            rhs_base = g_kai_half_tls.rhs_converted.data();
            rhs_ld = N;
        }

        if (rhs_packed == nullptr) {
            auto* rhs_packed_buffer = g_kai_half_tls.rhs_packed.data();

            size_t ldb_bytes = 0;
            if (MlasMultiplyOverflowsSizeT(rhs_ld, sizeof(MLAS_FP16), &ldb_bytes)) {
                return false;
            }
            if (data.Bias == nullptr) {
                if (!TryResizeVector(g_kai_half_tls.bias_zero, N)) {
                    return false;
                }
                std::fill(g_kai_half_tls.bias_zero.begin(), g_kai_half_tls.bias_zero.end(), MLAS_FP16::FromBits(0));
            }

            kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
                1, N, K, nr, kr, sr, ldb_bytes,
                rhs_base,
                data.Bias != nullptr ? data.Bias : g_kai_half_tls.bias_zero.data(),
                nullptr,
                rhs_packed_buffer,
                0,
                nullptr);
            rhs_packed = rhs_packed_buffer;
        }

        size_t lda_bytes = 0;
        if (MlasMultiplyOverflowsSizeT(lhs_ld, sizeof(MLAS_FP16), &lda_bytes)) {
            return false;
        }
        size_t dst_stride_bytes = 0;
        if (MlasMultiplyOverflowsSizeT(data.ldc, sizeof(MLAS_FP16), &dst_stride_bytes)) {
            return false;
        }

        MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(M), [&](ptrdiff_t m_idx) {
            const size_t m = static_cast<size_t>(m_idx);
            const auto* lhs = lhs_base + m * lhs_ld;
            auto* dst_row = data.C + m * data.ldc;
            const auto* rhs_packed_base = rhs_packed;
            // The selected KleidiAI HGEMM micro-kernel is 1xN by design.
            // We execute one output row per call and parallelize over rows.
            constexpr size_t kernel_m = 1;
            for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
                const size_t tile_n = std::min(n_step, N - n_idx);
                const auto* rhs_tile = rhs_packed_base + hgemm.ukernel.get_rhs_packed_offset(n_idx, K);
                auto* dst_tile = reinterpret_cast<MLAS_FP16*>(
                    reinterpret_cast<std::byte*>(dst_row) +
                    hgemm.ukernel.get_dst_offset(0, n_idx, dst_stride_bytes));

                hgemm.ukernel.run_matmul(
                    kernel_m,
                    tile_n,
                    K,
                    lhs,
                    lda_bytes,
                    rhs_tile,
                    dst_tile,
                    dst_stride_bytes,
                    sizeof(MLAS_FP16),
                    clamp_min,
                    clamp_max);
            }
        });

    }

    return true;
}
