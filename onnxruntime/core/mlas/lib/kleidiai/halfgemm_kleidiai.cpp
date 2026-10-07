//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "mlas.h"

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

namespace {
constexpr const char* Sve2p1HalfGemmKernelName =
    "kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot";
constexpr size_t Sve2p1PackedRhsMetadataSize = MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT;
static_assert(sizeof(size_t) <= Sve2p1PackedRhsMetadataSize);

enum class KaiHalfGemmBackend {
    None,
    Sme,
    Sve2p1,
};

KaiHalfGemmBackend SelectKaiHalfGemmBackend() {
    const auto& cpuid_info = MLAS_CPUIDINFO::GetCPUIDInfo();
    // SME/SME2 share the existing RHS layout; SVE2.1 must use its paired SVE layout.
    if (cpuid_info.HasArm_SME() || cpuid_info.HasArm_SME2()) {
        return KaiHalfGemmBackend::Sme;
    }
    if (cpuid_info.HasArmSVE2p1()) {
        return KaiHalfGemmBackend::Sve2p1;
    }
    return KaiHalfGemmBackend::None;
}

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

size_t GetSve2p1PackedRhsSize(size_t N, size_t K) {
    if (K > (std::numeric_limits<uint32_t>::max)()) {
        return 0;
    }

    const auto& packer = kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve();
    const kai_matmul_pack_rhs_uker_config config{};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape{N, K};
    const auto stride = packer.get_rhs_packed_stride(&config, &shape);
    return packer.get_rhs_packed_size(&config, &shape, &stride);
}

size_t GetSve2p1NativePackedRhsSize(size_t N, size_t K) {
    const size_t packed_rhs_size = GetSve2p1PackedRhsSize(N, K);
    size_t total_size = 0;
    return packed_rhs_size != 0 &&
                   !MlasAddOverflowsSizeT(Sve2p1PackedRhsMetadataSize, packed_rhs_size, &total_size)
               ? total_size
               : 0;
}

bool ReadSve2p1PackedRhsMetadata(
    const void* packed_rhs,
    size_t& vector_length,
    const std::byte*& packed_rhs_data
) {
    std::memcpy(&vector_length, packed_rhs, sizeof(vector_length));
    if (vector_length == 0) {
        return false;
    }

    packed_rhs_data = static_cast<const std::byte*>(packed_rhs) + Sve2p1PackedRhsMetadataSize;
    return true;
}

size_t GetPackedRhsSize(KaiHalfGemmBackend backend, size_t N, size_t K) {
    switch (backend) {
        case KaiHalfGemmBackend::Sme:
            return kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(N, K);
        case KaiHalfGemmBackend::Sve2p1:
            return GetSve2p1PackedRhsSize(N, K);
        case KaiHalfGemmBackend::None:
            return 0;
    }
    return 0;
}

size_t PackRhs(
    KaiHalfGemmBackend backend,
    size_t N,
    size_t K,
    size_t ldb_bytes,
    const MLAS_FP16* rhs,
    const MLAS_FP16* bias,
    void* rhs_packed
) {
    if (backend == KaiHalfGemmBackend::Sve2p1) {
        const auto& packer = kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve();
        const kai_matmul_pack_rhs_uker_config config{};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape{N, K};
        const auto packed_stride = packer.get_rhs_packed_stride(&config, &packed_shape);

        kai_matmul_pack_rhs_uker_args args{};
        args.shape = {N, K};
        args.operand.rhs.ptr = rhs;
        args.operand.rhs.stride = {sizeof(MLAS_FP16), ldb_bytes};
        args.operand.rhs_packed.ptr = rhs_packed;
        args.operand.rhs_packed.stride = packed_stride;
        args.operand.bias_n.ptr = bias;
        packer.run(&config, &args);
        // The 16vs packer's N step equals the active SVE vector length in bytes.
        return packer.get_step(&config).n;
    }

    const auto& hgemm = GetKleidiAIHgemmUKernel();
    kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
        1, N, K, hgemm.ukernel.get_nr(), hgemm.ukernel.get_kr(), hgemm.ukernel.get_sr(), ldb_bytes,
        rhs, bias, nullptr, rhs_packed, 0, nullptr);
    return 0;
}

bool RunSve2p1HalfGemm(
    size_t M,
    size_t N,
    size_t K,
    const MLAS_FP16* lhs,
    size_t lhs_stride_bytes,
    const std::byte* rhs_packed,
    MLAS_FP16* dst,
    size_t dst_stride_bytes,
    MLAS_THREADPOOL* thread_pool,
    size_t expected_vector_length,
    const float* clamp_min,
    const float* clamp_max
) {
    // MlasHalfGemmBatch prevalidates native-packed metadata. Keep this local check
    // as defense in depth for this execution boundary.
    const auto& hgemm = kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot();
    const kai_matmul_uker_config config{};
    if (hgemm.get_step(&config).n != expected_vector_length) {
        return false;
    }

    const kai_matmul_uker_lhs_stride_args lhs_stride{lhs_stride_bytes};
    const kai_matmul_uker_dst_stride_args dst_stride{dst_stride_bytes};
    std::atomic<bool> vector_length_mismatch{false};

    constexpr size_t kernel_m = 6;
    const size_t m_tiles = M / kernel_m + (M % kernel_m != 0);
    const size_t n_tiles = N / expected_vector_length + (N % expected_vector_length != 0);
    size_t tile_count = 0;
    if (MlasMultiplyOverflowsSizeT(m_tiles, n_tiles, &tile_count)) {
        return false;
    }

    // TODO: Tune this task-count heuristic on real SVE2.1 hardware. QEMU does not model the
    // balance between kernel throughput and thread-pool scheduling overhead, so its timings
    // cannot determine when distributing M/N tiles across more workers improves latency.
    const size_t maximum_thread_count = static_cast<size_t>(MlasGetMaximumThreadCount(thread_pool));
    const double complexity = double(M) * double(N) * double(K);
    const double target_thread_count = complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY) + 1.0;
    const size_t task_count = std::min(
        tile_count,
        target_thread_count >= double(maximum_thread_count)
            ? maximum_thread_count
            : static_cast<size_t>(target_thread_count)
    );

    const auto run_task = [&](ptrdiff_t task_idx) {
        const auto step = hgemm.get_step(&config);
        const kai_matmul_uker_rhs_dim_args rhs_shape{N, K};
        const auto rhs_stride = hgemm.get_rhs_stride(&config, &rhs_shape);
        size_t tile_idx = 0;
        size_t tiles_remaining = 0;
        MlasPartitionWork(
            task_idx, static_cast<ptrdiff_t>(task_count), tile_count, &tile_idx, &tiles_remaining
        );

        while (tiles_remaining-- > 0) {
            const size_t start_m = tile_idx / n_tiles * kernel_m;
            const size_t start_n = tile_idx % n_tiles * step.n;
            const size_t tile_m = std::min(kernel_m, M - start_m);
            const size_t tile_n = std::min(step.n, N - start_n);
            const kai_matmul_uker_lhs_dim_args lhs_index{start_m, 0};
            const size_t lhs_offset = hgemm.get_lhs_offset(&config, &lhs_index, &lhs_stride);
            const kai_matmul_uker_rhs_dim_args rhs_index{start_n, 0};
            const size_t rhs_offset = hgemm.get_rhs_offset(&config, &rhs_index, &rhs_stride);
            const kai_matmul_uker_dst_dim_args dst_index{start_m, start_n};
            const size_t dst_offset = hgemm.get_dst_offset(&config, &dst_index, &dst_stride);

            kai_matmul_uker_args args{};
            args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
            args.shape = {tile_m, tile_n, K};
            args.operand.lhs.ptr = reinterpret_cast<const std::byte*>(lhs) + lhs_offset;
            args.operand.lhs.stride = lhs_stride;
            args.operand.rhs.ptr = rhs_packed + rhs_offset;
            args.operand.rhs.stride = rhs_stride;
            args.operand.dst.ptr = reinterpret_cast<std::byte*>(dst) + dst_offset;
            args.operand.dst.stride = dst_stride;
            args.activation.clamp.min_ptr = clamp_min;
            args.activation.clamp.max_ptr = clamp_max;
            hgemm.run(&config, &args);
            ++tile_idx;
        }
    };

    MlasTrySimpleParallel(thread_pool, static_cast<ptrdiff_t>(task_count), [&](ptrdiff_t task_idx) {
        if (vector_length_mismatch.load()) {
            return;
        }

        if (hgemm.get_step(&config).n != expected_vector_length) {
            vector_length_mismatch.store(true);
            return;
        }

        run_task(task_idx);
    });

    if (vector_length_mismatch.load()) {
        // Replay every tile on the validated caller so a rejected worker cannot leave partial output.
        for (size_t task_idx = 0; task_idx < task_count; ++task_idx) {
            run_task(static_cast<ptrdiff_t>(task_idx));
        }
    }

    return true;
}
}  // namespace

#if defined(MLAS_ENABLE_TEST_HOOKS) && defined(USE_KLEIDIAI)
const char*
ArmKleidiAI::GetKleidiAIHalfGemmKernelNameForTesting()
{
    switch (SelectKaiHalfGemmBackend()) {
        case KaiHalfGemmBackend::Sme:
            return GetKleidiAIHgemmUKernel().name;
        case KaiHalfGemmBackend::Sve2p1:
            return Sve2p1HalfGemmKernelName;
        case KaiHalfGemmBackend::None:
            return nullptr;
    }
    return nullptr;
}

size_t
ArmKleidiAI::GetKleidiAISve2p1HalfGemmNStepForTesting()
{
    const auto& hgemm = kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot();
    const kai_matmul_uker_config config{};
    return hgemm.get_step(&config).n;
}

#endif

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

    const auto backend = SelectKaiHalfGemmBackend();
    return backend == KaiHalfGemmBackend::Sve2p1
               ? GetSve2p1NativePackedRhsSize(N, K)
               : GetPackedRhsSize(backend, N, K);
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

    const auto backend = SelectKaiHalfGemmBackend();
    const size_t packed_rhs_size = backend == KaiHalfGemmBackend::Sve2p1
                                       ? GetSve2p1NativePackedRhsSize(N, K)
                                       : GetPackedRhsSize(backend, N, K);
    if (packed_rhs_size == 0) {
        return false;
    }

    std::vector<MLAS_FP16> zero_bias(N, MLAS_FP16::FromBits(0));

    size_t ldb_bytes = 0;
    if (MlasMultiplyOverflowsSizeT(ldb, sizeof(MLAS_FP16), &ldb_bytes)) {
        return false;
    }

    void* packed_rhs_data = PackedB;
    if (backend == KaiHalfGemmBackend::Sve2p1) {
        std::memset(PackedB, 0, Sve2p1PackedRhsMetadataSize);
        packed_rhs_data = static_cast<std::byte*>(PackedB) + Sve2p1PackedRhsMetadataSize;
    }

    const size_t vector_length = PackRhs(backend, N, K, ldb_bytes, B, zero_bias.data(), packed_rhs_data);
    if (backend == KaiHalfGemmBackend::Sve2p1) {
        std::memcpy(PackedB, &vector_length, sizeof(vector_length));
    }

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
    MLAS_THREADPOOL* ThreadPool
) {
    if (BatchN == 0 || M == 0 || N == 0) {
        return true;
    }
    if (K == 0) {
        return false;
    }
    if (DataParams == nullptr) {
        return false;
    }

    const auto backend = SelectKaiHalfGemmBackend();
    if (backend == KaiHalfGemmBackend::None) {
        return false;
    }

    ScopedKaiHalfTlsCleanup cleanup{g_kai_half_tls};

    size_t caller_vector_length = 0;
    if (backend == KaiHalfGemmBackend::Sve2p1) {
        const auto& hgemm = kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot();
        const kai_matmul_uker_config config{};
        caller_vector_length = hgemm.get_step(&config).n;
    }

    // MatMul currently supplies one backend-native packed RHS (BatchN == 1), but
    // this override accepts arbitrary MLAS batches. Validate every native-packed
    // RHS before any entry writes output so a later mismatch cannot partially modify C.
    bool needs_rhs_packing = false;
    for (size_t b = 0; b < BatchN; ++b) {
        const auto& data = DataParams[b];
        if (data.OutputProcessor != nullptr) {
            return false;
        }
        if (data.BIsBackendNativePacked && (data.ldb != 0 || data.Bias != nullptr)) {
            return false;
        }
        if (backend == KaiHalfGemmBackend::Sve2p1 && data.BIsBackendNativePacked) {
            size_t packed_vector_length = 0;
            const std::byte* packed_rhs = nullptr;
            if (!ReadSve2p1PackedRhsMetadata(data.B, packed_vector_length, packed_rhs) ||
                packed_vector_length != caller_vector_length) {
                return false;
            }
        }
        // Native-packed RHS is consumed directly below. Only allocate the
        // runtime RHS packing scratch when at least one batch entry needs it.
        needs_rhs_packing = needs_rhs_packing || !data.BIsBackendNativePacked;
    }

    const KaiF16HgemmKernel* sme_hgemm = nullptr;
    if (backend == KaiHalfGemmBackend::Sme) {
        sme_hgemm = &GetKleidiAIHgemmUKernel();
        KLEIDIAI_KERNEL_LOG(sme_hgemm->name);
    } else {
        KLEIDIAI_KERNEL_LOG(Sve2p1HalfGemmKernelName);
    }

    const size_t packed_rhs_size = GetPackedRhsSize(backend, N, K);
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
        size_t rhs_vector_length = 0;
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
            if (backend == KaiHalfGemmBackend::Sve2p1) {
                if (!ReadSve2p1PackedRhsMetadata(data.B, rhs_vector_length, rhs_packed)) {
                    return false;
                }
            } else {
                rhs_packed = reinterpret_cast<const std::byte*>(data.B);
            }
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

            rhs_vector_length = PackRhs(
                backend, N, K, ldb_bytes, rhs_base,
                data.Bias != nullptr ? data.Bias : g_kai_half_tls.bias_zero.data(), rhs_packed_buffer);
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

        if (backend == KaiHalfGemmBackend::Sve2p1) {
            if (!RunSve2p1HalfGemm(
                M, N, K, lhs_base, lda_bytes, rhs_packed, data.C, dst_stride_bytes, ThreadPool,
                rhs_vector_length, &clamp_min, &clamp_max)) {
                return false;
            }
            continue;
        }

        const auto& hgemm = *sme_hgemm;
        const size_t n_step = hgemm.ukernel.get_n_step();
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
