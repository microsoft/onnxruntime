//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <vector>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <arm_neon.h>

#include "mlas.h"

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

// Thread-local reusable buffers to reduce allocation overhead across tiles.
struct KaiTlsBuffers {
    std::vector<float> output_tile;
    std::vector<float> bias_zero;
    std::vector<std::byte> rhs_packed;
    std::vector<std::byte> lhs_packed;
    std::vector<float> gemv_lhs_row_tmp;
};
static thread_local KaiTlsBuffers g_kai_tls;

const KaiF32SgemmKernel& sgemm_gemm = GetKleidiAISGemmUKernel();
const KaiF32SgemvKernel& sgemm_gemv = GetKleidiAISGemvUKernel();

namespace {

const kai_matmul_uker_config kSme2MatmulConfig{};
const kai_matmul_pack_lhs_uker_config kSme2PackLhsConfig{};
const kai_matmul_pack_rhs_uker_config kSme2PackRhsConfig{};

const kai_matmul_uker_api kSme2Gemm4vsx1 =
    kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
const kai_matmul_pack_lhs_uker_api kSme2PackLhs4vsx1 =
    kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
const kai_matmul_pack_rhs_uker_api kSme2PackRhsKxN4vsx1 =
    kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
const kai_matmul_pack_rhs_uker_api kSme2PackRhsNxK4vsx1 =
    kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();

constexpr float kNoActivationClampMin = -std::numeric_limits<float>::infinity();
constexpr float kNoActivationClampMax = std::numeric_limits<float>::infinity();

const kai_matmul_pack_rhs_uker_api* GetSme2PackRhs(CBLAS_TRANSPOSE TransB) {
    switch (TransB) {
        case CblasNoTrans:
            return &kSme2PackRhsKxN4vsx1;
        case CblasTrans:
            return &kSme2PackRhsNxK4vsx1;
        default:
            return nullptr;
    }
}

size_t GetSme2PackedBSize(CBLAS_TRANSPOSE TransB, size_t N, size_t K) {
    const auto* pack_rhs = GetSme2PackRhs(TransB);
    if (pack_rhs == nullptr || N == 0 || K == 0) {
        return 0;
    }

    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape{N, K};
    const auto stride = pack_rhs->get_rhs_packed_stride(&kSme2PackRhsConfig, &shape);
    return pack_rhs->get_rhs_packed_size(&kSme2PackRhsConfig, &shape, &stride);
}

bool PackSme2B(CBLAS_TRANSPOSE TransB,
               size_t N,
               size_t K,
               const float* B,
               size_t ldb,
               const float* Bias,
               void* PackedB) {
    const auto* pack_rhs = GetSme2PackRhs(TransB);
    if (pack_rhs == nullptr || N == 0 || K == 0) {
        return false;
    }

    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape{N, K};
    const auto packed_stride = pack_rhs->get_rhs_packed_stride(&kSme2PackRhsConfig, &shape);

    kai_matmul_pack_rhs_uker_args args{};
    args.shape = {N, K};
    args.operand.rhs.ptr = B;
    args.operand.rhs_packed.ptr = PackedB;
    args.operand.rhs_packed.stride = packed_stride;
    args.operand.bias_n.ptr = Bias;

    if (TransB == CblasNoTrans) {
        args.operand.rhs.stride.n = sizeof(float);
        args.operand.rhs.stride.k = ldb * sizeof(float);
    } else {
        args.operand.rhs.stride.n = ldb * sizeof(float);
        args.operand.rhs.stride.k = sizeof(float);
    }

    pack_rhs->run(&kSme2PackRhsConfig, &args);
    return true;
}

size_t GetSme2PackedASize(size_t M, size_t K) {
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape{M, K};
    const auto stride = kSme2PackLhs4vsx1.get_lhs_packed_stride(&kSme2PackLhsConfig, &shape);
    return kSme2PackLhs4vsx1.get_lhs_packed_size(&kSme2PackLhsConfig, &shape, &stride);
}

void PackSme2A(size_t M, size_t K, const float* A, size_t lda, void* PackedA) {
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape{M, K};
    const auto packed_stride = kSme2PackLhs4vsx1.get_lhs_packed_stride(&kSme2PackLhsConfig, &shape);

    kai_matmul_pack_lhs_uker_args args{};
    args.shape = {M, K};
    args.operand.lhs.ptr = A;
    args.operand.lhs.stride.m = lda * sizeof(float);
    args.operand.lhs_packed.ptr = PackedA;
    args.operand.lhs_packed.stride = packed_stride;

    kSme2PackLhs4vsx1.run(&kSme2PackLhsConfig, &args);
}

bool MlasGemmBatchSme2(CBLAS_TRANSPOSE TransB,
                       size_t M,
                       size_t N,
                       size_t K,
                       const MLAS_SGEMM_DATA_PARAMS* Data,
                       size_t BatchSize,
                       MLAS_THREADPOOL* ThreadPool);

}  // namespace

// Avoid vector setup overhead on tiny outputs.
constexpr size_t kAlphaBetaNeonMinElements = 32;


// Helpers for GEMV
/*++
Routine Description:
    Apply alpha/beta scaling to a 1-D vector with arbitrary destination stride.

Arguments:
    src          - Pointer to the temporary A*B results (length L).
    num_elements - Number of elements.
    alpha        - Scale for the computed product (A*B).
    beta         - Scale for the existing C values.
    dst          - Pointer to the destination in C.
    dst_stride   - Stride, in elements, between successive outputs in C.
    allow_memcpy - If true, allows memcpy path when alpha==1, beta==0, and dst_stride==1.

Notes:
    Uses a memcpy path when alpha==1, beta==0, allow_memcpy is true, and dst_stride==1.
--*/
static inline void ApplyAlphaBetaStrided(const float* src, size_t num_elements, float alpha, float beta, float* dst, size_t dst_stride, bool allow_memcpy) {
    if (alpha == 1.0f && beta == 0.0f && allow_memcpy && dst_stride == 1) {
        std::memcpy(dst, src, num_elements * sizeof(float));
        return;
    }

    // Contiguous-only vectorized path with strict correctness guards.
    if (dst_stride == 1 && num_elements >= kAlphaBetaNeonMinElements) {
        size_t i = 0;
        if (alpha == 1.0f && beta == 0.0f) {
            for (; i + 4 <= num_elements; i += 4) {
                vst1q_f32(dst + i, vld1q_f32(src + i));
            }
        } else if (alpha == 1.0f) {
            const float32x4_t vbeta = vdupq_n_f32(beta);
            for (; i + 4 <= num_elements; i += 4) {
                const float32x4_t vab = vld1q_f32(src + i);
                const float32x4_t vc = vld1q_f32(dst + i);
                vst1q_f32(dst + i, vmlaq_f32(vab, vbeta, vc));
            }
        } else if (beta == 0.0f) {
            const float32x4_t valpha = vdupq_n_f32(alpha);
            for (; i + 4 <= num_elements; i += 4) {
                const float32x4_t vab = vld1q_f32(src + i);
                vst1q_f32(dst + i, vmulq_f32(valpha, vab));
            }
        } else {
            const float32x4_t valpha = vdupq_n_f32(alpha);
            const float32x4_t vbeta = vdupq_n_f32(beta);
            for (; i + 4 <= num_elements; i += 4) {
                const float32x4_t vab = vld1q_f32(src + i);
                const float32x4_t vc = vld1q_f32(dst + i);
                vst1q_f32(dst + i, vmlaq_f32(vmulq_f32(valpha, vab), vbeta, vc));
            }
        }

        for (; i < num_elements; ++i) {
            const float ab = src[i];
            const float c_orig = dst[i];
            if (alpha == 1.0f && beta == 0.0f) {
                dst[i] = ab;
            } else if (alpha == 1.0f) {
                dst[i] = ab + beta * c_orig;
            } else if (beta == 0.0f) {
                dst[i] = alpha * ab;
            } else {
                dst[i] = alpha * ab + beta * c_orig;
            }
        }
        return;
    }

    for (size_t i = 0; i < num_elements; ++i) {
        const float ab = src[i];
        float& d = dst[i * dst_stride];
        const float c_orig = d;
        if (alpha == 1.0f && beta == 0.0f) {
            d = ab;
        } else if (alpha == 1.0f) {
            d = ab + beta * c_orig;
        } else if (beta == 0.0f) {
            d = alpha * ab;
        } else {
            d = alpha * ab + beta * c_orig;
        }
    }
}

/*++
Routine Description:
    Apply alpha/beta scaling to a 2-D tile (rows x cols).

Arguments:
    src   - Pointer to the temporary A*B results (row-major, rows x cols).
    rows  - Number of rows in the tile.
    cols  - Number of columns in the tile.
    alpha - Scale for the computed product (A*B).
    beta  - Scale for the existing C values.
    dst   - Pointer to the destination tile in C (row-major with leading dimension ldc).
    ldc   - Leading dimension of C (in elements).

Notes:
    For contiguous destination tiles (ldc==cols), flattens (rows*cols) and routes
    through ApplyAlphaBetaStrided to enable contiguous SIMD and memcpy fast paths.
    For non-contiguous destination tiles, applies per-row scaling via
    ApplyAlphaBetaStrided.
--*/
static inline void ApplyAlphaBeta2D(const float* src, size_t rows, size_t cols,
                                    float alpha, float beta,
                                    float* dst, size_t ldc) {
    if (rows == 0 || cols == 0) {
        return;
    }

    if (ldc == cols) {
        // Contiguous destination: flatten so we can hit the contiguous SIMD path.
        ApplyAlphaBetaStrided(src, rows * cols, alpha, beta, dst, 1, /*allow_memcpy*/ true);
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        const float* src_row = src + i * cols;
        float* dst_row = dst + i * ldc;
        ApplyAlphaBetaStrided(src_row, cols, alpha, beta, dst_row, 1, /*allow_memcpy*/ false);
    }
}

static inline void ApplyBetaToC(float* C, size_t ldc, size_t M, size_t N, float beta) {
    if (beta == 0.0f) {
        for (size_t i = 0; i < M; ++i) {
            std::fill_n(C + i * ldc, N, 0.0f);
        }
        return;
    }
    if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }
}

namespace {

bool MlasGemmBatchSme2(CBLAS_TRANSPOSE TransB,
                       size_t M,
                       size_t N,
                       size_t K,
                       const MLAS_SGEMM_DATA_PARAMS* Data,
                       size_t BatchSize,
                       MLAS_THREADPOOL* ThreadPool) {
    bool has_unpacked_b = false;
    for (size_t batch = 0; batch < BatchSize; ++batch) {
        has_unpacked_b |= !Data[batch].BIsPacked;
    }

    // Both source orientations produce the same p4vsx1 packed layout. Packed
    // MLAS calls do not preserve the source TransB value, so use one canonical
    // packer contract when every B is already packed. If any B is raw, retain
    // TransB so those entries are packed from the correct source orientation.
    if (!has_unpacked_b) {
        TransB = CblasNoTrans;
    }

    const auto* pack_rhs = GetSme2PackRhs(TransB);
    if (pack_rhs == nullptr) {
        return false;
    }

    const auto step = kSme2Gemm4vsx1.get_step(&kSme2MatmulConfig);
    const auto lhs_pack_step = kSme2PackLhs4vsx1.get_step(&kSme2PackLhsConfig);
    const auto rhs_pack_step = pack_rhs->get_step(&kSme2PackRhsConfig);
    if (step.m == 0 || step.n == 0 || lhs_pack_step.m != step.m || rhs_pack_step.n != step.n ||
        lhs_pack_step.k != 0 || rhs_pack_step.k != 0) {
        return false;
    }

    const kai_matmul_uker_lhs_dim_args lhs_shape{M, K};
    const kai_matmul_uker_rhs_dim_args rhs_shape{N, K};
    const auto lhs_stride = kSme2Gemm4vsx1.get_lhs_stride(&kSme2MatmulConfig, &lhs_shape);
    const auto rhs_stride = kSme2Gemm4vsx1.get_rhs_stride(&kSme2MatmulConfig, &rhs_shape);

    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape{M, K};
    const auto lhs_packed_stride =
        kSme2PackLhs4vsx1.get_lhs_packed_stride(&kSme2PackLhsConfig, &lhs_packed_shape);
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape{N, K};
    const auto rhs_packed_stride =
        pack_rhs->get_rhs_packed_stride(&kSme2PackRhsConfig, &rhs_packed_shape);
    if (lhs_stride.m != lhs_packed_stride.m || rhs_stride.n != rhs_packed_stride.n) {
        return false;
    }

    const size_t lhs_packed_size = GetSme2PackedASize(M, K);
    const size_t rhs_packed_size = GetSme2PackedBSize(TransB, N, K);
    if (lhs_packed_size == 0 || rhs_packed_size == 0) {
        return false;
    }

    size_t lhs_buffer_size = 0;
    size_t rhs_buffer_size = 0;
    if (MlasMultiplyOverflowsSizeT(lhs_packed_size, BatchSize, &lhs_buffer_size) ||
        (has_unpacked_b && MlasMultiplyOverflowsSizeT(rhs_packed_size, BatchSize, &rhs_buffer_size))) {
        return false;
    }

    size_t packing_iterations = BatchSize;
    if (has_unpacked_b && MlasMultiplyOverflowsSizeT(BatchSize, size_t{2}, &packing_iterations)) {
        return false;
    }
    if (packing_iterations > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return false;
    }

    size_t m_step = step.m;
    size_t n_step = step.n;
    std::array<size_t, 3> dim{
        BatchSize,
        MlasDivRoundup(M, m_step),
        MlasDivRoundup(N, n_step)};

    size_t initial_mn_tiles = 0;
    size_t initial_tile_count = 0;
    if (MlasMultiplyOverflowsSizeT(dim[1], dim[2], &initial_mn_tiles) ||
        MlasMultiplyOverflowsSizeT(dim[0], initial_mn_tiles, &initial_tile_count)) {
        return false;
    }

    const size_t maximum_thread_count =
        std::max<size_t>(1, static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)));
    const size_t required_tiles = std::min(maximum_thread_count, initial_tile_count);

    size_t required_m_tiles = 0;
    size_t required_n_tiles = 0;
    if (required_tiles == 0 ||
        MlasMultiplyOverflowsSizeT(required_tiles, dim[1], &required_m_tiles) ||
        MlasMultiplyOverflowsSizeT(required_tiles, dim[2], &required_n_tiles)) {
        return false;
    }

    dim[1] = MlasDivRoundup(required_m_tiles, initial_mn_tiles);
    size_t n_tile_denominator = 0;
    if (MlasMultiplyOverflowsSizeT(dim[1], dim[2], &n_tile_denominator) || n_tile_denominator == 0) {
        return false;
    }
    dim[2] = MlasDivRoundup(required_n_tiles, n_tile_denominator);
    if (dim[1] == 0 || dim[2] == 0) {
        return false;
    }

    const size_t m_step_scale = MlasDivRoundup(MlasDivRoundup(M, dim[1]), m_step);
    const size_t n_step_scale = MlasDivRoundup(MlasDivRoundup(N, dim[2]), n_step);
    if (MlasMultiplyOverflowsSizeT(m_step, m_step_scale, &m_step) ||
        MlasMultiplyOverflowsSizeT(n_step, n_step_scale, &n_step)) {
        return false;
    }

    dim[1] = MlasDivRoundup(M, m_step);
    dim[2] = MlasDivRoundup(N, n_step);

    size_t mn_tiles = 0;
    size_t total_tiles = 0;
    if (MlasMultiplyOverflowsSizeT(dim[1], dim[2], &mn_tiles) ||
        MlasMultiplyOverflowsSizeT(dim[0], mn_tiles, &total_tiles) ||
        total_tiles > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        return false;
    }

    size_t max_tile_elements = 0;
    if (MlasMultiplyOverflowsSizeT(m_step, n_step, &max_tile_elements)) {
        return false;
    }

    g_kai_tls.lhs_packed.resize(lhs_buffer_size);
    if (has_unpacked_b) {
        g_kai_tls.rhs_packed.resize(rhs_buffer_size);
        g_kai_tls.bias_zero.assign(N, 0.0f);
    }

    std::byte* const lhs_packed_data = g_kai_tls.lhs_packed.data();
    std::byte* const rhs_packed_data = has_unpacked_b ? g_kai_tls.rhs_packed.data() : nullptr;
    const float* const bias_zero = has_unpacked_b ? g_kai_tls.bias_zero.data() : nullptr;

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(packing_iterations), [&](ptrdiff_t tid) {
        const size_t batch_idx = has_unpacked_b ? static_cast<size_t>(tid >> 1) : static_cast<size_t>(tid);
        if (!has_unpacked_b || (tid & 0x1)) {
            PackSme2A(M,
                      K,
                      Data[batch_idx].A,
                      Data[batch_idx].lda,
                      lhs_packed_data + lhs_packed_size * batch_idx);
        } else if (!Data[batch_idx].BIsPacked) {
            PackSme2B(TransB,
                      N,
                      K,
                      reinterpret_cast<const float*>(Data[batch_idx].B),
                      Data[batch_idx].ldb,
                      bias_zero,
                      rhs_packed_data + rhs_packed_size * batch_idx);
        }
    });

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(total_tiles), [=](ptrdiff_t tid) {
        const size_t work_idx = static_cast<size_t>(tid);
        const size_t batch_idx = work_idx / mn_tiles;
        const size_t tile_idx = work_idx % mn_tiles;
        const size_t m_idx = tile_idx / dim[2];
        const size_t n_idx = tile_idx % dim[2];

        const size_t start_m = m_idx * m_step;
        const size_t start_n = n_idx * n_step;
        const size_t tile_m = std::min(m_step, M - start_m);
        const size_t tile_n = std::min(n_step, N - start_n);

        const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_index{start_m, 0};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_index{start_n, 0};
        const size_t lhs_offset = kSme2PackLhs4vsx1.get_lhs_packed_offset(
            &kSme2PackLhsConfig, &lhs_index, &lhs_packed_stride);
        const size_t rhs_offset = pack_rhs->get_rhs_packed_offset(
            &kSme2PackRhsConfig, &rhs_index, &rhs_packed_stride);

        const auto* lhs_tile = lhs_packed_data + lhs_packed_size * batch_idx + lhs_offset;
        const std::byte* rhs_base = Data[batch_idx].BIsPacked
            ? reinterpret_cast<const std::byte*>(Data[batch_idx].B)
            : rhs_packed_data + rhs_packed_size * batch_idx;
        const auto* rhs_tile = rhs_base + rhs_offset;
        float* const dst_tile = Data[batch_idx].C + start_m * Data[batch_idx].ldc + start_n;

        const float alpha = Data[batch_idx].alpha;
        const float beta = Data[batch_idx].beta;
        const bool direct_to_c = alpha == 1.0f && beta == 0.0f;

        float* output = dst_tile;
        size_t output_stride = Data[batch_idx].ldc * sizeof(float);
        if (!direct_to_c) {
            g_kai_tls.output_tile.resize(tile_m * tile_n);
            output = g_kai_tls.output_tile.data();
            output_stride = tile_n * sizeof(float);
        }

        kai_matmul_uker_args args{};
        args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
        args.shape = {tile_m, tile_n, K};
        args.operand.lhs.ptr = lhs_tile;
        args.operand.lhs.stride = lhs_stride;
        args.operand.rhs.ptr = rhs_tile;
        args.operand.rhs.stride = rhs_stride;
        args.operand.dst.ptr = output;
        args.operand.dst.stride.m = output_stride;
        args.activation.clamp.min_ptr = &kNoActivationClampMin;
        args.activation.clamp.max_ptr = &kNoActivationClampMax;

        KLEIDIAI_KERNEL_LOG("kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa"
                            << " M=" << tile_m << " N=" << tile_n << " K=" << K);
        kSme2Gemm4vsx1.run(&kSme2MatmulConfig, &args);

        if (!direct_to_c) {
            ApplyAlphaBeta2D(output, tile_m, tile_n, alpha, beta, dst_tile, Data[batch_idx].ldc);
        }
    });

    return true;
}

}  // namespace

/*++
Routine Description:
    Execute GEMV using the retained SME 1xN microkernel for degenerate GEMM shapes:
    - M == 1 (row-vector times matrix)
    - N == 1 (matrix times column-vector)

N == 1 mapping (y = A(MxK) * b(Kx1)):
    The 1xN microkernel computes a single LHS row against multiple RHS columns.
    To reuse it for N == 1, we present A as the "RHS" by transpose-packing it
    so that each of A's M rows becomes a "column" for the kernel:
      - rhsBase := A, rhsShape := M, ldl := lda, tb := CblasTrans
      - lhsBase := B (the vector b), length K
    The kernel expects the LHS vector to be a contiguous K-length row:
      - If TransB == CblasNoTrans, b is stored as a Kx1 column with stride ldb.
        We gather it into a thread-local contiguous buffer when ldb != 1.
      - If TransB == CblasTrans, b is a 1xK row and is already contiguous.

Unsupported:
    When N == 1 and Data->BIsPacked is true (except M == N == 1), this path is
    disabled because we need to pack A (as RHS) and pass B as an unpacked vector.

Post-processing:
    The kernel produces M outputs into a temporary buffer. We apply alpha/beta
    and write to C using ldc as the destination stride.

Return Value:
    true  - A GEMV path was executed (M == 1 or N == 1).
    false - Fall back to the general GEMM path.
--*/

bool
MLASCALL
ArmKleidiAI::MlasGemvBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize
) {
        // Only two paths: M-path (M == 1, also covers M == N == 1) or N-path (N == 1).
        if (M != 1 && N != 1) {
            return false;
        }

        const bool m_path = (M == 1);

        // We cannot support cases where N == 1 and B is already packed.
        // When both are 1, we route through the M-path, so this naturally doesn't trigger.
        if (!m_path && Data->BIsPacked) {
            return false;
        }

        // Decide RHS and transposition once based on the path
        CBLAS_TRANSPOSE tb = m_path ? TransB : CblasTrans;
        size_t rhs_shape = m_path ? N : M;

        for (size_t b = 0; b < BatchSize; ++b) {

            size_t rhs_ld    = m_path ? Data[b].ldb : Data[b].lda;
            // LHS is the vector row we feed to the GEMV microkernel
            // - M-path: LHS is A, stride = lda
            // - N-path: LHS is B, stride = ldb
            size_t lhs_ld = m_path ? Data[b].lda : Data[b].ldb;

            const float* rhs_base = m_path ? static_cast<const float*>(Data[b].B)
                                           : static_cast<const float*>(Data[b].A);
            const float* lhs_base = m_path ? static_cast<const float*>(Data[b].A)
                                           : static_cast<const float*>(Data[b].B);

            // Prepare packed RHS if needed
            const void* rhs_packed_ptr = nullptr;

            // The if branch can only be taken in cases where we are dealing with M == 1
            // We previously reject any prepacked B where N == 1
            // In cases where N == 1 we Pack A Matrix as the RHS using tb = CBlasTrans
            // After which the rhs_packed_ptr points to Packed A not B
            // rhs_packed_ptr = Data[b].B only when M == 1
            if (Data[b].BIsPacked) {
                rhs_packed_ptr = Data[b].B;
            } else {
                const size_t rhs_size = ArmKleidiAI::MlasGemmPackBSize(TransA, tb, rhs_shape, K);
                if (rhs_size == 0) {
                    return false;
                }
                g_kai_tls.rhs_packed.resize(rhs_size);

                ArmKleidiAI::MlasGemmPackB(
                    TransA, tb, rhs_shape, K,
                    rhs_base,
                    rhs_ld,
                    g_kai_tls.rhs_packed.data());
                rhs_packed_ptr = g_kai_tls.rhs_packed.data();
            }
            // Ensure LHS is a contiguous K-length row for the GEMV microkernel.
            // Compute once whether we need to gather based on which side is LHS.
            const bool needs_gather = m_path ? (TransA == CblasTrans) : (TransB == CblasNoTrans);
            if (needs_gather) {
                g_kai_tls.gemv_lhs_row_tmp.resize(K);
                for (size_t k = 0; k < K; ++k) {
                    g_kai_tls.gemv_lhs_row_tmp[k] = lhs_base[k * lhs_ld];
                }
                lhs_base = g_kai_tls.gemv_lhs_row_tmp.data();
            }

            // Temporary buffer for output row
            g_kai_tls.output_tile.resize(rhs_shape);

            // Run specialized 1xN-by-K kernel
            sgemm_gemv.ukernel.run_matmul(
                1,                                          // Value of 1 for M == 1 and this value represents N when N == 1 case
                rhs_shape,                                  // Value of N for M == 1 and this value is M when N == 1
                K,                                          // K
                lhs_base,                                   // lhs
                K * sizeof(float),                          // lhs stride (bytes)
                rhs_packed_ptr,                             // packed rhs
                g_kai_tls.output_tile.data(),               // output
                rhs_shape * sizeof(float),                  // dst row stride (bytes)
                sizeof(float),                              // dst col stride (bytes)
                -std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()
            );
            // Apply alpha/beta to destination C row
            bool allowMemCopy = m_path ? (Data[b].ldc == N) : (Data[b].ldc == 1);
            size_t destStride = m_path ? 1 : Data[b].ldc;
            ApplyAlphaBetaStrided(g_kai_tls.output_tile.data(), rhs_shape, Data[b].alpha, Data[b].beta, Data[b].C, destStride, allowMemCopy);
        }
        return true;
}

size_t
MLASCALL
ArmKleidiAI::MlasGemmPackBSize(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
)
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    TransA - Supplies the transpose operation on A matrix

    TransB - Supplies the transpose operation on B matrix

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    if (TransA != CblasNoTrans ||  N == 0  || K == 0) {
        KLEIDIAI_DEBUG_LOG("MlasGemmPackBSize returning 0 size. N=" << N << " K=" << K);
        return 0;
    }

    if (ArmKleidiAI::UseSME2) {
        return GetSme2PackedBSize(TransB, N, K);
    }

    //
    // Compute the number of bytes required to hold the packed buffer.
    //
    size_t bytes = 0;
    switch (TransB) {
        case CblasNoTrans:
            bytes = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
            break;
        case CblasTrans:
            bytes = kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(N, K);
            break;
        default:
            KLEIDIAI_DEBUG_LOG("MlasGemmPackBSize TransB is neither CblasNoTrans nor CblasTrans, returning 0.");
            return 0;
    }

    return bytes;
}

bool
MLASCALL
ArmKleidiAI::MlasGemmPackB(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
)
/*++

Routine Description:

    This routine packs the contents of matrix B to the destination buffer. The
    destination buffer should be sized based on MlasGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    Returns true if the packing operation was handled by KleidiAI.
    Returns false if the configuration requires a fallback to the default MLAS implementation.

--*/
{
    if (N == 0 || K == 0) {
        return false;
    }

    if (TransA == CblasNoTrans) {

        if (ArmKleidiAI::UseSME2) {
            const auto* pack_rhs = GetSme2PackRhs(TransB);
            if (pack_rhs == nullptr) {
                return false;
            }

            g_kai_tls.bias_zero.assign(N, 0.0f);
            return PackSme2B(
                TransB, N, K, B, ldb, g_kai_tls.bias_zero.data(), PackedB);
        }

        const size_t nr = sgemm_gemm.ukernel.get_nr();
        const size_t kr = sgemm_gemm.ukernel.get_kr();
        const size_t sr = sgemm_gemm.ukernel.get_sr();

        // Ensure size and zero the used span.
        g_kai_tls.bias_zero.resize(N, 0.0f);

        switch (TransB) {
            case CblasNoTrans:
            KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme Groups=1"
                                    << " N="<< N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr << " rhs_stride_row=" << ldb * sizeof(float));
                kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, g_kai_tls.bias_zero.data(), nullptr, PackedB, 0, nullptr);
                break;
            case CblasTrans:
            KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme Groups=1"
                                    << " N="<< N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr << " rhs_stride_row=" << ldb * sizeof(float));
                kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(1, N, K, nr, kr, sr, ldb * sizeof(float), B, g_kai_tls.bias_zero.data(), nullptr, PackedB, 0, nullptr);
                break;
            default:
            KLEIDIAI_DEBUG_LOG("MlasGemmPackB TransB is neither CblasNoTrans nor CblasTrans, falling back to MLAS.");
                return false;
        }
        return true;
    }
    else{
        KLEIDIAI_DEBUG_LOG("MlasGemmPackB TransA is CblasTrans, falling back to MLAS.");
        return false;
    }
}

bool
MLASCALL
ArmKleidiAI::MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine performs a batched matrix multiplication (GEMM or GemV) operation using KleidiAI kernels.
    It handles both packed and unpacked inputs and manages tiling and kernel selection depending on
    SME2 availability. If packing is needed, it prepares the required buffers and invokes the
    appropriate left-hand side (LHS) and right-hand side (RHS) pack functions.

    The function also applies alpha and beta scaling to the result, supports efficient memcpy
    paths where possible, and dispatches tile-level GEMM work using multithreading.

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and rows of matrix B.

    Data - Supplies a pointer to the MLAS_SGEMM_DATA_PARAMS array containing per-batch input/output pointers and parameters.

    BatchSize - Supplies the number of independent GEMM computations to perform in the batch.

    ThreadPool - Supplies the thread pool to parallelize computation across batches and tiles.

Return Value:

    Returns true if the GEMM operation was handled by KleidiAI.
    Returns false if the configuration requires a fallback to the default MLAS implementation.

--*/
{
    if (M == 0 || N == 0 || BatchSize == 0) {
        return true;
    }

    if (K == 0) {
        for (size_t batch = 0; batch < BatchSize; ++batch) {
            ApplyBetaToC(Data[batch].C, Data[batch].ldc, M, N, Data[batch].beta);
        }
        return true;
    }

    bool all_alpha_zero = true;
    for (size_t batch = 0; batch < BatchSize; ++batch) {
        if (Data[batch].alpha != 0.0f) {
            all_alpha_zero = false;
            break;
        }
    }

    if (all_alpha_zero) {
        for (size_t batch = 0; batch < BatchSize; ++batch) {
            ApplyBetaToC(Data[batch].C, Data[batch].ldc, M, N, Data[batch].beta);
        }
        return true;
    }

    if (ArmKleidiAI::UseSME2) {
        if (TransA != CblasNoTrans) {
            return false;
        }

        bool has_packed_b = false;
        for (size_t batch = 0; batch < BatchSize; ++batch) {
            has_packed_b |= Data[batch].BIsPacked;
        }

        const bool handled = MlasGemmBatchSme2(TransB, M, N, K, Data, BatchSize, ThreadPool);
        if (!handled && has_packed_b) {
            // Public SME2 packing uses p4vsx1, which the generic MLAS fallback
            // and the retained SME implementation cannot consume.
            MLAS_THROW_EX(std::runtime_error,
                          "KleidiAI SME2 cannot fall back with an SME2-packed SGEMM B.");
        }
        return handled;
    }

    // Attempt GEMV (M==1 or N==1)
    if (M == 1 || N == 1)
    {
        // TODO: Investigate passing threadpool and multithreading of gemv op
        if (ArmKleidiAI::MlasGemvBatch(TransA, TransB, M, N, K, Data, BatchSize)) {
            return true;
        }
    }

    size_t m_step = sgemm_gemm.ukernel.get_m_step();
    size_t n_step = sgemm_gemm.ukernel.get_n_step();

    if ((M < m_step || N < n_step) && !Data->BIsPacked) {
        // Fallback to MLAS
        return false;
    }

    const size_t mr = sgemm_gemm.ukernel.get_mr();
    const size_t kr = sgemm_gemm.ukernel.get_kr();
    const size_t sr = sgemm_gemm.ukernel.get_sr();

    size_t LhsPackedStride = 0;
    std::byte* LhsPackedData = nullptr;

    LhsPackedStride = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);

    size_t lhs_resize = 0;
    if(MlasMultiplyOverflowsSizeT(LhsPackedStride, BatchSize, &lhs_resize))
    {
        // size_t wraparound detected for LhsPackedStride, fallback to MLAS
        return false;
    }

    g_kai_tls.lhs_packed.resize(lhs_resize);
    LhsPackedData = g_kai_tls.lhs_packed.data();

    // RHS packed buffer: use TLS reusable vector to minimize allocations
    size_t RhsPackedStride = 0;
    std::byte* RhsPackedData = nullptr;

    // It is assumed all B batches require packing or not
    if (Data[0].BIsPacked) {
        // We have already decided the matmul variant we are using, before having values for M,N,K
        MlasTrySimpleParallel(ThreadPool, BatchSize, [&](ptrdiff_t batch_idx) {
            std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);
            KLEIDIAI_KERNEL_LOG("kai_run_lhs_pack_f32p2vlx1_f32_sme"
                                    << " M=" << M << " K=" << K << " mr=" << mr << " kr=" << kr << " sr=" << sr);
            kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);
        });
    } else {
        // Multithread pack lhs and rhs
        RhsPackedStride = ArmKleidiAI::MlasGemmPackBSize(TransA, TransB, N, K);
        size_t rhs_resize = 0;
        if (MlasMultiplyOverflowsSizeT(RhsPackedStride, BatchSize, &rhs_resize))
        {
            // size_t wraparound detected for RhsPackedStride, fallback to MLAS
            return false;
        }

        g_kai_tls.rhs_packed.resize(rhs_resize);
        RhsPackedData = g_kai_tls.rhs_packed.data();

        MlasTrySimpleParallel(ThreadPool, BatchSize * 2, [&](ptrdiff_t batch_idx) {
            if (batch_idx & 0x1) {
                batch_idx >>= 1;
                std::byte* LhsPackedPtr = &(LhsPackedData[LhsPackedStride * batch_idx]);
                kai_run_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr, 0, Data[batch_idx].A, Data[batch_idx].lda * sizeof(float), LhsPackedPtr);
            } else {
                batch_idx >>= 1;
                std::byte* RhsPackedPtr = &(RhsPackedData[RhsPackedStride * batch_idx]);
                ArmKleidiAI::MlasGemmPackB(TransA, TransB, N, K,
                                           reinterpret_cast<const float*>(Data[batch_idx].B),
                                           Data[batch_idx].ldb, RhsPackedPtr);
            }
        });
    }

    // tile iteration dimensions
    std::array<size_t, 3> dim;
    dim[0] = BatchSize;                  // B
    dim[1] = MlasDivRoundup(M, m_step);  // M
    dim[2] = MlasDivRoundup(N, n_step);  // N

    // Minimize the kernel call count for the number of available threads
    auto RequiredTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0] * dim[1] * dim[2]);

    // scale required tiles over available tile processors
    dim[1] = MlasDivRoundup(RequiredTiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(RequiredTiles * dim[2], dim[1] * dim[2]);

    // compute new step sizes
    m_step *= MlasDivRoundup(MlasDivRoundup(M, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(N, dim[2]), n_step);

    // update tile iterations
    dim[1] = MlasDivRoundup(M, m_step);
    dim[2] = MlasDivRoundup(N, n_step);

    // Pre-check maximum tile size to avoid per-iteration overflow inside the parallel loop.
    // Any TileSizeM/TileSizeN used below will be <= m_step/n_step respectively.
    size_t max_tile_elems = 0;
    if (MlasMultiplyOverflowsSizeT(m_step, n_step, &max_tile_elems)) {
        // size_t wraparound detected for tile size, fallback to MLAS
        return false;
    }

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [=](ptrdiff_t tid) {
        // compute B,M,N index from iteration index
        ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
        ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
        ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

        // Get rhs tile, B
        const size_t rhs_packed_offset = sgemm_gemm.ukernel.get_rhs_packed_offset(NIdx * n_step, K);

        const std::byte* B_base = Data[0].BIsPacked
            ? reinterpret_cast<const std::byte*>(Data[BIdx].B)
            : (RhsPackedData + RhsPackedStride * BIdx);
        auto BTile = reinterpret_cast<const void*>(B_base + rhs_packed_offset);

        // Get lhs tile, A
        const size_t lhs_packed_offset = sgemm_gemm.ukernel.get_lhs_packed_offset(MIdx * m_step, K);

        const std::byte* A_base = LhsPackedData + LhsPackedStride * BIdx;
        auto ATile = reinterpret_cast<const float*>(A_base + lhs_packed_offset);

        auto TileSizeM = (MIdx + 1) * m_step > M ? (M - MIdx * m_step) : m_step;
        auto TileSizeN = (NIdx + 1) * n_step > N ? (N - NIdx * n_step) : n_step;

        // Get result tile, C
        auto CTile = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(Data[BIdx].C) +
            MIdx * m_step * Data[BIdx].ldc * sizeof(float) +
            NIdx * n_step * sizeof(float)
        );

        // Final output tile pointer
        float* dst_tile = reinterpret_cast<float*>(CTile);

        const float alpha = Data[BIdx].alpha;
        const float beta = Data[BIdx].beta;
        const size_t ldc = Data[BIdx].ldc;

        // Select output destination and strides once, then run_matmul exactly once.
        const bool direct_to_c = (
            alpha == 1.0f &&
            beta == 0.0f);

        float* out_tile = nullptr;
        size_t out_row_stride_bytes = 0;

        if (direct_to_c) {
            out_tile = dst_tile;
            out_row_stride_bytes = ldc * sizeof(float);
        } else {
            // Compute into a temporary buffer for raw A*B result (TLS reusable buffer)
            const size_t tile_elems = TileSizeM * TileSizeN;
            g_kai_tls.output_tile.resize(tile_elems);
            out_tile = g_kai_tls.output_tile.data();
            out_row_stride_bytes = TileSizeN * sizeof(float);
        }

        KLEIDIAI_KERNEL_LOG(sgemm_gemm.name
                            << " M=" << TileSizeM << " N=" << TileSizeN << " K=" << K);
        sgemm_gemm.ukernel.run_matmul(
            TileSizeM,
            TileSizeN,
            K,
            ATile, BTile, out_tile,
            out_row_stride_bytes, sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );

        if (direct_to_c) {
            return;
        }

        ApplyAlphaBeta2D(out_tile, TileSizeM, TileSizeN, alpha, beta, dst_tile, ldc);
        return;
    });
    return true;
}
