// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "mlasi.h"

#if !defined(DISABLE_FLOAT8_TYPES)

#include "core/common/common.h"
#include "core/common/float8.h"

namespace {

inline float Fp8ByteToFloat(uint8_t value, mlas_fp8_mode mode) {
    using onnxruntime::Float8E4M3FN;
    using onnxruntime::Float8E4M3FNUZ;
    using onnxruntime::Float8E5M2;
    using onnxruntime::Float8E5M2FNUZ;
    switch (mode) {
        case MLAS_FP8_MODE_E4M3_INF:
            return Float8E4M3FN(value, Float8E4M3FN::FromBits()).ToFloat();
        case MLAS_FP8_MODE_E4M3_SAT:
            return Float8E4M3FNUZ(value, Float8E4M3FNUZ::FromBits()).ToFloat();
        case MLAS_FP8_MODE_E5M2_INF:
            return Float8E5M2(value, Float8E5M2::FromBits()).ToFloat();
        case MLAS_FP8_MODE_E5M2_SAT:
            return Float8E5M2FNUZ(value, Float8E5M2FNUZ::FromBits()).ToFloat();
        default:
            MLAS_THROW_EX(std::invalid_argument, "Unsupported FP8 GEMM mode.");
            break;
    }
}

inline bool IsValidFp8Mode(mlas_fp8_mode mode) {
    switch (mode) {
        case MLAS_FP8_MODE_E4M3_INF:
        case MLAS_FP8_MODE_E4M3_SAT:
        case MLAS_FP8_MODE_E5M2_INF:
        case MLAS_FP8_MODE_E5M2_SAT:
            return true;
        default:
            return false;
    }
}

// Computes the parallel work item count without wrapping before ptrdiff_t narrowing.
inline ptrdiff_t CheckedWorkItems(size_t batch_count, size_t m) {
    size_t work_items = 0;
    ORT_ENFORCE(!MlasMultiplyOverflowsSizeT(batch_count, m, &work_items), "FP8 GEMM work item count overflow.");
    ORT_ENFORCE(work_items <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                "FP8 GEMM work item count exceeds ptrdiff_t range.");
    return static_cast<ptrdiff_t>(work_items);
}

// Validates the largest row-major strided matrix offset used by the GEMM loops.
inline void CheckStridedMatrixOffset(size_t rows, size_t cols, size_t row_stride) {
    if (rows == 0 || cols == 0) {
        return;
    }

    size_t offset = 0;
    ORT_ENFORCE(!MlasMultiplyOverflowsSizeT(rows - 1, row_stride, &offset),
                "FP8 GEMM matrix offset overflow.");
    ORT_ENFORCE(cols - 1 <= std::numeric_limits<size_t>::max() - offset,
                "FP8 GEMM matrix offset overflow.");
}

// Validates the largest block-scale or zero-point offset used by the GEMM loops.
inline void CheckBlockedMatrixOffset(size_t blocks0, size_t stride0, size_t blocks1, size_t stride1) {
    if (blocks0 == 0 || blocks1 == 0) {
        return;
    }

    size_t offset0 = 0;
    ORT_ENFORCE(!MlasMultiplyOverflowsSizeT(blocks0 - 1, stride0, &offset0),
                "FP8 GEMM block offset overflow.");
    size_t offset1 = 0;
    ORT_ENFORCE(!MlasMultiplyOverflowsSizeT(blocks1 - 1, stride1, &offset1),
                "FP8 GEMM block offset overflow.");
    ORT_ENFORCE(offset1 <= std::numeric_limits<size_t>::max() - offset0,
                "FP8 GEMM block offset overflow.");
}

inline void CheckBlockCount(size_t actual, size_t supplied, const char* dimension_name) {
    ORT_ENFORCE(actual == supplied, "FP8 GEMM ", dimension_name, " block count must match shape and block size.");
}

// Validate caller-provided buffers, strides, and block metadata before parallel workers dereference them.
inline void CheckFp8GemmBatchParams(
    const MLAS_FP8_GEMM_SHAPE_PARAMS& shape,
    const MLAS_FP8_GEMM_DATA_PARAMS& params) {
    ORT_ENFORCE(IsValidFp8Mode(params.Fp8Type), "FP8 GEMM mode must be valid.");
    ORT_ENFORCE(params.BlockSizeM != 0 && params.BlockSizeK != 0 && params.BlockSizeN != 0,
                "FP8 GEMM block sizes must be non-zero.");

    const bool writes_output = shape.M != 0 && shape.N != 0;
    const bool reads_reduction_data = writes_output && shape.K != 0;

    // Empty-output GEMMs do not dereference C, and empty reductions do not read A/B.
    if (reads_reduction_data) {
        ORT_ENFORCE(params.A != nullptr, "FP8 GEMM A buffer must not be null.");
        ORT_ENFORCE(params.B != nullptr, "FP8 GEMM B buffer must not be null.");
        ORT_ENFORCE(params.lda >= shape.K, "FP8 GEMM lda must be greater than or equal to K.");
        ORT_ENFORCE(params.ldb >= shape.N, "FP8 GEMM ldb must be greater than or equal to N.");

        CheckStridedMatrixOffset(shape.M, shape.K, params.lda);
        CheckStridedMatrixOffset(shape.K, shape.N, params.ldb);
    }

    if (writes_output) {
        ORT_ENFORCE(params.C != nullptr, "FP8 GEMM C buffer must not be null.");
        ORT_ENFORCE(params.ldc >= shape.N, "FP8 GEMM ldc must be greater than or equal to N.");

        CheckStridedMatrixOffset(shape.M, shape.N, params.ldc);
    }

    const size_t blocks_m = shape.M == 0 ? 0 : ((shape.M - 1) / params.BlockSizeM) + 1;
    const size_t blocks_k = shape.K == 0 ? 0 : ((shape.K - 1) / params.BlockSizeK) + 1;
    const size_t blocks_n = shape.N == 0 ? 0 : ((shape.N - 1) / params.BlockSizeN) + 1;

    if (reads_reduction_data && params.ScaleA != nullptr) {
        CheckBlockCount(blocks_m, params.BlocksM, "M");
        CheckBlockCount(blocks_k, params.BlocksK, "K");
        CheckBlockedMatrixOffset(blocks_m, params.ScaleAStrideM, blocks_k, params.ScaleAStrideK);
    }
    if (reads_reduction_data && params.ScaleB != nullptr) {
        CheckBlockCount(blocks_k, params.BlocksK, "K");
        CheckBlockCount(blocks_n, params.BlocksN, "N");
        CheckBlockedMatrixOffset(blocks_k, params.ScaleBStrideK, blocks_n, params.ScaleBStrideN);
    }
    if (reads_reduction_data && params.ZeroPointA != nullptr) {
        CheckBlockCount(blocks_m, params.BlocksM, "M");
        CheckBlockCount(blocks_k, params.BlocksK, "K");
        CheckBlockedMatrixOffset(blocks_m, params.ZeroPointAStrideM, blocks_k, params.ZeroPointAStrideK);
    }
    if (reads_reduction_data && params.ZeroPointB != nullptr) {
        CheckBlockCount(blocks_k, params.BlocksK, "K");
        CheckBlockCount(blocks_n, params.BlocksN, "N");
        CheckBlockedMatrixOffset(blocks_k, params.ZeroPointBStrideK, blocks_n, params.ZeroPointBStrideN);
    }
}

}  // namespace

void
MLASCALL
MlasFp8GemmBatch(
    const MLAS_FP8_GEMM_SHAPE_PARAMS& Shape,
    const MLAS_FP8_GEMM_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
    )
{
    const size_t M = Shape.M;
    const size_t N = Shape.N;
    const size_t K = Shape.K;

    if (BatchN == 0 || M == 0 || N == 0) {
        return;
    }

    const ptrdiff_t WorkItems = CheckedWorkItems(BatchN, M);

    ORT_ENFORCE(DataParams != nullptr, "FP8 GEMM data parameters must not be null.");

    for (size_t batch = 0; batch < BatchN; ++batch) {
        CheckFp8GemmBatchParams(Shape, DataParams[batch]);
    }

    MlasTrySimpleParallel(ThreadPool, WorkItems, [&](ptrdiff_t tid) {
        const size_t batch = static_cast<size_t>(tid) / M;
        const size_t m = static_cast<size_t>(tid) % M;
        const auto& params = DataParams[batch];

        const auto* a_fp8 = static_cast<const uint8_t*>(params.A);
        const auto* b_fp8 = static_cast<const uint8_t*>(params.B);
        auto* c = static_cast<float*>(params.C);
        const auto* scale_a = params.ScaleA;
        const auto* scale_b = params.ScaleB;
        const auto* zp_a = static_cast<const uint8_t*>(params.ZeroPointA);
        const auto* zp_b = static_cast<const uint8_t*>(params.ZeroPointB);

        const size_t block_m = m / params.BlockSizeM;
        for (size_t n = 0; n < N; ++n) {
            const size_t block_n = n / params.BlockSizeN;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                const size_t block_k = k / params.BlockSizeK;

                const size_t a_scale_idx = block_m * params.ScaleAStrideM + block_k * params.ScaleAStrideK;
                const size_t b_scale_idx = block_k * params.ScaleBStrideK + block_n * params.ScaleBStrideN;
                const float scale_a_val = scale_a ? scale_a[a_scale_idx] : 1.0f;
                const float scale_b_val = scale_b ? scale_b[b_scale_idx] : 1.0f;

                const size_t a_zp_idx = block_m * params.ZeroPointAStrideM + block_k * params.ZeroPointAStrideK;
                const size_t b_zp_idx = block_k * params.ZeroPointBStrideK + block_n * params.ZeroPointBStrideN;
                const float zp_a_val = zp_a ? Fp8ByteToFloat(zp_a[a_zp_idx], params.Fp8Type) : 0.0f;
                const float zp_b_val = zp_b ? Fp8ByteToFloat(zp_b[b_zp_idx], params.Fp8Type) : 0.0f;

                const float a_val = Fp8ByteToFloat(a_fp8[m * params.lda + k], params.Fp8Type);
                const float b_val = Fp8ByteToFloat(b_fp8[k * params.ldb + n], params.Fp8Type);

                const float a_deq = (a_val - zp_a_val) * scale_a_val;
                const float b_deq = (b_val - zp_b_val) * scale_b_val;
                acc += a_deq * b_deq;
            }

            if (params.ScaleY != nullptr) {
                acc *= params.ScaleY[0];
            }
            if (params.ZeroPointY != nullptr) {
                acc += params.ZeroPointY[0];
            }
            c[m * params.ldc + n] = acc;
        }
    });
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)
