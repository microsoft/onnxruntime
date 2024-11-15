/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm_kernel_neon.h

Abstract:

    This module includes function declarations and common helper functions for
    SQNBitGemm ARM NEON kernels.

--*/

#pragma once

#include <arm_neon.h>

#include <cassert>
#include <cstddef>
#include <utility>

#include "mlasi.h"

namespace sqnbitgemm_neon
{

//
// Function declarations for SQNBitGemm ARM NEON kernel entry points.
// Refer to the prototypes in qnbitgemm.h for documentation.
// These are declared here so they can be used to initialize the
// MLAS_QNBIT_GEMM_DISPATCH structure and also be implemented in separate
// files.
//

// SQNBIT_CompFp32 declarations

void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
);

void
SQ4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
);

// HQNBIT_CompFp16 declarations
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
void
HQ4BitGemmPackQuantBData_CompFp16(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
);

void
HQ4BitBlkDequantBForHgemm_CompFp16(
    size_t BlkLen,
    MLAS_FP16* FpData,
    const std::byte* QuantBData,
    const MLAS_FP16* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t K,
    size_t BlockCountK
);

void
HQ4BitGemmKernel_CompFp16(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    const MLAS_FP16* Bias,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc
);

#endif  // !(defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64))

// SQNBIT_CompInt8 declarations

void
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
);

size_t
SQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
);

//
// General helpers.
//

template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}

template <size_t Capacity>
MLAS_FORCEINLINE void
LoadFloatData(const float* src, size_t count, float32x4_t (&dst)[Capacity / 4])
{
    static_assert(Capacity % 4 == 0, "Capacity must be divisible by 4.");

    assert(count <= Capacity);

    size_t vi = 0;  // vector index

    // handle 4 values at a time
    while (count > 3) {
        dst[vi] = vld1q_f32(src);

        vi += 1;
        src += 4;
        count -= 4;
    }

    // handle remaining values
    if (count > 0) {
        dst[vi] = vsetq_lane_f32(src[0], dst[vi], 0);

        if (count > 1) {
            dst[vi] = vsetq_lane_f32(src[1], dst[vi], 1);

            if (count > 2) {
                dst[vi] = vsetq_lane_f32(src[2], dst[vi], 2);
            }
        }
    }
}

}  // namespace sqnbitgemm_neon
