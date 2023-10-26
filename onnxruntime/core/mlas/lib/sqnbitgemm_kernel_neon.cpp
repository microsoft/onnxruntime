/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <arm_neon.h>

#include "sqnbitgemm.h"

//
// Hardware-specific kernel type.
//
struct MLAS_SQNBIT_GEMM_KERNEL_NEON {
};

//
// MlasSQNBitGemmKernel and helpers.
//

template <size_t BlkBitWidth, size_t BlkLen>
MLAS_FORCEINLINE size_t
MlasSQNBitGemmKernelNeon(
    const float* A,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t BlockStridePackedB,
    size_t ldc,
    const float* Bias
)
{
    auto impl0_reference = [&]() {
        static_assert(BlkBitWidth == 4);

        for (size_t m = 0; m < CountM; ++m) {
            for (size_t n = 0; n < CountN; ++n) {
                for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                    const size_t kblocklen = std::min(CountK - k, BlkLen);

                    const float b_s = PackedBScale[n * BlockStridePackedB + k_blk_idx];
                    const uint8_t b_z = [&]() -> uint8_t {
                        if (PackedBZeroPoint != nullptr) {
                            const size_t i = n * BlockStridePackedB + k_blk_idx;
                            uint8_t zp_packed = PackedBZeroPoint[i / 2];
                            return ((i & 1) == 1) ? (zp_packed >> 4) : (zp_packed & 0x0F);
                        } else {
                            return 8;
                        }
                    }();
                    const uint8_t* b_data =
                        PackedBData + (n * BlockStridePackedB + k_blk_idx) * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);

                    for (size_t kk = 0; kk < kblocklen; ++kk) {
                        uint8_t b_packed = b_data[kk / 2];
                        uint8_t b_byte = ((kk & 1) == 1) ? (b_packed >> 4) : (b_packed & 0x0F);
                        float b_value = (b_byte - b_z) * b_s;

                        C[m * ldc + n] += A[m * lda + k + kk] * b_value;
                    }
                }

                if (Bias) {
                    C[m * ldc + n] += Bias[n];
                }
            }
        }

        return CountM;
    };

    return impl0_reference();
}

template <>
MLAS_FORCEINLINE size_t
MlasSQNBitGemmKernel<4, 32, MLAS_SQNBIT_GEMM_KERNEL_NEON>(
    const float* A,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t BlockStridePackedB,
    size_t ldc,
    const float* Bias
)
{
    return MlasSQNBitGemmKernelNeon<4, 32>(
        A, PackedBData, PackedBScale, PackedBZeroPoint, C, CountM, CountN, CountK, lda,
        BlockStridePackedB, ldc, Bias
    );
}

//
// MlasQNBitBlkDequantBForSgemm and helpers.
//

template <size_t BlkBitWidth, size_t BlkLen>
MLAS_FORCEINLINE void
MlasQNBitBlkDequantBForSgemmNeon(
    float* FpData,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStridePackedB
)
{
    auto impl0_reference = [&]() {
        static_assert(BlkBitWidth == 4);

        float* Dst = FpData;

        const uint8_t* PackedBDataCol = PackedBData;
        const float* PackedBScaleCol = PackedBScale;
        const uint8_t* PackedBZeroPointCol = PackedBZeroPoint;

        for (size_t n = 0; n < CountN; n += 16) {
            const size_t nnlen = std::min(CountN - n, size_t{16});

            for (size_t nn = 0; nn < nnlen; ++nn) {

                for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                    const size_t kklen = std::min(CountK - k, BlkLen);

                    const uint8_t* b_data =
                        PackedBDataCol + k_blk_idx * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                    const float b_s = PackedBScaleCol[k_blk_idx];
                    const uint8_t b_z =
                        (PackedBZeroPointCol != nullptr)
                            ? ((k_blk_idx & 1) == 1)
                                  ? PackedBZeroPointCol[k_blk_idx / 2] >> 4
                                  : PackedBZeroPointCol[k_blk_idx / 2] & 0x0F
                            : 8;

                    for (size_t kk = 0; kk < kklen; ++kk) {
                        const uint8_t b_packed = b_data[kk / 2];
                        const uint8_t b_byte = ((kk & 1) == 1) ? b_packed >> 4 : b_packed & 0x0F;
                        const float b_value = (b_byte - b_z) * b_s;

                        Dst[(k + kk) * 16 + nn] = b_value;
                    }
                }

                PackedBDataCol += BlockStridePackedB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                PackedBScaleCol += BlockStridePackedB;
                if (PackedBZeroPointCol != nullptr) {
                    PackedBZeroPointCol += MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockStridePackedB);
                }
            }

            // zero out any remaining columns

            if (nnlen < 16) {
                for (size_t k = 0; k < CountK; ++k) {
                    std::fill_n(Dst + (k * 16) + nnlen, 16 - nnlen, 0.0f);
                }
            }

            Dst += CountK * 16;
        }
    };

    impl0_reference();
}

template <>
MLAS_FORCEINLINE void
MlasQNBitBlkDequantBForSgemm<4, 32, MLAS_SQNBIT_GEMM_KERNEL_NEON>(
    float* FpData,
    const uint8_t* PackedBData,
    const float* PackedBScale,
    const uint8_t* PackedBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStridePackedB
)
{
    MlasQNBitBlkDequantBForSgemmNeon<4, 32>(
        FpData, PackedBData, PackedBScale, PackedBZeroPoint, CountN, CountK, BlockStridePackedB
    );
}

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;
    d.Operations[QuantVariant_BitWidth4_BlockSize32] = MlasSQNBitGemmOperation<4, 32, MLAS_SQNBIT_GEMM_KERNEL_NEON>;
    return d;
}();
