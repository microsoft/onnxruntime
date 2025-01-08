/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_fp32.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON specific to
    input type T1 as float32 and
    MLAS_QNBIT_GEMM_COMPUTE_TYPE SQNBIT_CompFp32.

--*/

#include <arm_neon.h>

#include <cassert>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"

namespace sqnbitgemm_neon
{

namespace
{

//
// SQNBIT_CompFp32 kernel implementation.
//

MLAS_FORCEINLINE void
Transpose4x4(float32x4_t& a0, float32x4_t& a1, float32x4_t& a2, float32x4_t& a3)
{
    // aN: aN_0 aN_1 aN_2 aN_3

    float32x4_t b0 = vzip1q_f32(a0, a1);  // a0_0 a1_0 a0_1 a1_1
    float32x4_t b1 = vzip2q_f32(a0, a1);  // a0_2 a1_2 a0_3 a1_3
    float32x4_t b2 = vzip1q_f32(a2, a3);  // a2_0 a3_0 a2_1 a3_1
    float32x4_t b3 = vzip2q_f32(a2, a3);  // a2_2 a3_2 a2_3 a3_3

    // a0_0 a1_0 a2_0 a3_0
    a0 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(b0), vreinterpretq_f64_f32(b2)));
    // a0_1 a1_1 a2_1 a3_1
    a1 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(b0), vreinterpretq_f64_f32(b2)));
    // a0_2 a1_2 a3_2 a3_2
    a2 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(b1), vreinterpretq_f64_f32(b3)));
    // a0_3 a1_3 a2_3 a3_3
    a3 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(b1), vreinterpretq_f64_f32(b3)));
}

MLAS_FORCEINLINE float32x4_t
FoldAccumulators(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3)
{
    Transpose4x4(a0, a1, a2, a3);
    return vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
}

namespace fp32_conversion
{

// Manual conversion to float takes place in two steps:
// 1. Map 4-bit values from [0, 15] to float values from [16.0f, 31.0f].
//    This target float range is convenient because the 4-bit source values can be placed directly into the
//    target float bits.
// 2. Subtract the conversion offset of 16 from the float result.

// The high 16 bits of an IEEE 754 32-bit float used as a template for creating float values.
constexpr uint16_t float_high_half_template = 0b0'10000011'0000000;
//                                           sign|exponent|partial mantissa
//                                              +|131: 2^4|~~~~ <- 4 bits go here

const uint16x8_t float_high_half_template_v = vdupq_n_u16(float_high_half_template);

constexpr float offset = 16.0f;

}  // namespace fp32_conversion

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompFp32(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkLen = 16;

    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    assert(BlkLen >= SubBlkLen && BlkLen % SubBlkLen == 0);

    const uint8x8_t LowMask = vdup_n_u8(0x0F);

    float32x4_t acc[NCols]{};

    const std::byte* QuantBData = QuantBDataColPtr;
    const float* QuantBScale = QuantBScaleColPtr;
    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
                                                     // only used if HasZeroPoint is true

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        float scale[NCols];
        UnrolledLoop<NCols>(
            [&](size_t i) { scale[i] = QuantBScale[i * StrideQuantBScale]; }
        );

        [[maybe_unused]] float offset[NCols];  // Includes zero point and float conversion offset.
                                               // only used if HasZeroPoint is true
        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
            });
        }

        for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += SubBlkLen) {
            // load A row vector elements

            // load `SubBlkLen` elements from A, padded with 0's if there aren't enough
            const size_t k_subblk_len = std::min(k_blk_len - k_idx_in_blk, SubBlkLen);
            float32x4_t av[4]{};
            LoadFloatData<SubBlkLen>(ARowPtr + k + k_idx_in_blk, k_subblk_len, av);

            // load B column vectors
            uint8x8_t bv_packed[NCols];
            const size_t b_data_block_offset = k_idx_in_blk * BlkBitWidth / 8;
            UnrolledLoop<NCols>([&](size_t i) {
                bv_packed[i] = vld1_u8(
                    reinterpret_cast<const uint8_t*>(QuantBData) + i * StrideQuantBData + b_data_block_offset
                );
            });

            uint8x8_t bv_u8[NCols][2];
            UnrolledLoop<NCols>([&](size_t i) {
                bv_u8[i][0] = vand_u8(bv_packed[i], LowMask);
                bv_u8[i][1] = vshr_n_u8(bv_packed[i], 4);
            });

            // shift left 3 and widen to 16 bits
            uint16x8_t bv_u16[NCols][2];
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr int shift = 3;
                bv_u16[i][0] = vshll_n_u8(bv_u8[i][0], shift);
                bv_u16[i][1] = vshll_n_u8(bv_u8[i][1], shift);
            });

            // combine 4 bits with float high half template
            UnrolledLoop<NCols>([&](size_t i) {
                bv_u16[i][0] = vorrq_u16(bv_u16[i][0], fp32_conversion::float_high_half_template_v);
                bv_u16[i][1] = vorrq_u16(bv_u16[i][1], fp32_conversion::float_high_half_template_v);
            });

            // `SubBlkLen` floats of B
            float32x4_t bv[NCols][4];

            // shift left 16, widen to 32 bits, and reinterpret as float
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr int shift = 16;
                bv[i][0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][0]), shift));
                bv[i][1] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][0], shift));

                bv[i][2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][1]), shift));
                bv[i][3] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][1], shift));
            });

            // subtract float conversion offset and zero point
            if constexpr (HasZeroPoint) {
                UnrolledLoop<NCols>([&](size_t i) {
                    const float32x4_t offset_v = vdupq_n_f32(offset[i]);
                    UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
                });
            } else {
                const float32x4_t offset_v = vdupq_n_f32(fp32_conversion::offset + 8.0f);
                UnrolledLoop<NCols>([&](size_t i) {
                    UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
                });
            }

            // multiply by scale
            UnrolledLoop<NCols>([&](size_t i) {
                const float32x4_t scale_v = vdupq_n_f32(scale[i]);
                UnrolledLoop<4>([&](size_t j) { bv[i][j] = vmulq_f32(bv[i][j], scale_v); });
            });

            // c[m,n] += a[m,k] * b[k,n]
            UnrolledLoop<4>([&](size_t j) {
                UnrolledLoop<NCols>([&](size_t i) { acc[i] = vfmaq_f32(acc[i], av[j], bv[i][j]); });
            });
        }

        // increment pointers to next block
        QuantBData += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        QuantBScale += 1;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }

    if constexpr (NCols == 4) {
        float32x4_t sum = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);

        if (BiasPtr != nullptr) {
            sum = vaddq_f32(sum, vld1q_f32(BiasPtr));
        }

        vst1q_f32(SumPtr, sum);
    } else {
        for (size_t i = 0; i < NCols; ++i) {
            SumPtr[i] = vaddvq_f32(acc[i]);
            if (BiasPtr != nullptr) {
                SumPtr[i] += BiasPtr[i];
            }
        }
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
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
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

}  // namespace

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
)
{
    if (QuantBZeroPoint != nullptr) {
        constexpr bool HasZeroPoint = true;
        SQ4BitGemmM1Kernel_CompFp32_Impl<HasZeroPoint>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockCountK,
            Bias
        );
    } else {
        constexpr bool HasZeroPoint = false;
        SQ4BitGemmM1Kernel_CompFp32_Impl<HasZeroPoint>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockCountK,
            Bias
        );
    }
}

namespace
{

// Block dequantize a 16 x NCols section of B from column major source to row major destination.
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4BitBlkDequantB_16xNCols(
    const std::byte* QuantBDataPtr,
    size_t StrideQuantBData,
    const float* QuantBColScalePtr,                    // pointer to NCols scales of adjacent columns
    [[maybe_unused]] const float* QuantBColOffsetPtr,  // pointer to NCols offsets of adjacent columns
                                                       // only used if HasZeroPoint is true
    float* DstColPtr
)
{
    const uint8x8_t LowMask = vdup_n_u8(0x0F);

    // load B column vectors
    uint8x8_t bv_packed[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
        bv_packed[i] = vld1_u8(
            reinterpret_cast<const uint8_t*>(QuantBDataPtr) + i * StrideQuantBData
        );
    });

    uint8x8_t bv_u8[NCols][2];
    UnrolledLoop<NCols>([&](size_t i) {
        bv_u8[i][0] = vand_u8(bv_packed[i], LowMask);
        bv_u8[i][1] = vshr_n_u8(bv_packed[i], 4);
    });

    // shift left 3 and widen to 16 bits
    uint16x8_t bv_u16[NCols][2];
    UnrolledLoop<NCols>([&](size_t i) {
        constexpr int shift = 3;
        bv_u16[i][0] = vshll_n_u8(bv_u8[i][0], shift);
        bv_u16[i][1] = vshll_n_u8(bv_u8[i][1], shift);
    });

    // combine 4 bits with float high half template
    UnrolledLoop<NCols>([&](size_t i) {
        bv_u16[i][0] = vorrq_u16(bv_u16[i][0], fp32_conversion::float_high_half_template_v);
        bv_u16[i][1] = vorrq_u16(bv_u16[i][1], fp32_conversion::float_high_half_template_v);
    });

    // `SubBlkLen` floats of B
    float32x4_t bv[NCols][4];

    // shift left 16, widen to 32 bits, and reinterpret as float
    UnrolledLoop<NCols>([&](size_t i) {
        constexpr int shift = 16;
        bv[i][0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][0]), shift));
        bv[i][1] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][0], shift));

        bv[i][2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][1]), shift));
        bv[i][3] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][1], shift));
    });

    // subtract float conversion offset and zero point
    if constexpr (HasZeroPoint) {
        UnrolledLoop<NCols>([&](size_t i) {
            const float32x4_t offset_v = vdupq_n_f32(QuantBColOffsetPtr[i]);
            UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
        });
    } else {
        const float32x4_t offset_v = vdupq_n_f32(fp32_conversion::offset + 8.0f);
        UnrolledLoop<NCols>([&](size_t i) {
            UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
        });
    }

    // multiply by scale
    UnrolledLoop<NCols>([&](size_t i) {
        const float32x4_t scale_v = vdupq_n_f32(QuantBColScalePtr[i]);
        UnrolledLoop<4>([&](size_t j) { bv[i][j] = vmulq_f32(bv[i][j], scale_v); });
    });

    // write, transposed, 16 x NCols values
    if constexpr (NCols == 4) {
        UnrolledLoop<4>([&](size_t j) {
            Transpose4x4(bv[0][j], bv[1][j], bv[2][j], bv[3][j]);

            vst1q_f32(&DstColPtr[(j * 4 + 0) * 16], bv[0][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 1) * 16], bv[1][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 2) * 16], bv[2][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 3) * 16], bv[3][j]);
        });
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            UnrolledLoop<4>([&](size_t j) {
                DstColPtr[(j * 4 + 0) * 16 + i] = vgetq_lane_f32(bv[i][j], 0);
                DstColPtr[(j * 4 + 1) * 16 + i] = vgetq_lane_f32(bv[i][j], 1);
                DstColPtr[(j * 4 + 2) * 16 + i] = vgetq_lane_f32(bv[i][j], 2);
                DstColPtr[(j * 4 + 3) * 16 + i] = vgetq_lane_f32(bv[i][j], 3);
            });
        });
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth = 4;

    float* Dst = FpData;

    const std::byte* QuantBDataCol = QuantBData;
    const float* QuantBScaleCol = QuantBScale;
    [[maybe_unused]] const std::byte* QuantBZeroPointCol = QuantBZeroPoint;  // only used if HasZeroPoint is true

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    [[maybe_unused]] const size_t StrideQuantBZeroPoint =  // only used if HasZeroPoint is true
        MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    //
    // Proceed down 16 column-wide regions of B. Dequantize and write output 16 x 16 elements at a time.
    //

    // scales of blocks from 16 adjacent columns
    float scale[16];
    // float conversion offsets (including zero point) of blocks from 16 adjacent columns
    [[maybe_unused]] float offset[16];  // only used if HasZeroPoint is true

    size_t n_cols_remaining = CountN;
    while (n_cols_remaining > 15) {
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            for (size_t nn = 0; nn < 16; ++nn) {
                scale[nn] = QuantBScaleCol[nn * BlockCountK + k_blk_idx];

                if constexpr (HasZeroPoint) {
                    const std::byte zp_packed =
                        QuantBZeroPointCol[nn * StrideQuantBZeroPoint + k_blk_idx / 2];
                    const std::byte zp = ((k_blk_idx & 1) == 1)
                                             ? (zp_packed >> 4)
                                             : (zp_packed & std::byte{0x0F});
                    offset[nn] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
                }
            }

            const size_t kklen = std::min(CountK - k, BlkLen);

            for (size_t kk = 0; kk < kklen; kk += 16) {
                constexpr size_t NCols = 4;

                const float* ScalePtr = &scale[0];
                const float* OffsetPtr = HasZeroPoint ? &offset[0] : nullptr;

                float* DstColPtr = Dst;

                for (size_t nn = 0; nn < 16; nn += NCols) {
                    const std::byte* QuantBDataPtr = QuantBDataCol + nn * StrideQuantBData + (k + kk) * BlkBitWidth / 8;

                    Q4BitBlkDequantB_16xNCols<NCols, HasZeroPoint>(
                        QuantBDataPtr,
                        StrideQuantBData,
                        ScalePtr,
                        OffsetPtr,
                        DstColPtr
                    );

                    ScalePtr += NCols;
                    if constexpr (HasZeroPoint) {
                        OffsetPtr += NCols;
                    }
                    DstColPtr += NCols;
                }

                Dst += 16 * std::min(kklen - kk, size_t{16});
            }
        }

        n_cols_remaining -= 16;

        QuantBDataCol += 16 * StrideQuantBData;
        QuantBScaleCol += 16 * BlockCountK;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointCol += 16 * StrideQuantBZeroPoint;
        }
    }

    if (n_cols_remaining > 0) {
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            for (size_t nn = 0; nn < n_cols_remaining; ++nn) {
                scale[nn] = QuantBScaleCol[nn * BlockCountK + k_blk_idx];

                if constexpr (HasZeroPoint) {
                    const std::byte zp_packed =
                        QuantBZeroPointCol[nn * StrideQuantBZeroPoint + k_blk_idx / 2];
                    const std::byte zp = ((k_blk_idx & 1) == 1)
                                             ? (zp_packed >> 4)
                                             : (zp_packed & std::byte{0x0F});
                    offset[nn] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
                }
            }

            const size_t kklen = std::min(CountK - k, BlkLen);

            for (size_t kk = 0; kk < kklen; kk += 16) {
                // zero out the 16x16 block in Dst first to ensure zero padding
                const float32x4_t zero_v = vdupq_n_f32(0.0f);
                UnrolledLoop<16 * 4>([&](size_t i) {
                    vst1q_f32(Dst + 4 * i, zero_v);
                });

                const float* ScalePtr = &scale[0];
                const float* OffsetPtr = HasZeroPoint ? &offset[0] : nullptr;

                float* DstColPtr = Dst;

                for (size_t nn = 0; nn < n_cols_remaining; ++nn) {
                    const std::byte* QuantBDataPtr = QuantBDataCol + nn * StrideQuantBData + (k + kk) * BlkBitWidth / 8;

                    Q4BitBlkDequantB_16xNCols<1, HasZeroPoint>(
                        QuantBDataPtr,
                        StrideQuantBData,
                        ScalePtr,
                        OffsetPtr,
                        DstColPtr
                    );

                    ScalePtr += 1;
                    if constexpr (HasZeroPoint) {
                        OffsetPtr += 1;
                    }
                    DstColPtr += 1;
                }

                Dst += 16 * std::min(kklen - kk, size_t{16});
            }
        }
    }
}

}  // namespace

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
)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen,
            FpData,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            CountN,
            CountK,
            BlockCountK
        );
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen,
            FpData,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            CountN,
            CountK,
            BlockCountK
        );
    }
}

}  // namespace sqnbitgemm_neon
