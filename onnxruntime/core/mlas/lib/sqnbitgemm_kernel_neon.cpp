/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include "sqnbitgemm.h"

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <utility>

namespace
{

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

MLAS_FORCEINLINE float32x4_t
FoldAccumulators(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3)
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

    return vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
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

template <size_t NCols>
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

    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    constexpr size_t SubBlkLen = 16;  // number of block elements to process in a sub-block iteration
    assert(BlkLen % SubBlkLen == 0);

    const uint8x8_t LowMask = vdup_n_u8(0x0F);

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

    float32x4_t acc[NCols]{};

    const std::byte* QuantBData = QuantBDataColPtr;
    const float* QuantBScale = QuantBScaleColPtr;
    size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        float scale[NCols];
        UnrolledLoop<NCols>(
            [&](size_t i) { scale[i] = QuantBScale[i * StrideQuantBScale]; }
        );

        float offset[NCols];  // Includes zero point and float conversion offset of 16.
        if (QuantBZeroPointColPtr != nullptr) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = 16.0f + std::to_integer<uint8_t>(zp);
            });
        } else {
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr float zp = 8.0f;
                offset[i] = 16.0f + zp;
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

            // dequantize B

            // shift left 3 and widen to 16 bits
            uint16x8_t bv_u16[NCols][2];
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr int shift = 3;
                bv_u16[i][0] = vshll_n_u8(bv_u8[i][0], shift);
                bv_u16[i][1] = vshll_n_u8(bv_u8[i][1], shift);
            });

            // combine 4 bits with float high half template
            UnrolledLoop<NCols>([&](size_t i) {
                bv_u16[i][0] = vorrq_u16(bv_u16[i][0], float_high_half_template_v);
                bv_u16[i][1] = vorrq_u16(bv_u16[i][1], float_high_half_template_v);
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

            // subtract float conversion offset (16) and zero point
            UnrolledLoop<NCols>([&](size_t i) {
                const float32x4_t offset_v = vdupq_n_f32(offset[i]);
                UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
            });

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
        QuantBZeroPointIdx += 1;
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

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

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
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if (QuantBZeroPointColPtr != nullptr) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if (QuantBZeroPointColPtr != nullptr) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

MLAS_FORCEINLINE void
Q4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB
)
{
    auto impl0_reference = [&]() {
        constexpr size_t BlkBitWidth = 4;

        float* Dst = FpData;

        const std::byte* QuantBDataCol = QuantBData;
        const float* QuantBScaleCol = QuantBScale;
        const std::byte* QuantBZeroPointCol = QuantBZeroPoint;

        for (size_t n = 0; n < CountN; n += 16) {
            const size_t nnlen = std::min(CountN - n, size_t{16});

            for (size_t nn = 0; nn < nnlen; ++nn) {
                for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                    const size_t kklen = std::min(CountK - k, BlkLen);

                    const std::byte* b_data =
                        QuantBDataCol + k_blk_idx * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                    const float b_s = QuantBScaleCol[k_blk_idx];
                    const uint8_t b_z =
                        (QuantBZeroPointCol != nullptr)
                            ? ((k_blk_idx & 1) == 1)
                                  ? std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] >> 4)
                                  : std::to_integer<uint8_t>(QuantBZeroPointCol[k_blk_idx / 2] & std::byte{0x0F})
                            : 8;

                    for (size_t kk = 0; kk < kklen; ++kk) {
                        const size_t packed_idx = kk % 16;

                        const bool is_low_half = packed_idx < 8;
                        const size_t packed_byte_idx = packed_idx % 8;
                        const size_t packed_range_offset = (kk / 16) * 8;

                        const std::byte b_packed = b_data[packed_range_offset + packed_byte_idx];
                        const std::byte b_byte = is_low_half ? (b_packed & std::byte{0x0F}) : (b_packed >> 4);
                        const float b_value = (std::to_integer<int8_t>(b_byte) - b_z) * b_s;

                        Dst[(k + kk) * 16 + nn] = b_value;
                    }
                }

                QuantBDataCol += BlockStrideQuantB * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
                QuantBScaleCol += BlockStrideQuantB;
                if (QuantBZeroPointCol != nullptr) {
                    QuantBZeroPointCol += MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockStrideQuantB);
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

//
// CompInt8 kernel implementation and related helpers
//

template <size_t SubBlkLen>
MLAS_FORCEINLINE void
QuantizeBlock(
    size_t BlkLen,
    const float* A,
    size_t ElementCount,
    std::byte* QuantA
)
{
    static_assert(SubBlkLen >= 16 && SubBlkLen % 16 == 0);

    assert(BlkLen % SubBlkLen == 0);

    constexpr size_t VectorCount = SubBlkLen / 4;

    //
    // Scan block values first to determine scale.
    //

    float amax = 0.0f;  // max of absolute values of A block

    size_t k;
    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[VectorCount]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        float32x4_t abs_a[VectorCount];
        UnrolledLoop<VectorCount>([&](size_t i) {
            abs_a[i] = vabsq_f32(a[i]);
        });

        // find amax of SubBlkLen elements
        for (size_t interval = VectorCount / 2; interval > 0; interval /= 2) {
            for (size_t i = 0; i < interval; ++i) {
                abs_a[i] = vmaxq_f32(abs_a[i], abs_a[i + interval]);
            }
        }

        // update existing amax
        amax = std::max(amax, vmaxvq_f32(abs_a[0]));
    }

    constexpr float range_max = (1 << 7) - 1;
    const float scale = amax / range_max;
    const float scale_reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;

    Q8BlkScale(QuantA) = scale;

    //
    // Compute quantized block values.
    //

    int8_t* QuantAData = Q8BlkData(QuantA);

    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[VectorCount]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        UnrolledLoop<VectorCount>([&](size_t i) {
            a[i] = vmulq_n_f32(a[i], scale_reciprocal);
        });

        int32x4_t a_s32[VectorCount];
        UnrolledLoop<VectorCount>([&](size_t i) {
            a_s32[i] = vcvtaq_s32_f32(a[i]);
        });

        UnrolledLoop<VectorCount>([&](size_t i) {
            QuantAData[k + i * 4 + 0] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 0));
            QuantAData[k + i * 4 + 1] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 1));
            QuantAData[k + i * 4 + 2] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 2));
            QuantAData[k + i * 4 + 3] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 3));
        });
    }

    //
    // Zero out any remaining sub-block elements.
    //

    for (; k < BlkLen; k += SubBlkLen) {
        const int8x16_t Zeros = vdupq_n_s8(0);
        UnrolledLoop<SubBlkLen / 16>([&](size_t i) {
            vst1q_s8(QuantAData + k + i * 16, Zeros);
        });
    }
}

void MLASCALL
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    const float* ADataBlkPtr = A;
    std::byte* QuantABlkPtr = QuantA;

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        QuantizeBlock<16>(BlkLen, ADataBlkPtr, k_blk_len, QuantABlkPtr);

        ADataBlkPtr += BlkLen;
        QuantABlkPtr += Q8BlkSize(BlkLen);
    }
}

template <size_t NCols>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompInt8(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
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
    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    constexpr size_t BlkBitWidth = 4;

    constexpr size_t SubBlkLen = 16;  // number of block elements to process in a sub-block iteration
    assert(BlkLen % SubBlkLen == 0);

    const uint8x8_t LowMask = vdup_n_u8(0x0F);

    const std::byte* QuantA = QuantARowPtr;

    const std::byte* QuantBData = QuantBDataColPtr;
    const float* QuantBScale = QuantBScaleColPtr;
    size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    float32x4_t acc[NCols]{};

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        const float a_scale = Q8BlkScale(QuantA);
        const int8_t* a_data = Q8BlkData(QuantA);

        float b_scale[NCols];
        UnrolledLoop<NCols>([&](size_t i) { b_scale[i] = QuantBScale[i * StrideQuantBScale]; });

        int8_t b_zp[NCols];
        if (QuantBZeroPointColPtr != nullptr) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                b_zp[i] = ((QuantBZeroPointIdx & 1) == 1)
                              ? std::to_integer<int8_t>(zp_packed >> 4)
                              : std::to_integer<int8_t>(zp_packed & std::byte{0x0F});
            });
        } else {
            UnrolledLoop<NCols>([&](size_t i) {
                b_zp[i] = 8;
            });
        }

        for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += SubBlkLen) {
            // load A row vector
            int8x16_t av = vld1q_s8(a_data + k_idx_in_blk);

            // load B column vectors
            uint8x8_t bv_packed[NCols];
            const size_t b_data_block_offset = k_idx_in_blk * BlkBitWidth / 8;
            UnrolledLoop<NCols>([&](size_t i) {
                bv_packed[i] = vld1_u8(
                    reinterpret_cast<const uint8_t*>(QuantBData) + i * StrideQuantBData + b_data_block_offset
                );
            });

            int8x16_t bv[NCols];
            UnrolledLoop<NCols>([&](size_t i) {
                const int8x8_t lo = vreinterpret_s8_u8(vand_u8(bv_packed[i], LowMask));
                const int8x8_t hi = vreinterpret_s8_u8(vshr_n_u8(bv_packed[i], 4));
                bv[i] = vcombine_s8(lo, hi);
            });

            // subtract B zero point
            UnrolledLoop<NCols>([&](size_t i) {
                const int8x16_t zp_v = vdupq_n_s8(b_zp[i]);
                bv[i] = vsubq_s8(bv[i], zp_v);
            });

            // compute quantized dot product
            int32x4_t dot[NCols]{};
            UnrolledLoop<NCols>([&](size_t i) {
                dot[i] = vdotq_s32(dot[i], av, bv[i]);
            });

            // convert dot product result to float
            float32x4_t dot_f32[NCols];
            UnrolledLoop<NCols>([&](size_t i) {
                dot_f32[i] = vcvtq_f32_s32(dot[i]);
            });

            // multiply dot product result by scale and update accumulator
            UnrolledLoop<NCols>([&](size_t i) {
                const float32x4_t scale_v = vdupq_n_f32(a_scale * b_scale[i]);
                acc[i] = vfmaq_f32(acc[i], dot_f32[i], scale_v);
            });
        }

        // increment pointers to next block
        QuantA += Q8BlkSize(BlkLen);
        QuantBData += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        QuantBScale += 1;
        QuantBZeroPointIdx += 1;
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

MLAS_FORCEINLINE
void
SQ4BitGemmM1Kernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const std::byte* QuantARowPtr = QuantA;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

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
        ComputeDotProducts_BlkBitWidth4_CompInt8<NCols>(
            BlkLen,
            QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if (QuantBZeroPointColPtr != nullptr) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompInt8<1>(
            BlkLen,
            QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if (QuantBZeroPointColPtr != nullptr) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

}  // namespace

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32;
    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8;

    return d;
}();
