/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm_neon.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory.

    This implementation is for ARM NEON.

--*/

#include <arm_neon.h>

#include "q4gemm.h"

struct MLAS_FP_Q4_GEMM_KERNEL_NEON {
};

//
// MlasQ4GemmKernel and related helper functions
//

template <typename Q4Type>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernelNeon(const float* A,
                     const uint8_t* PackedB,
                     float* C,
                     size_t CountM,
                     size_t CountN,
                     size_t CountK,
                     size_t lda,
                     size_t ldb,
                     size_t ldc,
                     const float* Bias);

namespace q4gemm_neon
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

template <typename Q4Type, size_t NCols>
MLAS_FORCEINLINE void
ComputeDotProducts(const float* ARowPtr,
                   const uint8_t* PackedBColPtr,
                   float* SumPtr,
                   size_t CountK,
                   size_t ldb,
                   const float* BiasPtr)
{
    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    const int8x16_t LowMask = vdupq_n_s8(0x0F);

    float32x4_t acc[NCols];
    for (int i = 0; i < NCols; ++i) {
        acc[i] = vdupq_n_f32(0.0f);
    }

    const uint8_t* PackedBBlobPtr = PackedBColPtr;

    for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, Q4Type::BlkLen);

        float scale[NCols];
        UnrolledLoop<NCols>(
            [&](size_t i) { scale[i] = MlasQ4BlkScale<Q4Type>(PackedBBlobPtr + i * ldb); });

        uint8_t zp[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            if constexpr (MlasQ4BlkHasZeroPoint<Q4Type>()) {
                zp[i] = MlasQ4BlkZeroPoint<Q4Type>(PackedBBlobPtr + i * ldb);
            } else {
                zp[i] = 8;
            }
        });

        const uint8_t* b_data[NCols];
        UnrolledLoop<NCols>(
            [&](size_t i) { b_data[i] = MlasQ4BlkData<Q4Type>(PackedBBlobPtr + i * ldb); });

        for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += 32) {
            // load A row vector elements

            // load 32 elements from A padding with 0's if there aren't enough
            float a_segment[32];
            {
                const size_t k_subblk_len = std::min(k_blk_len - k_idx_in_blk, size_t{32});
                const float* a_begin = ARowPtr + k + k_idx_in_blk;
                std::copy(a_begin, a_begin + k_subblk_len, a_segment);
                std::fill(a_segment + k_subblk_len, a_segment + 32, 0.0f);
            }

            // 32 floats of A
            float32x4_t av[8];
            UnrolledLoop<8>([&](size_t i) { av[i] = vld1q_f32(a_segment + i * 4); });

            // load B column vectors
            int8x16_t bv_packed[NCols];
            UnrolledLoop<NCols>([&](size_t i) { bv_packed[i] = vld1q_s8(b_data[i]); });

            int8x16_t bv_bytes[NCols][2];

            UnrolledLoop<NCols>([&](size_t i) {
                bv_bytes[i][0] = vandq_s8(bv_packed[i], LowMask);
                bv_bytes[i][1] = vandq_s8(vshrq_n_s8(bv_packed[i], 4), LowMask);
            });

            // dequantize B

            // subtract zero point
            UnrolledLoop<NCols>([&](size_t i) {
                const int8x16_t zpv = vdupq_n_s8(zp[i]);
                bv_bytes[i][0] = vsubq_s8(bv_bytes[i][0], zpv);
                bv_bytes[i][1] = vsubq_s8(bv_bytes[i][1], zpv);
            });

            // widen to int16
            int16x8_t bv_int16[NCols][4];

            UnrolledLoop<NCols>([&](size_t i) {
                bv_int16[i][0] = vmovl_s8(vget_low_s8(bv_bytes[i][0]));
                bv_int16[i][1] = vmovl_s8(vget_high_s8(bv_bytes[i][0]));
                bv_int16[i][2] = vmovl_s8(vget_low_s8(bv_bytes[i][1]));
                bv_int16[i][3] = vmovl_s8(vget_high_s8(bv_bytes[i][1]));
            });

            // 32 floats of B
            float32x4_t bv[NCols][8];

            // widen to int32, cast to float32

            UnrolledLoop<NCols>([&](size_t i) {
                bv[i][0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][0])));
                bv[i][1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][0])));

                bv[i][2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][1])));
                bv[i][3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][1])));
            });

            UnrolledLoop<NCols>([&](size_t i) {
                bv[i][4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][2])));
                bv[i][5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][2])));

                bv[i][6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][3])));
                bv[i][7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][3])));
            });

            // multiply by scale
            UnrolledLoop<NCols>([&](size_t i) {
                UnrolledLoop<8>([&](size_t j) { bv[i][j] = vmulq_n_f32(bv[i][j], scale[i]); });
            });

            // c += a * b
            UnrolledLoop<8>([&](size_t j) {
                UnrolledLoop<NCols>([&](size_t i) { acc[i] = vfmaq_f32(acc[i], av[j], bv[i][j]); });
            });

            // increment b data pointers to next 32 elements
            UnrolledLoop<NCols>([&](size_t i) { b_data[i] += 16; });
        }

        PackedBBlobPtr += Q4Type::BlobSize;
    }

    if constexpr (NCols == 4) {
        float32x4_t sum = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);

        if (BiasPtr != nullptr) {
            sum = vaddq_f32(sum, vld1q_f32(BiasPtr));
        }

        vst1q_f32(SumPtr, sum);
    } else {
        for (int i = 0; i < NCols; ++i) {
            SumPtr[i] = vaddvq_f32(acc[i]);
            if (BiasPtr != nullptr) {
                SumPtr[i] += BiasPtr[i];
            }
        }
    }
}

}  // namespace q4gemm_neon

template <typename Q4Type>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernelNeon(const float* A,
                     const uint8_t* PackedB,
                     float* C,
                     size_t CountM,
                     size_t CountN,
                     size_t CountK,
                     size_t lda,
                     size_t ldb,
                     size_t ldc,
                     const float* Bias)
{
    static constexpr size_t NCols = 4;  // columns to handle at once

    auto impl0_reference = [&]() {
        for (size_t m = 0; m < CountM; ++m) {
            for (size_t n = 0; n < CountN; ++n) {
                const uint8_t* PackedBBlock = PackedB + n * ldb;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    float b_blk_unpacked[Q4Type::BlkLen]{};

                    const size_t kblocklen = std::min(CountK - k, Q4Type::BlkLen);

                    const float s = MlasQ4BlkScale<Q4Type>(PackedBBlock);
                    const uint8_t z = [PackedBBlock]() -> uint8_t {
                        if constexpr (MlasQ4BlkHasZeroPoint<Q4Type>()) {
                            return MlasQ4BlkZeroPoint<Q4Type>(PackedBBlock);
                        } else {
                            return 8;
                        }
                    }();
                    const uint8_t* PackedBData = MlasQ4BlkData<Q4Type>(PackedBBlock);

                    for (size_t kk = 0; kk < kblocklen; kk += 32) {
                        const size_t ksubblocklen = std::min(size_t{32}, kblocklen - kk);

                        for (size_t l0 = 0; l0 < 16; ++l0) {
                            const uint8_t PackedByte = PackedBData[l0];

                            if (l0 < ksubblocklen) {
                                const int8_t PackedByteLo = PackedByte & 0x0F;
                                const float UnpackedValue0 = (PackedByteLo - z) * s;
                                b_blk_unpacked[kk + l0] = UnpackedValue0;
                            }

                            const size_t l1 = l0 + 16;
                            if (l1 < ksubblocklen) {
                                const int8_t PackedByteHi = PackedByte >> 4;
                                const float UnpackedValue1 = (PackedByteHi - z) * s;
                                b_blk_unpacked[kk + l1] = UnpackedValue1;
                            }
                        }

                        PackedBData += 16;
                    }

                    for (size_t kk = 0; kk < kblocklen; ++kk) {
                        C[m * ldc + n] += A[m * lda + k + kk] * b_blk_unpacked[kk];
                    }

                    PackedBBlock += Q4Type::BlobSize;
                }

                if (Bias) {
                    C[m * ldc + n] += Bias[n];
                }
            }
        }

        return CountM;
    };

    auto impl3_four_cols = [&]() {
        const float* ARowPtr = A;
        float* CRowPtr = C;

        for (size_t m = 0; m < CountM; ++m) {
            const float* BiasPtr = Bias;
            const uint8_t* PackedBColPtr = PackedB;
            float* SumPtr = CRowPtr;

            int64_t nblk = static_cast<int64_t>(CountN) - NCols;

            while (nblk >= 0) {
                q4gemm_neon::ComputeDotProducts<Q4Type, NCols>(ARowPtr, PackedBColPtr, SumPtr,
                                                               CountK, ldb, BiasPtr);

                // move to next `NCols` columns

                PackedBColPtr += NCols * ldb;
                BiasPtr += BiasPtr != nullptr ? NCols : 0;
                SumPtr += NCols;

                nblk -= NCols;
            }

            // left over columns less than `NCols`?
            nblk += NCols;
            for (int64_t n = 0; n < nblk; ++n) {
                q4gemm_neon::ComputeDotProducts<Q4Type, 1>(ARowPtr, PackedBColPtr, SumPtr, CountK,
                                                           ldb, BiasPtr);

                PackedBColPtr += ldb;
                BiasPtr += BiasPtr != nullptr ? 1 : 0;
                SumPtr += 1;
            }

            ARowPtr += lda;
            CRowPtr += ldc;
        }

        return CountM;
    };

    auto impl5_four_cols_inline = [&]() {
        const float* ARowPtr = A;
        float* CRowPtr = C;

        const int8x16_t LowMask = vdupq_n_s8(0x0F);

        for (size_t m = 0; m < CountM; ++m) {
            const uint8_t* PackedBColPtr = PackedB;
            const float* BiasPtr = Bias;
            float* SumPtr = CRowPtr;

            int64_t nblk = CountN - NCols;

            while (nblk >= 0) {
                float32x4_t acc[NCols]{};

                const uint8_t* PackedBBlobPtr = PackedBColPtr;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    const size_t k_blk_len = std::min(CountK - k, Q4Type::BlkLen);

                    float scale[NCols];
                    q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                        scale[i] = MlasQ4BlkScale<Q4Type>(PackedBBlobPtr + i * ldb);
                    });

                    uint8_t zp[NCols];
                    q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                        if constexpr (MlasQ4BlkHasZeroPoint<Q4Type>()) {
                            zp[i] = MlasQ4BlkZeroPoint<Q4Type>(PackedBBlobPtr + i * ldb);
                        } else {
                            zp[i] = 8;
                        }
                    });

                    const uint8_t* b_data[NCols];
                    q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                        b_data[i] = MlasQ4BlkData<Q4Type>(PackedBBlobPtr + i * ldb);
                    });

                    for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += 32) {
                        // load A row vector elements

                        // load 32 elements from A padding with 0's if there aren't enough
                        float a_segment[32];
                        {
                            const size_t k_subblk_len =
                                std::min(k_blk_len - k_idx_in_blk, size_t{32});
                            const float* a_begin = ARowPtr + k + k_idx_in_blk;
                            std::copy(a_begin, a_begin + k_subblk_len, a_segment);
                            std::fill(a_segment + k_subblk_len, a_segment + 32, 0.0f);
                        }

                        // 32 floats of A
                        float32x4_t av[8];
                        q4gemm_neon::UnrolledLoop<8>(
                            [&](size_t i) { av[i] = vld1q_f32(a_segment + i * 4); });

                        // load B column vector
                        int8x16_t bv_packed[NCols];
                        q4gemm_neon::UnrolledLoop<NCols>(
                            [&](size_t i) { bv_packed[i] = vld1q_s8(b_data[i]); });

                        int8x16_t bv_bytes[NCols][2];
                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            bv_bytes[i][0] = vandq_s8(bv_packed[i], LowMask);
                            bv_bytes[i][1] = vandq_s8(vshrq_n_s8(bv_packed[i], 4), LowMask);
                        });

                        // dequantize B

                        // subtract zero point
                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            const int8x16_t zpv = vdupq_n_s8(zp[i]);

                            bv_bytes[i][0] = vsubq_s8(bv_bytes[i][0], zpv);
                            bv_bytes[i][1] = vsubq_s8(bv_bytes[i][1], zpv);
                        });

                        // widen to int16
                        int16x8_t bv_int16[NCols][4];
                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            bv_int16[i][0] = vmovl_s8(vget_low_s8(bv_bytes[i][0]));
                            bv_int16[i][1] = vmovl_s8(vget_high_s8(bv_bytes[i][0]));
                            bv_int16[i][2] = vmovl_s8(vget_low_s8(bv_bytes[i][1]));
                            bv_int16[i][3] = vmovl_s8(vget_high_s8(bv_bytes[i][1]));
                        });

                        // 32 floats of B
                        float32x4_t bv[NCols][8];

                        // widen to int32, cast to float32

                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            bv[i][0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][0])));
                            bv[i][1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][0])));

                            bv[i][2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][1])));
                            bv[i][3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][1])));
                        });

                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            bv[i][4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][2])));
                            bv[i][5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][2])));

                            bv[i][6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[i][3])));
                            bv[i][7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[i][3])));
                        });

                        // multiply by scale
                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            q4gemm_neon::UnrolledLoop<8>(
                                [&](size_t j) { bv[i][j] = vmulq_n_f32(bv[i][j], scale[i]); });
                        });

                        // c += a * b
                        q4gemm_neon::UnrolledLoop<8>([&](size_t j) {
                            q4gemm_neon::UnrolledLoop<NCols>(
                                [&](size_t i) { acc[i] = vfmaq_f32(acc[i], av[j], bv[i][j]); });
                        });

                        // increment b data pointers to next 32 elements
                        q4gemm_neon::UnrolledLoop<NCols>([&](size_t i) {
                            b_data[i] += 16;
                        });
                    }

                    PackedBBlobPtr += Q4Type::BlobSize;
                }

                float32x4_t sums = q4gemm_neon::FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);

                if (Bias) {
                    sums = vaddq_f32(sums, vld1q_f32(BiasPtr));
                }

                vst1q_f32(SumPtr, sums);

                PackedBColPtr += NCols * ldb;
                BiasPtr += NCols;
                SumPtr += NCols;

                nblk -= NCols;
            }

            nblk += NCols;

            if (nblk > 0) {
                float32x4_t acc[NCols]{};

                const uint8_t* PackedBBlobPtr = PackedBColPtr;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    const size_t k_blk_len = std::min(CountK - k, Q4Type::BlkLen);

                    float scale[NCols];
                    const uint8_t* b_data[NCols];
                    uint8_t zp[NCols];

                    for (int64_t nn = 0; nn < nblk; ++nn) {
                        scale[nn] = MlasQ4BlkScale<Q4Type>(PackedBBlobPtr + nn * ldb);
                        b_data[nn] = MlasQ4BlkData<Q4Type>(PackedBBlobPtr + nn * ldb);
                        if constexpr (MlasQ4BlkHasZeroPoint<Q4Type>()) {
                            zp[nn] = MlasQ4BlkZeroPoint<Q4Type>(PackedBBlobPtr + nn * ldb);
                        } else {
                            zp[nn] = 8;
                        }
                    }

                    for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += 32) {
                        // load A row vector elements

                        // load 32 elements from A padding with 0's if there aren't enough
                        float a_segment[32];
                        {
                            const size_t k_subblk_len =
                                std::min(k_blk_len - k_idx_in_blk, size_t{32});
                            const float* a_begin = ARowPtr + k + k_idx_in_blk;
                            std::copy(a_begin, a_begin + k_subblk_len, a_segment);
                            std::fill(a_segment + k_subblk_len, a_segment + 32, 0.0f);
                        }

                        // 32 floats of A
                        float32x4_t av[8];
                        q4gemm_neon::UnrolledLoop<8>(
                            [&](size_t i) { av[i] = vld1q_f32(a_segment + i * 4); });

                        for (int64_t nn = 0; nn < nblk; ++nn) {
                            // load B column vector
                            int8x16_t bv_packed = vld1q_s8(b_data[nn]);

                            int8x16_t bv_bytes[2];
                            bv_bytes[0] = vandq_s8(bv_packed, LowMask);
                            bv_bytes[1] = vandq_s8(vshrq_n_s8(bv_packed, 4), LowMask);

                            // dequantize B

                            // subtract zero point
                            const int8x16_t zpv = vdupq_n_s8(zp[nn]);

                            bv_bytes[0] = vsubq_s8(bv_bytes[0], zpv);
                            bv_bytes[1] = vsubq_s8(bv_bytes[1], zpv);

                            // widen to int16
                            int16x8_t bv_int16[4];
                            bv_int16[0] = vmovl_s8(vget_low_s8(bv_bytes[0]));
                            bv_int16[1] = vmovl_s8(vget_high_s8(bv_bytes[0]));
                            bv_int16[2] = vmovl_s8(vget_low_s8(bv_bytes[1]));
                            bv_int16[3] = vmovl_s8(vget_high_s8(bv_bytes[1]));

                            // 32 floats of B
                            float32x4_t bv[8];

                            // widen to int32, cast to float32
                            bv[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[0])));
                            bv[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[0])));

                            bv[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[1])));
                            bv[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[1])));

                            bv[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[2])));
                            bv[5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[2])));

                            bv[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16[3])));
                            bv[7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16[3])));

                            // multiply by scale
                            q4gemm_neon::UnrolledLoop<8>(
                                [&](size_t j) { bv[j] = vmulq_n_f32(bv[j], scale[nn]); });

                            // c += a * b
                            q4gemm_neon::UnrolledLoop<8>(
                                [&](size_t j) { acc[nn] = vfmaq_f32(acc[nn], av[j], bv[j]); });

                            // increment b data pointers to next 32 elements
                            b_data[nn] += 16;
                        }
                    }

                    PackedBBlobPtr += Q4Type::BlobSize;
                }

                for (int64_t nn = 0; nn < nblk; ++nn) {
                    SumPtr[nn] = vaddvq_f32(acc[nn]);
                    SumPtr[nn] += Bias != nullptr ? BiasPtr[nn] : 0.0f;
                }
            }

            ARowPtr += lda;
            CRowPtr += ldc;
        }

        return CountM;
    };

    // return impl0_reference();
    // return impl3_four_cols();
    // return impl5_four_cols_inline();
}

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>(const float* A,
                                                                const uint8_t* PackedB,
                                                                float* C,
                                                                size_t CountM,
                                                                size_t CountN,
                                                                size_t CountK,
                                                                size_t lda,
                                                                size_t ldb,
                                                                size_t ldc,
                                                                const float* Bias)
{
    return MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK0>(A, PackedB, C, CountM, CountN, CountK, lda, ldb,
                                                  ldc, Bias);
}

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_NEON>(const float* A,
                                                                const uint8_t* PackedB,
                                                                float* C,
                                                                size_t CountM,
                                                                size_t CountN,
                                                                size_t CountK,
                                                                size_t lda,
                                                                size_t ldb,
                                                                size_t ldc,
                                                                const float* Bias)
{
    return MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK1>(A, PackedB, C, CountM, CountN, CountK, lda, ldb,
                                                  ldc, Bias);
}

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_NEON>(const float* A,
                                                                const uint8_t* PackedB,
                                                                float* C,
                                                                size_t CountM,
                                                                size_t CountN,
                                                                size_t CountK,
                                                                size_t lda,
                                                                size_t ldb,
                                                                size_t ldc,
                                                                const float* Bias)
{
    return MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK2>(A, PackedB, C, CountM, CountN, CountK, lda, ldb,
                                                  ldc, Bias);
}

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_NEON>(const float* A,
                                                                const uint8_t* PackedB,
                                                                float* C,
                                                                size_t CountM,
                                                                size_t CountN,
                                                                size_t CountK,
                                                                size_t lda,
                                                                size_t ldb,
                                                                size_t ldc,
                                                                const float* Bias)
{
    return MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK4>(A, PackedB, C, CountM, CountN, CountK, lda, ldb,
                                                  ldc, Bias);
}

//
// MlasBlkQ4DequantB and related helper functions
//

template <typename Q4Type>
MLAS_FORCEINLINE void
MlasBlkQ4DequantBNeon(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    // unpack B in format suitable for MlasSgemmKernelZero

    auto impl0_reference = [&]() {
        float* Dst = FpData;
        const uint8_t* PackedBCol = PackedB;

        for (size_t n = 0; n < CountN; n += 16) {
            const size_t nnlen = std::min(CountN - n, size_t{16});

            for (size_t nn = 0; nn < nnlen; ++nn) {
                const uint8_t* PackedBBlock = PackedBCol;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    float b_blk_unpacked[Q4Type::BlkLen]{};

                    const size_t kblocklen = std::min(CountK - k, Q4Type::BlkLen);

                    const float s = MlasQ4BlkScale<Q4Type>(PackedBBlock);
                    const uint8_t z = [PackedBBlock]() -> uint8_t {
                        if constexpr (MlasQ4BlkHasZeroPoint<Q4Type>()) {
                            return MlasQ4BlkZeroPoint<Q4Type>(PackedBBlock);
                        } else {
                            return 8;
                        }
                    }();
                    const uint8_t* PackedBData = MlasQ4BlkData<Q4Type>(PackedBBlock);

                    for (size_t kk = 0; kk < kblocklen; kk += 32) {
                        const size_t ksubblocklen = std::min(size_t{32}, kblocklen - kk);

                        for (size_t l0 = 0; l0 < 16; ++l0) {
                            const uint8_t PackedByte = PackedBData[l0];

                            if (l0 < ksubblocklen) {
                                const int8_t PackedByteLo = PackedByte & 0x0F;
                                const float UnpackedValue0 = (PackedByteLo - z) * s;
                                b_blk_unpacked[kk + l0] = UnpackedValue0;
                            }

                            const size_t l1 = l0 + 16;
                            if (l1 < ksubblocklen) {
                                const int8_t PackedByteHi = PackedByte >> 4;
                                const float UnpackedValue1 = (PackedByteHi - z) * s;
                                b_blk_unpacked[kk + l1] = UnpackedValue1;
                            }
                        }

                        PackedBData += 16;
                    }

                    for (size_t kk = 0; kk < kblocklen; ++kk) {
                        Dst[(k + kk) * 16 + nn] = b_blk_unpacked[kk];
                    }

                    PackedBBlock += Q4Type::BlobSize;
                }

                PackedBCol += ldb;
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

    // TODO optimized implementation
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK0>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK1>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK2>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK4>(FpData, PackedB, CountN, CountK, ldb);
}

//
// MlasFpQ4GemmDispatchNeon structure population
//

static MLAS_Q4GEMM_OPERATION* Q4Operations_neon[] = {
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
    nullptr,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
};

const MLAS_FPQ4GEMM_DISPATCH MlasFpQ4GemmDispatchNeon = {Q4Operations_neon};
