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
    // static constexpr size_t StrideM = 256;
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

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK0>(const float* A,
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
    using Q4Type = MLAS_Q4TYPE_BLK0;

    auto impl0_reference = [&]() {
        for (size_t m = 0; m < CountM; ++m) {
            for (size_t n = 0; n < CountN; ++n) {
                const uint8_t* PackedBBlock = PackedB + n * ldb;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    float b_blk_unpacked[Q4Type::BlkLen]{};

                    const size_t kblocklen = std::min(CountK - k, Q4Type::BlkLen);

                    const float s = MlasQ4BlkScale<Q4Type>(PackedBBlock);
                    const uint8_t z = 8;  // MlasQ4BlkZeroPoint<Q4Type>(PackedBBlock);
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

    auto impl1 = [&]() {
        const float* ARowPtr = A;
        float* CRowPtr = C;
        const float* BiasPtr = Bias;

        const int8x16_t LowMask = vdupq_n_s8(0x0F);

        for (size_t m = 0; m < CountM; ++m) {
            const uint8_t* PackedBColPtr = PackedB;

            for (size_t n = 0; n < CountN; ++n) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                const uint8_t* PackedBBlobPtr = PackedBColPtr;

                for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                    const size_t k_blk_len = std::min(CountK - k, Q4Type::BlkLen);

                    const float scale = MlasQ4BlkScale<Q4Type>(PackedBBlobPtr);
                    const uint8_t zp = 8;
                    const uint8_t* b_data = MlasQ4BlkData<Q4Type>(PackedBBlobPtr);

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
                        for (int i = 0; i < 8; ++i) {
                            av[i] = vld1q_f32(a_segment + i * 4);
                        }

                        // load B column vector
                        int8x16_t bv_packed = vld1q_s8(b_data);

                        int8x16_t bv_bytes_0 = vandq_s8(bv_packed, LowMask);
                        int8x16_t bv_bytes_1 = vandq_s8(vshrq_n_s8(bv_packed, 4), LowMask);

                        // dequantize B

                        // subtract zero point
                        const int8x16_t zpv = vdupq_n_s8(zp);

                        bv_bytes_0 = vsubq_s8(bv_bytes_0, zpv);
                        bv_bytes_1 = vsubq_s8(bv_bytes_1, zpv);

                        // widen to int16
                        int16x8_t bv_int16_0 = vmovl_s8(vget_low_s8(bv_bytes_0));
                        int16x8_t bv_int16_1 = vmovl_s8(vget_high_s8(bv_bytes_0));
                        int16x8_t bv_int16_2 = vmovl_s8(vget_low_s8(bv_bytes_1));
                        int16x8_t bv_int16_3 = vmovl_s8(vget_high_s8(bv_bytes_1));

                        // 32 floats of B
                        float32x4_t bv[8];

                        // widen to int32, cast to float32

                        int32x4_t bv_int32_0 = vmovl_s16(vget_low_s16(bv_int16_0));

                        bv[0] = vcvtq_f32_s32(bv_int32_0);
                        bv[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16_0)));

                        bv[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16_1)));
                        bv[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16_1)));

                        bv[4] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16_2)));
                        bv[5] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16_2)));

                        bv[6] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(bv_int16_3)));
                        bv[7] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(bv_int16_3)));

                        // multiply by scale
                        for (int i = 0; i < 8; ++i) {
                            bv[i] = vmulq_n_f32(bv[i], scale);
                        }

                        // c += a * b
                        for (int i = 0; i < 8; ++i) {
                            acc = vfmaq_f32(acc, av[i], bv[i]);
                        }
                    }

                    PackedBBlobPtr += Q4Type::BlobSize;
                }

                float sum = vpadds_f32(vpadd_f32(vget_low_f32(acc), vget_high_f32(acc)));

                sum += BiasPtr ? BiasPtr[n] : 0.0f;

                CRowPtr[n] = sum;

                PackedBColPtr += ldb;
            }

            ARowPtr += lda;
            CRowPtr += ldc;
        }

        return CountM;
    };

    // return impl0_reference();
    return impl1();
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

//
// MlasBlkQ4DequantB and related helper functions
//

template <typename Q4Type>
MLAS_FORCEINLINE void
MlasBlkQ4DequantBNeon(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb);

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK0>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    using Q4Type = MLAS_Q4TYPE_BLK0;
    static_cast<void>(FpData, PackedB, CountN, CountK, ldb);
    // TODO ...
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK0>(FpData, PackedB, CountN, CountK, ldb);
}

//
// MlasFpQ4GemmDispatchNeon structure population
//

static MLAS_Q4GEMM_OPERATION* Q4Operations_neon[] = {
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

const MLAS_FPQ4GEMM_DISPATCH MlasFpQ4GemmDispatchNeon = {Q4Operations_neon};
