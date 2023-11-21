/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication hardware agnostic entrypoint, MlasSQNBitGemmBatch.
--*/

#include <iostream>

#include "sqnbitgemm.h"

namespace
{

// Get quantization variant based on `BlkBitWidth` and `BlkLen`.
// Return -1 if the input values are unsupported.
int32_t
GetDispatchQuantVariant(size_t BlkBitWidth, size_t BlkLen)
{
    int32_t type = -1;
    if (BlkBitWidth == 4 && BlkLen == 16) {
        type = QuantVariant_BitWidth4_BlockSize16;
    } else if (BlkBitWidth == 4 && BlkLen == 32) {
        type = QuantVariant_BitWidth4_BlockSize32;
    } else if (BlkBitWidth == 4 && BlkLen == 64) {
        type = QuantVariant_BitWidth4_BlockSize64;
    } else if (BlkBitWidth == 4 && BlkLen == 128) {
        type = QuantVariant_BitWidth4_BlockSize128;
    } else if (BlkBitWidth == 4 && BlkLen == 256) {
        type = QuantVariant_BitWidth4_BlockSize256;
    }

    return type;
}

}  // namespace

#ifdef MLAS_TARGET_ARM_ANY
void MLASCALL
MlasSQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
)
{
    const int32_t QuantVariant = GetDispatchQuantVariant(BlkBitWidth, BlkLen);
    MLAS_SQNBIT_GEMM_OPERATION* const Operation = GetMlasPlatform().SQNBitGemmDispatch->Operations[QuantVariant];

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            Operation(K, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    constexpr size_t StrideM = 128;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(
                nc, MlasDivRoundup(max_nc, MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                        MLAS_QGEMM_STRIDEN_THREAD_ALIGN
            );
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        Operation(K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}
#else

//
// Define the number of rows from matrix B to transpose to a local buffer.
//
// N.B. AVX processes a maximum of 4 rows, FMA3 processes a maximum of 6
// rows, and AVX512F processes a maximum of 12 rows.
//

#define MLAS_SGEMM_TRANSA_ROWS 12

template <unsigned N>
inline void
MlasSgemmTransposePackBNx32(
    float* D,
    const uint8_t* B,
    size_t ldb,
    const float* scale,
    const uint8_t* zero_point
)
{
    MLAS_INT32X4 LowMask = _mm_set1_epi8(0x0F);
    for (unsigned n = 0; n < N / 4; n++) {
#if defined(MLAS_NEON_INTRINSICS)
        float32x4x2_t z0 = vzipq_f32(t0, t2);
        float32x4x2_t z1 = vzipq_f32(t1, t3);
        float32x4x2_t o0 = vzipq_f32(z0.val[0], z1.val[0]);
        float32x4x2_t o1 = vzipq_f32(z0.val[1], z1.val[1]);
        t0 = o0.val[0];
        t1 = o0.val[1];
        t2 = o1.val[0];
        t3 = o1.val[1];
#else
        MLAS_INT32X4 t0 = MlasLoadInt32x4((const int32_t*)&B[ldb * 0]);
        MLAS_INT32X4 t1 = MlasLoadInt32x4((const int32_t*)&B[ldb * 1]);
        MLAS_INT32X4 t2 = MlasLoadInt32x4((const int32_t*)&B[ldb * 2]);
        MLAS_INT32X4 t3 = MlasLoadInt32x4((const int32_t*)&B[ldb * 3]);

        MLAS_INT32X4 zero_point_0 = _mm_set1_epi8(zero_point[n * 4]);
        MLAS_INT32X4 zero_point_1 = _mm_set1_epi8(zero_point[n * 4 + 1]);
        MLAS_INT32X4 zero_point_2 = _mm_set1_epi8(zero_point[n * 4 + 2]);
        MLAS_INT32X4 zero_point_3 = _mm_set1_epi8(zero_point[n * 4 + 3]);

        __m512 scale_0 = _mm512_set1_ps(scale[n * 4]);
        __m512 scale_1 = _mm512_set1_ps(scale[n * 4 + 1]);
        __m512 scale_2 = _mm512_set1_ps(scale[n * 4 + 2]);
        __m512 scale_3 = _mm512_set1_ps(scale[n * 4 + 3]);

        MLAS_INT32X4 t0_low = MlasAndInt32x4(t0, LowMask);  // [0, 2, 4, 6, ..., 30]
        MLAS_INT32X4 t1_low = MlasAndInt32x4(t1, LowMask);
        MLAS_INT32X4 t2_low = MlasAndInt32x4(t2, LowMask);
        MLAS_INT32X4 t3_low = MlasAndInt32x4(t3, LowMask);

        t0 = MlasAndInt32x4(_mm_srli_epi32(t0, 4), LowMask);  // [1, 3, 5, 7, ..., 31]
        t1 = MlasAndInt32x4(_mm_srli_epi32(t1, 4), LowMask);
        t2 = MlasAndInt32x4(_mm_srli_epi32(t2, 4), LowMask);
        t3 = MlasAndInt32x4(_mm_srli_epi32(t3, 4), LowMask);

        // subtract zero point
        t0_low = _mm_sub_epi8(t0_low, zero_point_0);
        t1_low = _mm_sub_epi8(t1_low, zero_point_1);
        t2_low = _mm_sub_epi8(t2_low, zero_point_2);
        t3_low = _mm_sub_epi8(t3_low, zero_point_3);
        t0 = _mm_sub_epi8(t0, zero_point_0);
        t1 = _mm_sub_epi8(t1, zero_point_1);
        t2 = _mm_sub_epi8(t2, zero_point_2);
        t3 = _mm_sub_epi8(t3, zero_point_3);

        __m512i t0_ps_low = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t0_low))), scale_0));
        __m512i t1_ps_low = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t1_low))), scale_1));
        __m512i t2_ps_low = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t2_low))), scale_2));
        __m512i t3_ps_low = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t3_low))), scale_3));
        __m512i t0_ps = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t0))), scale_0));
        __m512i t1_ps = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t1))), scale_1));
        __m512i t2_ps = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t2))), scale_2));
        __m512i t3_ps = _mm512_castps_si512(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_cvtepi8_epi16(t3))), scale_3));

        __m512i t0_t1_0_2_8_10_16_18_24_26 = _mm512_unpacklo_epi32(t0_ps_low, t1_ps_low);
        __m512i t2_t3_0_2_8_10_16_18_24_26 = _mm512_unpacklo_epi32(t2_ps_low, t3_ps_low);
        __m512i t0_t1_4_6_12_14_20_22_28_30 = _mm512_unpackhi_epi32(t0_ps_low, t1_ps_low);
        __m512i t2_t3_4_6_12_14_20_22_28_30 = _mm512_unpackhi_epi32(t2_ps_low, t3_ps_low);
        __m512i t0_t1_t2_t3_0_8_16_24 = _mm512_unpacklo_epi64(t0_t1_0_2_8_10_16_18_24_26, t2_t3_0_2_8_10_16_18_24_26);
        __m512i t0_t1_t2_t3_2_10_18_26 = _mm512_unpackhi_epi64(t0_t1_0_2_8_10_16_18_24_26, t2_t3_0_2_8_10_16_18_24_26);
        __m512i t0_t1_t2_t3_4_12_20_28 = _mm512_unpacklo_epi64(t0_t1_4_6_12_14_20_22_28_30, t2_t3_4_6_12_14_20_22_28_30);
        __m512i t0_t1_t2_t3_6_14_22_30 = _mm512_unpackhi_epi64(t0_t1_4_6_12_14_20_22_28_30, t2_t3_4_6_12_14_20_22_28_30);

        __m512i t0_t1_1_3_9_11_17_19_25_27 = _mm512_unpacklo_epi32(t0_ps, t1_ps);
        __m512i t2_t3_1_3_9_11_17_19_25_27 = _mm512_unpacklo_epi32(t2_ps, t3_ps);
        __m512i t0_t1_5_7_13_15_21_23_29_31 = _mm512_unpackhi_epi32(t0_ps, t1_ps);
        __m512i t2_t3_5_7_13_15_21_23_29_31 = _mm512_unpackhi_epi32(t2_ps, t3_ps);
        __m512i t0_t1_t2_t3_1_9_17_25 = _mm512_unpacklo_epi64(t0_t1_1_3_9_11_17_19_25_27, t2_t3_1_3_9_11_17_19_25_27);
        __m512i t0_t1_t2_t3_3_11_19_27 = _mm512_unpackhi_epi64(t0_t1_1_3_9_11_17_19_25_27, t2_t3_1_3_9_11_17_19_25_27);
        __m512i t0_t1_t2_t3_5_13_21_29 = _mm512_unpacklo_epi64(t0_t1_5_7_13_15_21_23_29_31, t2_t3_5_7_13_15_21_23_29_31);
        __m512i t0_t1_t2_t3_7_15_23_31 = _mm512_unpackhi_epi64(t0_t1_5_7_13_15_21_23_29_31, t2_t3_5_7_13_15_21_23_29_31);

#define store4x4(zmm, base_offset)                                                              \
    MlasStoreInt32x4((int32_t*)(&D[base_offset]), _mm512_extracti32x4_epi32(zmm, 0));           \
    MlasStoreInt32x4((int32_t*)(&D[base_offset + 8 * 16]), _mm512_extracti32x4_epi32(zmm, 1));  \
    MlasStoreInt32x4((int32_t*)(&D[base_offset + 16 * 16]), _mm512_extracti32x4_epi32(zmm, 2)); \
    MlasStoreInt32x4((int32_t*)(&D[base_offset + 24 * 16]), _mm512_extracti32x4_epi32(zmm, 3));

        store4x4(t0_t1_t2_t3_0_8_16_24, 0);
        store4x4(t0_t1_t2_t3_1_9_17_25, 1 * 16);
        store4x4(t0_t1_t2_t3_2_10_18_26, 2 * 16);
        store4x4(t0_t1_t2_t3_3_11_19_27, 3 * 16);
        store4x4(t0_t1_t2_t3_4_12_20_28, 4 * 16);
        store4x4(t0_t1_t2_t3_5_13_21_29, 5 * 16);
        store4x4(t0_t1_t2_t3_6_14_22_30, 6 * 16);
        store4x4(t0_t1_t2_t3_7_15_23_31, 7 * 16);
#undef store4x4
        D += 4;
        B += ldb * 4;
#endif
    }
}

void
MlasSgemmNBitsTransposePackB(
    float* D,
    const uint8_t* B,
    size_t ldb,
    const float* QuantBScale,
    size_t ld_scale,
    const uint8_t* QuantBZeroPoint,
    size_t ld_zp,
    size_t BlkLen,
    size_t CountN,
    size_t CountK,
    size_t CurrentK
)
{
    //
    // Transpose elements from matrix B into the packed buffer 16 rows at a
    // time.
    //

    while (CountN >= 16) {
        const uint8_t* b = B;
        size_t k = CountK;
        size_t ck = CurrentK;

// #if defined(MLAS_TARGET_AMD64)
#if 0

        MLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE* SgemmTransposePackB16x4Routine =
            GetMlasPlatform().TransposePackB16x4Routine;

        while (k >= 4) {
            SgemmTransposePackB16x4Routine(&D[0], &b[0], ldb);

            D += 16 * 4;
            b += 4;
            k -= 4;
        }

#else

        while (k >= 32) {
            float scale[16];
            uint8_t zero_point[16];
            for (int i = 0; i < 16; i++) {
                size_t block_id = ck / BlkLen;
                scale[i] = QuantBScale[i * ld_scale + block_id];
                if (QuantBZeroPoint != nullptr) {
                    zero_point[i] = QuantBZeroPoint[i * ld_zp + block_id / 2];
                    zero_point[i] = (block_id & 1) ? (zero_point[i] >> 4) : (zero_point[i] & 0x0F);
                } else {
                    zero_point[i] = 8;
                }
            }
            MlasSgemmTransposePackBNx32<16>(&D[0], b, ldb, scale, zero_point);

            D += 16 * 32;
            b += 32;
            k -= 32;
            ck += 32;
        }

#endif
        /*
        while (k > 0) {
            float t0 = b[0];
            float t1 = b[ldb];
            float t2 = b[ldb * 2];
            float t3 = b[ldb * 3];
            float t4 = b[ldb * 4];
            float t5 = b[ldb * 5];
            float t6 = b[ldb * 6];
            float t7 = b[ldb * 7];
            float t8 = b[ldb * 8];
            float t9 = b[ldb * 9];
            float t10 = b[ldb * 10];
            float t11 = b[ldb * 11];
            float t12 = b[ldb * 12];
            float t13 = b[ldb * 13];
            float t14 = b[ldb * 14];
            float t15 = b[ldb * 15];

            D[0] = t0;
            D[1] = t1;
            D[2] = t2;
            D[3] = t3;
            D[4] = t4;
            D[5] = t5;
            D[6] = t6;
            D[7] = t7;
            D[8] = t8;
            D[9] = t9;
            D[10] = t10;
            D[11] = t11;
            D[12] = t12;
            D[13] = t13;
            D[14] = t14;
            D[15] = t15;

            D += 16;
            b += 1;
            k--;
        }
        */

        B += ldb * 16;
        CountN -= 16;
        QuantBScale += ld_scale * 16;
        if (QuantBZeroPoint) {
            QuantBZeroPoint += ld_zp * 16;
        }

    }

    //
    // Special case the handling of the less than 16 remaining rows.
    //
    /*
    if (CountN > 0) {
        MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

        size_t x = CountK;

        //
        // Transpose 4 columns at a time.
        //

        while (x >= 4) {
            float* d = D;
            const uint8_t* b = B;

            if ((CountN & 8) != 0) {
                MlasSgemmTransposePackBNx4<8>(&d[0], &b[0], ldb);

                d += 8;
                b += ldb * 8;

            } else {
                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[24], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[28], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[40], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[44], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[56], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[60], ZeroFloat32x4);
            }

            if ((CountN & 4) != 0) {
                MlasSgemmTransposePackBNx4<4>(&d[0], &b[0], ldb);

                d += 4;
                b += ldb * 4;

            } else {
                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[20], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[36], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[52], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(&d[0], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[16], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[32], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[48], ZeroFloat32x4);

            if ((CountN & 2) != 0) {
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[ldb]);

#if defined(MLAS_SSE2_INTRINSICS)
                __m128 v0 = _mm_unpacklo_ps(t0, t1);
                __m128 v1 = _mm_unpackhi_ps(t0, t1);
                _mm_storel_pi((__m64*)&d[0], v0);
                _mm_storeh_pi((__m64*)&d[16], v0);
                _mm_storel_pi((__m64*)&d[32], v1);
                _mm_storeh_pi((__m64*)&d[48], v1);
#else
                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<0>(&d[1], t1);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<1>(&d[17], t1);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<2>(&d[33], t1);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
                MlasStoreLaneFloat32x4<3>(&d[49], t1);
#endif

                d += 2;
                b += ldb * 2;
            }

            if ((CountN & 1) != 0) {

#if defined(MLAS_NEON_INTRINSICS)
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);

                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
#else
                d[0] = b[0];
                d[16] = b[1];
                d[32] = b[2];
                d[48] = b[3];
#endif
            }

            D += 16 * 4;
            B += 4;
            x -= 4;
        }

        //
        // Transpose the remaining columns.
        //

        while (x > 0) {
            float* d = D;
            const uint8_t* b = B;

            if ((CountN & 8) != 0) {
                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];
                float t4 = b[ldb * 4];
                float t5 = b[ldb * 5];
                float t6 = b[ldb * 6];
                float t7 = b[ldb * 7];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;
                d[4] = t4;
                d[5] = t5;
                d[6] = t6;
                d[7] = t7;

                d += 8;
                b += ldb * 8;

            } else {
                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
            }

            if ((CountN & 4) != 0) {
                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;

                d += 4;
                b += ldb * 4;

            } else {
                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(d, ZeroFloat32x4);

            if ((CountN & 2) != 0) {
                float t0 = b[0];
                float t1 = b[ldb];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += ldb * 2;
            }

            if ((CountN & 1) != 0) {
                d[0] = b[0];
            }

            D += 16;
            B += 1;
            x--;
        }
    }
    */
}

MLAS_FORCEINLINE
float*
MlasSgemmKernelLoopNBits(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
)
/*++

Routine Description:

    This routine steps through the rows of the input and output matrices calling
    the kernel until all rows have been processed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the number of rows from matrix A and matrix C to iterate
        over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the next address of matrix C.

--*/
{
    while (CountM > 0) {
        size_t RowsHandled;

#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER)
        RowsHandled = GetMlasPlatform().GemmFloatKernel(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);
#else
        if (ZeroMode) {
            RowsHandled = MlasSgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        } else {
            RowsHandled = MlasSgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        }
#endif

        C += ldc * RowsHandled;
        A += lda * RowsHandled;
        CountM -= RowsHandled;
    }

    return C;
}

void
MlasSgemmNBitsOperation(
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN,
    const size_t BlkBitWidth,
    const size_t BlkLen
)
{
    MLAS_DECLSPEC_ALIGN(float PanelB[MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK], 16 * sizeof(float));

    const size_t M = RangeCountM;
    const size_t N = RangeCountN;

    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t k_blks_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes(BlkBitWidth, k_blks);

    const float* A = DataParams->A + RangeStartM * lda;

    const uint8_t* QuantBData = static_cast<const uint8_t*>(DataParams->QuantBData) + RangeStartN * ldb;
    const float* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blks;
    const uint8_t* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const uint8_t*>(DataParams->QuantBZeroPoint) + RangeStartN * k_blks_zp_bytes;

    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    const float* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias + RangeStartN;
    (void)Bias;
    (void)QuantBData;
    (void)QuantBScale;
    (void)QuantBZeroPoint;

    //
    // Handle the special case of a small M. The data from matrix B is not
    // referenced multiple times, so using a local packed buffer is a wasted
    // memory copy.
    //

    // TODO:
    if (M == 1) {
        return;
    }

    // TODO:
    //
    // Handle the case when both B and C are column-vectors that are contiguous in memory.
    // Because transposition of such vectors doesn't change their layout, and
    // Transpose(A*B) = Transpose(B) * Transpose(A), we can apply the same 'small-M'
    // optimization as above, with A and B flipped.
    //

    if (N == 1 && ldb == 1 && ldc == 1) {
        return;
    }

    //
    // Compute the strides to step through slices of the input matrices.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //

    size_t StrideN = MLAS_SGEMM_STRIDEN;
    size_t StrideK = MLAS_SGEMM_STRIDEK;

    if (N >= K) {
        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else{
        while (StrideN > 16 && StrideN / 2 >= N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < N; n += CountN) {
        CountN = std::min(N - n, StrideN);

        //
        // Step through each slice of matrix B along the K dimension.
        //

        size_t CountK;
        bool ZeroMode = true;

        // std::cout << "K:" << K << ",CountK:" << StrideK << ",CountN:" << CountN << std::endl;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, StrideK);

            //
            // Copy or transpose a panel of matrix B to a local packed buffer.
            //

            MlasSgemmNBitsTransposePackB(
                PanelB,
                QuantBData + k + n * ldb,
                ldb,
                QuantBScale + n * k_blks,
                k_blks,
                QuantBZeroPoint + n * k_blks_zp_bytes,
                k_blks_zp_bytes,
                BlkLen,
                CountN,
                CountK,
                k);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            float* c = C + n;
            MlasSgemmKernelLoopNBits(A + k, PanelB, c, CountK, M, CountN, lda, ldc, 1.0, ZeroMode);

            ZeroMode = false;
        }
    }
}

void
MlasSgemmNbitsThreaded(
    const ptrdiff_t ThreadCountM,
    const ptrdiff_t ThreadCountN,
    const size_t M,
    const size_t N,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    ptrdiff_t ThreadId,
    size_t BlkBitWidth,
    size_t BlkLen
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    SGEMM operation.

Arguments:

    ThreadCountM - Supplies the total thread partition on the M dimension.

    ThreadCountN - Supplies the total thread partition on the N dimension.

    TransA - Supplies the transpose operation on A matrix

    TransB - Supplies the transpose operation on B matrix

    M, N, K - Supplies the shape of the multiplication

    DataParams - Supplies the data position and layout of the matrices

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t RangeStartM;
    size_t RangeCountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t RangeStartN;
    size_t RangeCountN;

    const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
                            MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN, &RangeCountN);

    RangeStartN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    MlasSgemmNBitsOperation(K,
        DataParams,
        RangeStartM, RangeCountM,
        RangeStartN, RangeCountN,
        BlkBitWidth, BlkLen);
}


void MLASCALL
MlasSQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
)
{
    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t TargetThreadCount;

    if (Complexity < double(MLAS_QGEMM_THREAD_COMPLEXITY * GetMlasPlatform().MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = GetMlasPlatform().MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //

    ptrdiff_t ThreadsPerGemm = (TargetThreadCount + BatchN - 1) / BatchN;
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;

    if (N > M) {
        const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
                                MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }

        ThreadCountM = 1;
        ThreadCountN = ThreadsPerGemm;

    } else {
        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }

        ThreadCountM = ThreadsPerGemm;
        ThreadCountN = 1;
    }

    // std::cout << "ThreadCountM:" << ThreadCountM << ",ThreadCountN:"<<ThreadCountN << std::endl;
    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * static_cast<ptrdiff_t>(BatchN), [=](ptrdiff_t tid) {
        ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
        ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;
        MlasSgemmNbitsThreaded(
            ThreadCountM,
            ThreadCountN,
            M,
            N,
            K,
            &(DataParams[GemmIdx]),
            ThreadIdx,
            BlkBitWidth,
            BlkLen);
    });
#endif
}

bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen,
    size_t M
)
{
#ifdef MLAS_TARGET_ARM_ANY
    const int32_t QuantVariant = GetDispatchQuantVariant(BlkBitWidth, BlkLen);
    if (QuantVariant == -1) {
        return false;
    }

    if (GetMlasPlatform().SQNBitGemmDispatch == nullptr ||
        GetMlasPlatform().SQNBitGemmDispatch->Operations[QuantVariant] == nullptr) {
        return false;
    }

    return true;
#else
    (void)BlkBitWidth;
    (void)BlkLen;
    return M > 1;
#endif
}
