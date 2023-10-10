/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm_avx512.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory.
    Specificially on x64 avx512
--*/

#include "q4gemm.h"

#include <type_traits>
#include <immintrin.h>

struct MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI {
    static constexpr size_t StrideM = 256;
};

/**
 * @brief Horizontally sum 4 vectors and store
 *        the results in the returned vector
 */
static MLAS_FORCEINLINE __m128
FoldAccumulators(const __m512& acc0, const __m512& acc1, const __m512& acc2, const __m512& acc3)
{
    __m512 acc_lo01 = _mm512_unpacklo_ps(acc0, acc1);
    __m512 acc_hi01 = _mm512_unpackhi_ps(acc0, acc1);
    __m512 acc_lo23 = _mm512_unpacklo_ps(acc2, acc3);
    __m512 acc_hi23 = _mm512_unpackhi_ps(acc2, acc3);

    __m512 acc_lo0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    __m512 acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);

    __m256 acc_y =
        _mm256_add_ps(_mm512_extractf32x8_ps(acc_lo0123, 0), _mm512_extractf32x8_ps(acc_lo0123, 1));
    return _mm_add_ps(_mm256_extractf32x4_ps(acc_y, 0), _mm256_extractf32x4_ps(acc_y, 1));
}


template<typename Q4Type>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernelAvx512f(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    // We process 32 quantized values in a batch.
    static_assert(MLAS_QUANT4_BLK_UNIT == 32);
    static_assert(Q4Type::BlkLen % MLAS_QUANT4_BLK_UNIT == 0);

    const __m256i lowMask = _mm256_set1_epi8(0xF);

    for (size_t m = 0; m < CountM; m++) {
        const auto* b_col = PackedB;
        auto* sum_ptr = C;
        const auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN) - 4;
        while (nblk >= 0) {
            __m512 acc_lo0 = _mm512_setzero_ps();
            __m512 acc_lo1 = _mm512_setzero_ps();
            __m512 acc_lo2 = _mm512_setzero_ps();
            __m512 acc_lo3 = _mm512_setzero_ps();
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                size_t ck = std::min(CountK - k, Q4Type::BlkLen);

                const float scale_v0 = MlasQ4BlkScale<Q4Type>(b);
                const float scale_v1 = MlasQ4BlkScale<Q4Type>(b + ldb);
                const float scale_v2 = MlasQ4BlkScale<Q4Type>(b + ldb * 2);
                const float scale_v3 = MlasQ4BlkScale<Q4Type>(b + ldb * 3);

                const __m128i* b0ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b);
                const __m128i* b1ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb);
                const __m128i* b2ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 2);
                const __m128i* b3ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 3);

                for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT) {
                    size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT, ck - kk);

                    // Load A row vectors
                    uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT - kklen);
                    __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

                    mask = mask >> 16;
                    __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                                             : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

                    // Load B col vectors
                    const __m128i bvi4_0 = _mm_loadu_si128(b0ptr++);
                    const __m128i bvi4_1 = _mm_loadu_si128(b1ptr++);
                    const __m128i bvi4_2 = _mm_loadu_si128(b2ptr++);
                    const __m128i bvi4_3 = _mm_loadu_si128(b3ptr++);

                    // expand 4b into byte array
                    __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                    __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                    __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                    __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                    bytes0 = _mm256_and_si256(lowMask, bytes0);
                    bytes1 = _mm256_and_si256(lowMask, bytes1);
                    bytes2 = _mm256_and_si256(lowMask, bytes2);
                    bytes3 = _mm256_and_si256(lowMask, bytes3);

                    // Subtract zero-point from the integers
                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                        // Subtract zero-point from the integers
                        bytes0 = _mm256_sub_epi8(
                            bytes0, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b)));
                        bytes1 = _mm256_sub_epi8(
                            bytes1,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb)));
                        bytes2 = _mm256_sub_epi8(
                            bytes2,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 2)));
                        bytes3 = _mm256_sub_epi8(
                            bytes3,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 3)));
                    } else {
                        // Subtract 8 from the integers
                        const __m256i eight = _mm256_set1_epi8(8);
                        bytes0 = _mm256_sub_epi8(bytes0, eight);
                        bytes1 = _mm256_sub_epi8(bytes1, eight);
                        bytes2 = _mm256_sub_epi8(bytes2, eight);
                        bytes3 = _mm256_sub_epi8(bytes3, eight);
                    }

                    // Convert to 16-bit int
                    const __m256i vx16_lo0 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                    const __m256i vx16_hi0 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                    const __m256i vx16_lo1 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                    const __m256i vx16_hi1 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                    const __m256i vx16_lo2 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                    const __m256i vx16_hi2 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                    const __m256i vx16_lo3 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                    const __m256i vx16_hi3 =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                    // Convert to 32-bit int -> float 32
                    __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                    __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                    __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                    __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                    __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                    __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                    __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                    __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                    __m512 s = _mm512_set1_ps(scale_v0);
                    bvf_lo0 = _mm512_mul_ps(bvf_lo0, s);
                    bvf_hi0 = _mm512_mul_ps(bvf_hi0, s);
                    s = _mm512_set1_ps(scale_v1);
                    bvf_lo1 = _mm512_mul_ps(bvf_lo1, s);
                    bvf_hi1 = _mm512_mul_ps(bvf_hi1, s);
                    s = _mm512_set1_ps(scale_v2);
                    bvf_lo2 = _mm512_mul_ps(bvf_lo2, s);
                    bvf_hi2 = _mm512_mul_ps(bvf_hi2, s);
                    s = _mm512_set1_ps(scale_v3);
                    bvf_lo3 = _mm512_mul_ps(bvf_lo3, s);
                    bvf_hi3 = _mm512_mul_ps(bvf_hi3, s);

                    acc_lo0 = _mm512_fmadd_ps(bvf_lo0, av_lo, acc_lo0);
                    acc_lo0 = _mm512_fmadd_ps(bvf_hi0, av_hi, acc_lo0);
                    acc_lo1 = _mm512_fmadd_ps(bvf_lo1, av_lo, acc_lo1);
                    acc_lo1 = _mm512_fmadd_ps(bvf_hi1, av_hi, acc_lo1);
                    acc_lo2 = _mm512_fmadd_ps(bvf_lo2, av_lo, acc_lo2);
                    acc_lo2 = _mm512_fmadd_ps(bvf_hi2, av_hi, acc_lo2);
                    acc_lo3 = _mm512_fmadd_ps(bvf_lo3, av_lo, acc_lo3);
                    acc_lo3 = _mm512_fmadd_ps(bvf_hi3, av_hi, acc_lo3);
                }

                b += Q4Type::BlobSize;
            }

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_storeu_ps(sum_ptr, acc_x);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m512 acc_lo[4]{};
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                size_t ck = std::min(CountK - k, Q4Type::BlkLen);

                float scale_v[4];
                const __m128i* b_ptr[4];
                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    scale_v[nn] = MlasQ4BlkScale<Q4Type>(bb);
                    b_ptr[nn] = (const __m128i*)MlasQ4BlkData<Q4Type>(bb);
                }

                for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT) {
                    size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT, ck - kk);

                    uint32_t mask = 0xffffffff >> (MLAS_QUANT4_BLK_UNIT - kklen);
                    __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

                    mask = mask >> 16;
                    __m512 av_hi = mask == 0
                                       ? _mm512_setzero_ps()
                                       : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk + 16);

                    for (int64_t nn = 0; nn < nblk; nn++) {
                        const __m128i bvi4 = _mm_loadu_si128(b_ptr[nn]++);
                        __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                        bytes = _mm256_and_si256(lowMask, bytes);

                        if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                            // Subtract zero-point from the integers
                            const auto* bb = b + ldb * nn;
                            const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(bb);
                            bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
                        } else {
                            // Subtract 8 from the integers
                            const __m256i eight = _mm256_set1_epi8(8);
                            bytes = _mm256_sub_epi8(bytes, eight);
                        }

                        // Convert to 16-bit int
                        const __m256i vx16_lo =
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
                        const __m256i vx16_hi =
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                        // Convert to 32-bit int -> float 32
                        __m512 bvf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
                        __m512 bvf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
                        __m512 s = _mm512_set1_ps(scale_v[nn]);
                        bvf_lo = _mm512_mul_ps(bvf_lo, s);
                        bvf_hi = _mm512_mul_ps(bvf_hi, s);

                        acc_lo[nn] = _mm512_fmadd_ps(bvf_lo, av_lo, acc_lo[nn]);
                        acc_lo[nn] = _mm512_fmadd_ps(bvf_hi, av_hi, acc_lo[nn]);
                    }
                }
                b += Q4Type::BlobSize;
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        A += lda;
    }
    return CountM;
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK1,MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK1>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK2,MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK2>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK4,MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK4>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK0>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}


MLAS_FORCEINLINE
void
Transpose16x16Avx512(
    float* dest,
    __m512i r0,
    __m512i r1,
    __m512i r2,
    __m512i r3,
    __m512i r4,
    __m512i r5,
    __m512i r6,
    __m512i r7,
    __m512i r8,
    __m512i r9,
    __m512i ra,
    __m512i rb,
    __m512i rc,
    __m512i rd,
    __m512i re,
    __m512i rf)
{

    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(
        r0, r1);  //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(
        r0, r1);  //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2, r3);  //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2, r3);  //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4, r5);  //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4, r5);  //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6, r7);  //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6, r7);  //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8, r9);  // 128 ...
    t9 = _mm512_unpackhi_epi32(r8, r9);  // 130 ...
    ta = _mm512_unpacklo_epi32(ra, rb);  // 160 ...
    tb = _mm512_unpackhi_epi32(ra, rb);  // 162 ...
    tc = _mm512_unpacklo_epi32(rc, rd);  // 196 ...
    td = _mm512_unpackhi_epi32(rc, rd);  // 198 ...
    te = _mm512_unpacklo_epi32(re, rf);  // 228 ...
    tf = _mm512_unpackhi_epi32(re, rf);  // 230 ...

    r0 = _mm512_unpacklo_epi64(t0, t2);  //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0, t2);  //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1, t3);  //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1, t3);  //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4, t6);  //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4, t6);  //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5, t7);  //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5, t7);  //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8, ta);  // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8, ta);  // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9, tb);  // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9, tb);  // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc, te);  // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc, te);  // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td, tf);  // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td, tf);  // 195 211 231 243 ...

    t0 =
        _mm512_shuffle_i32x4(r0, r4, 0x88);  //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88);  //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88);  //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88);  //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd);  //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd);  //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd);  //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd);  //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88);  // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88);  // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88);  // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88);  // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd);  // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd);  // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd);  // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd);  // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88);  //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88);  //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88);  //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88);  //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88);  //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88);  //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88);  //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88);  //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd);  //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd);  //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd);  //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd);  //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd);  //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd);  //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd);  //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd);  //  15  31  47  63  79  96 111 127 ... 255

    _mm512_storeu_si512(dest, r0);
    dest += 16;
    _mm512_storeu_si512(dest, r1);
    dest += 16;
    _mm512_storeu_si512(dest, r2);
    dest += 16;
    _mm512_storeu_si512(dest, r3);
    dest += 16;
    _mm512_storeu_si512(dest, r4);
    dest += 16;
    _mm512_storeu_si512(dest, r5);
    dest += 16;
    _mm512_storeu_si512(dest, r6);
    dest += 16;
    _mm512_storeu_si512(dest, r7);
    dest += 16;
    _mm512_storeu_si512(dest, r8);
    dest += 16;
    _mm512_storeu_si512(dest, r9);
    dest += 16;
    _mm512_storeu_si512(dest, ra);
    dest += 16;
    _mm512_storeu_si512(dest, rb);
    dest += 16;
    _mm512_storeu_si512(dest, rc);
    dest += 16;
    _mm512_storeu_si512(dest, rd);
    dest += 16;
    _mm512_storeu_si512(dest, re);
    dest += 16;
    _mm512_storeu_si512(dest, rf);
    dest += 16;
}


template <typename Q4Type>
MLAS_FORCEINLINE
void
BlkQ4DequantBAvx512f(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    const auto* b_col = PackedB;

    int64_t nblk = (int64_t)(CountN)-16;
    while (nblk >= 0) {
        const auto* b = b_col;

        for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
            size_t ck = std::min(CountK - k, Q4Type::BlkLen);

            const float scale_v0 = MlasQ4BlkScale<Q4Type>(b);
            const float scale_v1 = MlasQ4BlkScale<Q4Type>(b + ldb);
            const float scale_v2 = MlasQ4BlkScale<Q4Type>(b + ldb * 2);
            const float scale_v3 = MlasQ4BlkScale<Q4Type>(b + ldb * 3);
            const float scale_v4 = MlasQ4BlkScale<Q4Type>(b + ldb * 4);
            const float scale_v5 = MlasQ4BlkScale<Q4Type>(b + ldb * 5);
            const float scale_v6 = MlasQ4BlkScale<Q4Type>(b + ldb * 6);
            const float scale_v7 = MlasQ4BlkScale<Q4Type>(b + ldb * 7);
            const float scale_v8 = MlasQ4BlkScale<Q4Type>(b + ldb * 8);
            const float scale_v9 = MlasQ4BlkScale<Q4Type>(b + ldb * 9);
            const float scale_va = MlasQ4BlkScale<Q4Type>(b + ldb * 10);
            const float scale_vb = MlasQ4BlkScale<Q4Type>(b + ldb * 11);
            const float scale_vc = MlasQ4BlkScale<Q4Type>(b + ldb * 12);
            const float scale_vd = MlasQ4BlkScale<Q4Type>(b + ldb * 13);
            const float scale_ve = MlasQ4BlkScale<Q4Type>(b + ldb * 14);
            const float scale_vf = MlasQ4BlkScale<Q4Type>(b + ldb * 15);

            const __m128i* b0ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b);
            const __m128i* b1ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb);
            const __m128i* b2ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 2);
            const __m128i* b3ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 3);
            const __m128i* b4ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 4);
            const __m128i* b5ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 5);
            const __m128i* b6ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 6);
            const __m128i* b7ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 7);
            const __m128i* b8ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 8);
            const __m128i* b9ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 9);
            const __m128i* baptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 10);
            const __m128i* bbptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 11);
            const __m128i* bcptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 12);
            const __m128i* bdptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 13);
            const __m128i* beptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 14);
            const __m128i* bfptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 15);

            for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT) {
                size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT, ck - kk);

                // Load B col vectors
                const __m128i bvi4_0 = _mm_loadu_si128(b0ptr++);
                const __m128i bvi4_1 = _mm_loadu_si128(b1ptr++);
                const __m128i bvi4_2 = _mm_loadu_si128(b2ptr++);
                const __m128i bvi4_3 = _mm_loadu_si128(b3ptr++);

                // expand 4b into byte array
                __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    // Subtract zero-point from the integers
                    bytes0 = _mm256_sub_epi8(
                        bytes0, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 2)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 3)));
                } else {
                    // Subtract 8 from the integers
                    const __m256i eight = _mm256_set1_epi8(8);
                    bytes0 = _mm256_sub_epi8(bytes0, eight);
                    bytes1 = _mm256_sub_epi8(bytes1, eight);
                    bytes2 = _mm256_sub_epi8(bytes2, eight);
                    bytes3 = _mm256_sub_epi8(bytes3, eight);
                }

                // Convert to 16-bit int
                __m256i vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                __m256i vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                __m256i vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                __m256i vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                __m256i vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                __m256i vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                __m256i vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                __m256i vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                __m512 s = _mm512_set1_ps(scale_v0);
                bvf_lo0 = _mm512_mul_ps(bvf_lo0, s);
                bvf_hi0 = _mm512_mul_ps(bvf_hi0, s);
                s = _mm512_set1_ps(scale_v1);
                bvf_lo1 = _mm512_mul_ps(bvf_lo1, s);
                bvf_hi1 = _mm512_mul_ps(bvf_hi1, s);
                s = _mm512_set1_ps(scale_v2);
                bvf_lo2 = _mm512_mul_ps(bvf_lo2, s);
                bvf_hi2 = _mm512_mul_ps(bvf_hi2, s);
                s = _mm512_set1_ps(scale_v3);
                bvf_lo3 = _mm512_mul_ps(bvf_lo3, s);
                bvf_hi3 = _mm512_mul_ps(bvf_hi3, s);

                // Load B col vectors
                const __m128i bvi4_4 = _mm_loadu_si128(b4ptr++);
                const __m128i bvi4_5 = _mm_loadu_si128(b5ptr++);
                const __m128i bvi4_6 = _mm_loadu_si128(b6ptr++);
                const __m128i bvi4_7 = _mm_loadu_si128(b7ptr++);

                // expand 4b into byte array
                bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_4, 4), bvi4_4);
                bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_5, 4), bvi4_5);
                bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_6, 4), bvi4_6);
                bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_7, 4), bvi4_7);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    // Subtract zero-point from the integers
                    bytes0 = _mm256_sub_epi8(
                        bytes0,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 4)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 5)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 6)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 7)));
                } else {
                    // Subtract 8 from the integers
                    const __m256i eight = _mm256_set1_epi8(8);
                    bytes0 = _mm256_sub_epi8(bytes0, eight);
                    bytes1 = _mm256_sub_epi8(bytes1, eight);
                    bytes2 = _mm256_sub_epi8(bytes2, eight);
                    bytes3 = _mm256_sub_epi8(bytes3, eight);
                }

                // Convert to 16-bit int
                vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_lo4 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hi4 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lo5 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hi5 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_lo6 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hi6 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lo7 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hi7 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                s = _mm512_set1_ps(scale_v4);
                bvf_lo4 = _mm512_mul_ps(bvf_lo4, s);
                bvf_hi4 = _mm512_mul_ps(bvf_hi4, s);
                s = _mm512_set1_ps(scale_v5);
                bvf_lo5 = _mm512_mul_ps(bvf_lo5, s);
                bvf_hi5 = _mm512_mul_ps(bvf_hi5, s);
                s = _mm512_set1_ps(scale_v6);
                bvf_lo6 = _mm512_mul_ps(bvf_lo6, s);
                bvf_hi6 = _mm512_mul_ps(bvf_hi6, s);
                s = _mm512_set1_ps(scale_v7);
                bvf_lo7 = _mm512_mul_ps(bvf_lo7, s);
                bvf_hi7 = _mm512_mul_ps(bvf_hi7, s);

                // Load B col vectors
                const __m128i bvi4_8 = _mm_loadu_si128(b8ptr++);
                const __m128i bvi4_9 = _mm_loadu_si128(b9ptr++);
                const __m128i bvi4_a = _mm_loadu_si128(baptr++);
                const __m128i bvi4_b = _mm_loadu_si128(bbptr++);

                // expand 4b into byte array
                bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_8, 4), bvi4_8);
                bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_9, 4), bvi4_9);
                bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_a, 4), bvi4_a);
                bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_b, 4), bvi4_b);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    // Subtract zero-point from the integers
                    bytes0 = _mm256_sub_epi8(
                        bytes0,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 8)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 9)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 10)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 11)));
                } else {
                    // Subtract 8 from the integers
                    const __m256i eight = _mm256_set1_epi8(8);
                    bytes0 = _mm256_sub_epi8(bytes0, eight);
                    bytes1 = _mm256_sub_epi8(bytes1, eight);
                    bytes2 = _mm256_sub_epi8(bytes2, eight);
                    bytes3 = _mm256_sub_epi8(bytes3, eight);
                }

                // Convert to 16-bit int
                vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_lo8 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hi8 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lo9 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hi9 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_loa = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hia = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lob = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hib = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                s = _mm512_set1_ps(scale_v8);
                bvf_lo8 = _mm512_mul_ps(bvf_lo8, s);
                bvf_hi8 = _mm512_mul_ps(bvf_hi8, s);
                s = _mm512_set1_ps(scale_v9);
                bvf_lo9 = _mm512_mul_ps(bvf_lo9, s);
                bvf_hi9 = _mm512_mul_ps(bvf_hi9, s);
                s = _mm512_set1_ps(scale_va);
                bvf_loa = _mm512_mul_ps(bvf_loa, s);
                bvf_hia = _mm512_mul_ps(bvf_hia, s);
                s = _mm512_set1_ps(scale_vb);
                bvf_lob = _mm512_mul_ps(bvf_lob, s);
                bvf_hib = _mm512_mul_ps(bvf_hib, s);

                // Load B col vectors
                const __m128i bvi4_c = _mm_loadu_si128(bcptr++);
                const __m128i bvi4_d = _mm_loadu_si128(bdptr++);
                const __m128i bvi4_e = _mm_loadu_si128(beptr++);
                const __m128i bvi4_f = _mm_loadu_si128(bfptr++);

                // expand 4b into byte array
                bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_c, 4), bvi4_c);
                bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_d, 4), bvi4_d);
                bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_e, 4), bvi4_e);
                bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_f, 4), bvi4_f);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    // Subtract zero-point from the integers
                    bytes0 = _mm256_sub_epi8(
                        bytes0,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 12)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 13)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 14)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 15)));
                } else {
                    // Subtract 8 from the integers
                    const __m256i eight = _mm256_set1_epi8(8);
                    bytes0 = _mm256_sub_epi8(bytes0, eight);
                    bytes1 = _mm256_sub_epi8(bytes1, eight);
                    bytes2 = _mm256_sub_epi8(bytes2, eight);
                    bytes3 = _mm256_sub_epi8(bytes3, eight);
                }

                // Convert to 16-bit int
                vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_loc = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hic = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lod = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hid = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_loe = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hie = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lof = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hif = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));

                s = _mm512_set1_ps(scale_vc);
                bvf_loc = _mm512_mul_ps(bvf_loc, s);
                bvf_hic = _mm512_mul_ps(bvf_hic, s);
                s = _mm512_set1_ps(scale_vd);
                bvf_lod = _mm512_mul_ps(bvf_lod, s);
                bvf_hid = _mm512_mul_ps(bvf_hid, s);
                s = _mm512_set1_ps(scale_ve);
                bvf_loe = _mm512_mul_ps(bvf_loe, s);
                bvf_hie = _mm512_mul_ps(bvf_hie, s);
                s = _mm512_set1_ps(scale_vf);
                bvf_lof = _mm512_mul_ps(bvf_lof, s);
                bvf_hif = _mm512_mul_ps(bvf_hif, s);
                Transpose16x16Avx512(FpData, _mm512_castps_si512(bvf_lo0),
                                     _mm512_castps_si512(bvf_lo1), _mm512_castps_si512(bvf_lo2),
                                     _mm512_castps_si512(bvf_lo3), _mm512_castps_si512(bvf_lo4),
                                     _mm512_castps_si512(bvf_lo5), _mm512_castps_si512(bvf_lo6),
                                     _mm512_castps_si512(bvf_lo7), _mm512_castps_si512(bvf_lo8),
                                     _mm512_castps_si512(bvf_lo9), _mm512_castps_si512(bvf_loa),
                                     _mm512_castps_si512(bvf_lob), _mm512_castps_si512(bvf_loc),
                                     _mm512_castps_si512(bvf_lod), _mm512_castps_si512(bvf_loe),
                                     _mm512_castps_si512(bvf_lof));
                if (kklen > 16) {
                    Transpose16x16Avx512(FpData + 16 * 16, _mm512_castps_si512(bvf_hi0),
                                         _mm512_castps_si512(bvf_hi1), _mm512_castps_si512(bvf_hi2),
                                         _mm512_castps_si512(bvf_hi3), _mm512_castps_si512(bvf_hi4),
                                         _mm512_castps_si512(bvf_hi5), _mm512_castps_si512(bvf_hi6),
                                         _mm512_castps_si512(bvf_hi7), _mm512_castps_si512(bvf_hi8),
                                         _mm512_castps_si512(bvf_hi9), _mm512_castps_si512(bvf_hia),
                                         _mm512_castps_si512(bvf_hib), _mm512_castps_si512(bvf_hic),
                                         _mm512_castps_si512(bvf_hid), _mm512_castps_si512(bvf_hie),
                                         _mm512_castps_si512(bvf_hif));
                }
                FpData += 16 * kklen;
            }

            b += Q4Type::BlobSize;
        }

        // move to next 16 columns
        b_col += 16 * ldb;
        nblk -= 16;
    }

    // left over columns less than 16 ?
    nblk += 16;
    if (nblk > 0) {
        const auto* b = b_col;

        for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
            size_t ck = std::min(CountK - k, Q4Type::BlkLen);

            float scale_v[16];
            const __m128i* b_ptr[16];
            for (int64_t nn = 0; nn < nblk; nn++) {
                const auto* bb = b + ldb * nn;
                scale_v[nn] = MlasQ4BlkScale<Q4Type>(bb);
                b_ptr[nn] = (const __m128i*)MlasQ4BlkData<Q4Type>(bb);
            }

            for (size_t kk = 0; kk < ck; kk += MLAS_QUANT4_BLK_UNIT) {
                size_t kklen = std::min((size_t)MLAS_QUANT4_BLK_UNIT, ck - kk);
                __m512 bvf_lo[16];
                __m512 bvf_hi[16];
                for (int64_t nn = 0; nn < nblk; nn++) {
                    const __m128i bvi4 = _mm_loadu_si128(b_ptr[nn]++);
                    __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                    bytes = _mm256_and_si256(lowMask, bytes);

                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                        // Subtract zero-point from the integers
                        const auto* bb = b + ldb * nn;
                        const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(bb);
                        bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
                    } else {
                        // Subtract 8 from the integers
                        const __m256i eight = _mm256_set1_epi8(8);
                        bytes = _mm256_sub_epi8(bytes, eight);
                    }

                    // Convert to 16-bit int
                    const __m256i vx16_lo =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
                    const __m256i vx16_hi =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                    // Convert to 32-bit int -> float 32
                    bvf_lo[nn] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
                    bvf_hi[nn] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
                    const __m512 s = _mm512_set1_ps(scale_v[nn]);
                    bvf_lo[nn] = _mm512_mul_ps(bvf_lo[nn], s);
                    bvf_hi[nn] = _mm512_mul_ps(bvf_hi[nn], s);
                }
                for (int64_t nn = nblk; nn < 16; nn++) {
                    bvf_lo[nn] = _mm512_setzero_ps();
                    bvf_hi[nn] = _mm512_setzero_ps();
                }
                Transpose16x16Avx512(
                    FpData, _mm512_castps_si512(bvf_lo[0]), _mm512_castps_si512(bvf_lo[1]),
                    _mm512_castps_si512(bvf_lo[2]), _mm512_castps_si512(bvf_lo[3]),
                    _mm512_castps_si512(bvf_lo[4]), _mm512_castps_si512(bvf_lo[5]),
                    _mm512_castps_si512(bvf_lo[6]), _mm512_castps_si512(bvf_lo[7]),
                    _mm512_castps_si512(bvf_lo[8]), _mm512_castps_si512(bvf_lo[9]),
                    _mm512_castps_si512(bvf_lo[10]), _mm512_castps_si512(bvf_lo[11]),
                    _mm512_castps_si512(bvf_lo[12]), _mm512_castps_si512(bvf_lo[13]),
                    _mm512_castps_si512(bvf_lo[14]), _mm512_castps_si512(bvf_lo[15]));
                if (kklen > 16) {
                    Transpose16x16Avx512(
                        FpData + 16 * 16, _mm512_castps_si512(bvf_hi[0]),
                        _mm512_castps_si512(bvf_hi[1]), _mm512_castps_si512(bvf_hi[2]),
                        _mm512_castps_si512(bvf_hi[3]), _mm512_castps_si512(bvf_hi[4]),
                        _mm512_castps_si512(bvf_hi[5]), _mm512_castps_si512(bvf_hi[6]),
                        _mm512_castps_si512(bvf_hi[7]), _mm512_castps_si512(bvf_hi[8]),
                        _mm512_castps_si512(bvf_hi[9]), _mm512_castps_si512(bvf_hi[10]),
                        _mm512_castps_si512(bvf_hi[11]), _mm512_castps_si512(bvf_hi[12]),
                        _mm512_castps_si512(bvf_hi[13]), _mm512_castps_si512(bvf_hi[14]),
                        _mm512_castps_si512(bvf_hi[15]));
                }
                FpData += 16 * kklen;
            }
            b += Q4Type::BlobSize;
        }
    }
}


template<>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK0>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK1>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK2>(FpData, PackedB, CountN, CountK, ldb);
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK4>(FpData, PackedB, CountN, CountK, ldb);
}

/**
 * @brief For testing purpose,
 *        Dequantize the data intp fp32, and then pack them for use
 *        in sgemm kernel. equivalent to MlasQ4GemmUnPackB and then
 *        MlasSgemmCopyPackB
 * @param QType
 * @param FpData
 * @param PackedB
 * @param CountN
 * @param CountK
 * @param ldb
 */
void
MlasBlkQ4DequantSgemmPackB(
    MLAS_BLK_QUANT_TYPE QType,
    float* FpData,
    const uint8_t* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb)
{
    switch (QType) {
        case BlkQ4Zp8:
            return BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK1>(FpData, PackedB, CountN, CountK, ldb);
        case BlkQ4Sym64:
            return BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK2>(FpData, PackedB, CountN, CountK, ldb);
        case BlkQ4Sym128:
            return BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK4>(FpData, PackedB, CountN, CountK, ldb);
        default:
            return BlkQ4DequantBAvx512f<MLAS_Q4TYPE_BLK0>(FpData, PackedB, CountN, CountK, ldb);
    }
}


static MLAS_Q4GEMM_OPERATION* Q4Operations_avx512vnni[] = {
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    nullptr,
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>
};

const MLAS_FPQ4GEMM_DISPATCH MlasFpQ4GemmDispatchAvx512 = {
    Q4Operations_avx512vnni
};


////////////////////////////////////////////////////////////
//  Block int8 quantization, currently we only
//  implement symmetric quant, with no zero-point

template <typename QType>
MLAS_FORCEINLINE void
MlasQ80BlkQuantRow(const float* A, void* Qblob, size_t size)
{
    static_assert(QType::BlkLen % 16 == 0);
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    int8_t* blob = reinterpret_cast<int8_t*>(Qblob);
    for (size_t k = 0; k < size; k += QType::BlkLen) {
        const size_t step = std::min(QType::BlkLen, size - k);

        __m512 maxAbs = _mm512_setzero_ps();
        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);

            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

            // Compute max(abs(e)) for the block
            maxAbs = _mm512_max_ps(maxAbs, _mm512_andnot_ps(signBit, v0));
        }

        __m256 max8 =
            _mm256_max_ps(_mm512_extractf32x8_ps(maxAbs, 1), _mm512_extractf32x8_ps(maxAbs, 0));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max8, 1), _mm256_castps256_ps128(max8));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float scale = maxScalar / 127.f;
        *reinterpret_cast<float*>(blob) = scale;
        blob += sizeof(float);

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(inverse_scale);
        __m128i* dst = reinterpret_cast<__m128i*>(blob);

        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);

            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);
            v0 = _mm512_mul_ps(v0, mul);

            // Round to nearest integer
            v0 = _mm512_roundscale_ps(v0, _MM_ROUND_NEAREST);

            // Convert floats to integers
            __m512i i0 = _mm512_cvtps_epi32(v0);

            // Convert int32 to int8
            _mm_storeu_si128(dst++, _mm512_cvtepi32_epi8(i0));
        }
        if (step < QType::BlkLen) {
            memset(blob + step, 0, QType::BlkLen - step);
        }
        blob += QType::BlkLen;
    }
}

template<typename QType>
void
Q80BlkQuant(void* Qblob, const float* A, size_t M, size_t K, size_t lda, MLAS_THREADPOOL* ThreadPool)
{
    const size_t parts = (size_t)ceil(double(M) * K / (16.0 * 1024));
    const size_t TargetThreadCnt =
        std::max(std::min(parts, (size_t)MlasGetMaximumThreadCount(ThreadPool)), (size_t)1);
    const size_t linesize = MlasQ80BlkQuantSizeImpl<QType>(1, K);

    size_t M_stride = MlasDivRoundup(M, TargetThreadCnt);
    size_t threads = MlasDivRoundup(M, M_stride);
    MlasTrySimpleParallel(ThreadPool, threads, [&](ptrdiff_t tid) {
        const size_t m = tid * M_stride;
        const float* src = A + lda * m;
        uint8_t* dst = reinterpret_cast<uint8_t*>(Qblob) + m * linesize;
        for (size_t i = 0; i < std::min(M_stride, M-m); i++) {
            MlasQ80BlkQuantRow<QType>(src, dst, K);
            src += lda;
            dst += linesize;
        }
    });
}

static MLAS_Q80_BLKQUANT* Q80Quant_avx512vnni[] = {
    Q80BlkQuant<MLAS_Q4TYPE_BLK0>,
    Q80BlkQuant<MLAS_Q4TYPE_BLK1>,
    Q80BlkQuant<MLAS_Q4TYPE_BLK2>,
    nullptr,
    Q80BlkQuant<MLAS_Q4TYPE_BLK4>
};


static
MLAS_FORCEINLINE
__m128
FoldAccumulators(
    const __m256& acc0,
    const __m256& acc1,
    const __m256& acc2,
    const __m256& acc3
    )
{
    __m256 acc_lo01 = _mm256_unpacklo_ps(acc0, acc1);
    __m256 acc_hi01 = _mm256_unpackhi_ps(acc0, acc1);
    __m256 acc_lo23 = _mm256_unpacklo_ps(acc2, acc3);
    __m256 acc_hi23 = _mm256_unpackhi_ps(acc2, acc3);

    __m256 acc_lo0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    __m256 acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);

    return _mm_add_ps(_mm256_extractf32x4_ps(acc_lo0123, 0), _mm256_extractf32x4_ps(acc_lo0123, 1));
}

static inline float
mm256_reduce_add_ps(__m256& x)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}


template<typename Q4Type>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernelAvx512f(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    // We process 32 quantized values in a batch.
    static_assert(MLAS_QUANT4_BLK_UNIT == 32);
    static_assert(Q4Type::BlkLen % MLAS_QUANT4_BLK_UNIT == 0);

    const __m256i zero = _mm256_setzero_si256();
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    for (size_t m = 0; m < CountM; m++) {
        const uint8_t* b_col = PackedB;
        auto* sum_ptr = C;
        auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN) - 4;
        while (nblk >= 0) {
            __m256 acc_lo0 = _mm256_setzero_ps();
            __m256 acc_lo1 = _mm256_setzero_ps();
            __m256 acc_lo2 = _mm256_setzero_ps();
            __m256 acc_lo3 = _mm256_setzero_ps();
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                const float a_scale = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);
                const float scale_v0 = MlasQ4BlkScale<Q4Type>(b) * a_scale;
                const float scale_v1 = MlasQ4BlkScale<Q4Type>(b + ldb) * a_scale;
                const float scale_v2 = MlasQ4BlkScale<Q4Type>(b + ldb * 2) * a_scale;
                const float scale_v3 = MlasQ4BlkScale<Q4Type>(b + ldb * 3) * a_scale;

                const __m128i* b0ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b);
                const __m128i* b1ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb);
                const __m128i* b2ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 2);
                const __m128i* b3ptr = (const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 3);

                for (size_t kk = 0; kk < Q4Type::BlkLen; kk += MLAS_QUANT4_BLK_UNIT) {
                    // Load A row vector
                    const __m256i a_bytes = _mm256_loadu_si256((const __m256i*)ablob);
                    ablob += MLAS_QUANT4_BLK_UNIT;

                    // Load 4 B column vectors (quantized to int4 blobs)
                    const __m128i bvi4_0 = _mm_loadu_si128(b0ptr++);
                    const __m128i bvi4_1 = _mm_loadu_si128(b1ptr++);
                    const __m128i bvi4_2 = _mm_loadu_si128(b2ptr++);
                    const __m128i bvi4_3 = _mm_loadu_si128(b3ptr++);

                    // expand 4b into byte array
                    __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                    __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                    __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                    __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                    bytes0 = _mm256_and_si256(lowMask, bytes0);
                    bytes1 = _mm256_and_si256(lowMask, bytes1);
                    bytes2 = _mm256_and_si256(lowMask, bytes2);
                    bytes3 = _mm256_and_si256(lowMask, bytes3);

                    // Subtract zero-point from the integers
                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                        bytes0 = _mm256_sub_epi8(
                            bytes0, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b)));
                        bytes1 = _mm256_sub_epi8(
                            bytes1,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb)));
                        bytes2 = _mm256_sub_epi8(
                            bytes2,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 2)));
                        bytes3 = _mm256_sub_epi8(
                            bytes3,
                            _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 3)));
                    } else {
                        const __m256i eight = _mm256_set1_epi8(8);
                        bytes0 = _mm256_sub_epi8(bytes0, eight);
                        bytes1 = _mm256_sub_epi8(bytes1, eight);
                        bytes2 = _mm256_sub_epi8(bytes2, eight);
                        bytes3 = _mm256_sub_epi8(bytes3, eight);
                    }

                    // to use vnni unsigned x signed int, negate all negative
                    // b vals to make it all positive, and then also negate the
                    // corresponding a vals to compensate
                    const __m256i summed_pairs0 = _mm256_dpbusd_epi32(
                        zero, _mm256_sign_epi8(bytes0, bytes0), _mm256_sign_epi8(a_bytes, bytes0));
                    const __m256i summed_pairs1 = _mm256_dpbusd_epi32(
                        zero, _mm256_sign_epi8(bytes1, bytes1), _mm256_sign_epi8(a_bytes, bytes1));
                    const __m256i summed_pairs2 = _mm256_dpbusd_epi32(
                        zero, _mm256_sign_epi8(bytes2, bytes2), _mm256_sign_epi8(a_bytes, bytes2));
                    const __m256i summed_pairs3 = _mm256_dpbusd_epi32(
                        zero, _mm256_sign_epi8(bytes3, bytes3), _mm256_sign_epi8(a_bytes, bytes3));

                    const __m256 sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                    const __m256 sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                    const __m256 sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                    const __m256 sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                    acc_lo0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sums0, acc_lo0);
                    acc_lo1 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v1), sums1, acc_lo1);
                    acc_lo2 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v2), sums2, acc_lo2);
                    acc_lo3 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v3), sums3, acc_lo3);
                }
                b += Q4Type::BlobSize;
            }

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_storeu_ps(sum_ptr, acc_x);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m256 acc_lo[4]{};
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += Q4Type::BlkLen) {
                const float a_scale = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);

                float scale_v[4];
                const __m128i* b_ptr[4];
                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    scale_v[nn] = MlasQ4BlkScale<Q4Type>(bb) * a_scale;
                    b_ptr[nn] = (const __m128i*)MlasQ4BlkData<Q4Type>(bb);
                }

                for (size_t kk = 0; kk < Q4Type::BlkLen; kk += MLAS_QUANT4_BLK_UNIT) {
                    const __m256i a_bytes = _mm256_loadu_si256((const __m256i*)ablob);
                    ablob += MLAS_QUANT4_BLK_UNIT;

                    for (int64_t nn = 0; nn < nblk; nn++) {
                        const __m128i bvi4 = _mm_loadu_si128(b_ptr[nn]++);
                        __m256i b_bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                        b_bytes = _mm256_and_si256(lowMask, b_bytes);

                        if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                            // Subtract zero-point from the integers
                            const auto* bb = b + ldb * nn;
                            const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(bb);
                            b_bytes = _mm256_sub_epi8(b_bytes, _mm256_set1_epi8(zp));
                        } else {
                            // Subtract 8 from the integers
                            const __m256i eight = _mm256_set1_epi8(8);
                            b_bytes = _mm256_sub_epi8(b_bytes, eight);
                        }

                        // to use vnni unsigned x signed int, negate all negative
                        // b vals to make it all positive,
                        const __m256i ax = _mm256_sign_epi8(b_bytes, b_bytes);
                        // and then also negate the corresponding a vals to compensate
                        const __m256i sy = _mm256_sign_epi8(a_bytes, b_bytes);
                        const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                        const __m256 sum = _mm256_cvtepi32_ps(summed_pairs);
                        acc_lo[nn] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v[nn]), sum, acc_lo[nn]);
                    }
                }
                b += Q4Type::BlobSize;
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        QuantA += lda;
    }
    return CountM;
}


template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK1,MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK1>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK0>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK2>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK4>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}


static MLAS_Q8Q4GEMM_OPERATION* Q8Q4Operations_avx512vnni[] = {
    MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK2, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>,
    nullptr,
    MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK4, MLAS_FP_Q4_GEMM_KERNEL_AVX512VNNI>
};


const MLAS_Q8Q4GEMM_DISPATCH MlasQ8Q4GemmDispatchAvx512vnni = {
    Q80Quant_avx512vnni,
    Q8Q4Operations_avx512vnni
};
