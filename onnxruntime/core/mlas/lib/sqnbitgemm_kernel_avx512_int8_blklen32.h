#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx2_int8_blklen32.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen64.h"

static MLAS_FORCEINLINE void
load_4blk_4b_packed_blklen32(const std::byte* QuantBDataPtr, __m512i& bv0_64_epi8, __m512i& bv1_64_epi8)
{
    // | v0 v64 | v1 v65 | ... | v62 v126 | v63 v127 |
    const __m512i bv_packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantBDataPtr));
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    bv0_64_epi8 = _mm512_and_si512(bv_packed, low_mask);                          // 0~63
    bv1_64_epi8 = _mm512_srli_epi16(_mm512_sub_epi8(bv_packed, bv0_64_epi8), 4);  // 64~127
}

static const uint32_t index_array[16] = {0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3};

static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk4_avx512(
  const __m512i& av0_64_epi8,
  const __m512i& av1_64_epi8,
  const std::byte* QuantBDataPtr,
  const float* scale_a,
  const float* scale_b,
  __m512& acc0)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_4blk_4b_packed_blklen32(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m128 scale_b_ps = _mm_loadu_ps(scale_b);  // 0123
    {
        const __m128 scale_a0_ps = _mm_loadu_ps(scale_a);  // 0123
        const __m128 scale_a0b_ps = _mm_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_broadcast_f32x4(scale_a0b_ps);  // 0123012301230123

        __m512i idx = _mm512_set_epi32(3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0);
        // __m512i idx = _mm512_loadu_epi8(&index_array[0]);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0022002211331133

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av0_64_epi8);  // 0~0,1~1
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av1_64_epi8);  // 2~2,3~3

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);                   // 00002222000022221111333311113333
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);  // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen32_r2c1blk4_avx512(
    const __m512i& av00_64_epi8,
    const __m512i& av01_64_epi8,
    const __m512i& av10_64_epi8,
    const __m512i& av11_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_2blk_4b_packed_blklen64(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m128 scale_b_ps = _mm_loadu_ps(scale_b); // 0123
    {
        const __m128 scale_a0_ps = _mm_loadu_ps(scale_a0);  // 0123
        const __m128 scale_a0b_ps = _mm_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_broadcast_f32x4(scale_a0b_ps);  // 0123012301230123

        __m512i idx = _mm512_set_epi32(3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0);
        // __m512i idx = _mm512_loadu_epi8(&index_array[0]);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0022002211331133

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av00_64_epi8);  // 0~0,1~1
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av01_64_epi8);  // 2~2,3~3

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);                   // 00002222000022221111333311113333
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);  // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
    {
        const __m128 scale_a1_ps = _mm_loadu_ps(scale_a1);  // 0123
        const __m128 scale_a1b_ps = _mm_mul_ps(scale_b_ps, scale_a1_ps);
        __m512 scale_a1b_16_ps = _mm512_broadcast_f32x4(scale_a1b_ps);  // 0123012301230123

        __m512i idx = _mm512_set_epi32(3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0);
        // __m512i idx = _mm512_loadu_epi8(&index_array[0]);
        scale_a1b_16_ps = _mm512_permutexvar_ps(idx, scale_a1b_16_ps);  // 0022002211331133

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av10_64_epi8);  // 0~0,1~1
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av11_64_epi8);  // 2~2,3~3

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);  // 00002222000022221111333311113333
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);                   // 00002222000022221111333311113333
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);  // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc1 = _mm512_fmadd_ps(sum_16_ps, scale_a1b_16_ps, acc1);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk4_avx512vnni(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m512& acc0
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_4blk_4b_packed_blklen32(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m128 scale_b_ps = _mm_loadu_ps(scale_b);  // 0123
    {
        const __m128 scale_a0_ps = _mm_loadu_ps(scale_a);  // 0123
        const __m128 scale_a0b_ps = _mm_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_broadcast_f32x4(scale_a0b_ps);  // 0123012301230123

        __m512i idx = _mm512_set_epi32(3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0);
        //__m512i idx = _mm512_loadu_epi8(&index_array[0]);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0022002211331133

        const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av0_64_epi8);  // 0000000011111111
        const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av1_64_epi8);  // 2222222233333333

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_16_epi32, dot1_16_epi32);  // 0022002211331133
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_16_epi32, dot1_16_epi32);  // 0022002211331133
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);          // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen32_r2c1blk4_avx512vnni(
    const __m512i& av00_64_epi8,
    const __m512i& av01_64_epi8,
    const __m512i& av10_64_epi8,
    const __m512i& av11_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_2blk_4b_packed_blklen64(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);
    __m512i idx = _mm512_set_epi32(3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0);
    //__m512i idx = _mm512_loadu_epi8(&index_array[0]);

    const __m128 scale_b_ps = _mm_loadu_ps(scale_b);  // 0123
    {
        const __m128 scale_a0_ps = _mm_loadu_ps(scale_a0);  // 0123
        const __m128 scale_a0b_ps = _mm_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_broadcast_f32x4(scale_a0b_ps);  // 0123012301230123

        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0022002211331133

        const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av00_64_epi8);  // 0000000011111111
        const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av01_64_epi8);  // 2222222233333333

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_16_epi32, dot1_16_epi32);  // 0022002211331133
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_16_epi32, dot1_16_epi32);  // 0022002211331133
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);          // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
    {
        const __m128 scale_a1_ps = _mm_loadu_ps(scale_a1);  // 0123
        const __m128 scale_a1b_ps = _mm_mul_ps(scale_b_ps, scale_a1_ps);
        __m512 scale_a1b_16_ps = _mm512_broadcast_f32x4(scale_a1b_ps);  // 0123012301230123

        scale_a1b_16_ps = _mm512_permutexvar_ps(idx, scale_a1b_16_ps);  // 0022002211331133

        const __m512i dot0_32_epi16 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av10_64_epi8);  // 0000000011111111
        const __m512i dot1_32_epi16 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av11_64_epi8);  // 2222222233333333

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);  // 0022002211331133
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);  // 0022002211331133
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);          // 0022002211331133
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc1 = _mm512_fmadd_ps(sum_16_ps, scale_a1b_16_ps, acc1);
    }
}

MLAS_FORCEINLINE void
accumulate_1blk_dot_avx512vnni(const __m256i& av_32_epi8, const __m256i& bv_32_epi8, const float& combined_scale, __m256& acc)
{
    __m256i sum_8_epi32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bv_32_epi8, av_32_epi8);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
    acc = _mm256_fmadd_ps(sum_ps, _mm256_set1_ps(combined_scale), acc);
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk1_avx512(
    const __m256i& av00_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale00,
    __m256& acc0
)
{
    if constexpr (vnni) {
        // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
        const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
        __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
        bv_32_epi8 = _mm256_and_si256(_mm256_set1_epi8(0x0F), bv_32_epi8);
        accumulate_1blk_dot_avx512vnni(av00_32_epi8, bv_32_epi8, combined_scale00, acc0);
    } else {
        accumulate_blklen32_r1c1blk1_avx2<false>(av00_32_epi8, QuantBDataPtr, combined_scale00, acc0);
    }
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r2c1blk1_avx512(
    const __m256i& av00_32_epi8,
    const __m256i& av10_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale00,
    const float& combined_scale10,
    __m256& acc0,
    __m256& acc1
)
{
    if constexpr (vnni) {
        // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
        const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
        __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
        bv_32_epi8 = _mm256_and_si256(_mm256_set1_epi8(0x0F), bv_32_epi8);

        accumulate_1blk_dot_avx512vnni(av00_32_epi8, bv_32_epi8, combined_scale00, acc0);
        accumulate_1blk_dot_avx512vnni(av10_32_epi8, bv_32_epi8, combined_scale10, acc1);
    } else {
        accumulate_blklen32_r2c1blk1_avx2<false>(av00_32_epi8, av10_32_epi8, QuantBDataPtr, combined_scale00, combined_scale10, acc0, acc1);
    }
}

static MLAS_FORCEINLINE void
accumulate_scaled_zp_prod_r1_c4(__m512& acc0, __m512& acc1, __m512& acc2, __m512& acc3, const float* scaled_zp_a, const float* scaled_zp_b, size_t BlockCountK)
{
    constexpr size_t PerAccuBlk16 = 16;
    size_t k_blks_remaining = BlockCountK;
    // process 2 blks of 64 4b weights a time
    for (; k_blks_remaining >= PerAccuBlk16; k_blks_remaining -= PerAccuBlk16) {
        const __m512 a_16_ps = _mm512_loadu_ps(scaled_zp_a);
        {
            const __m512 b_16_ps0 = _mm512_loadu_ps(scaled_zp_b);
            acc0 = _mm512_fmadd_ps(a_16_ps, b_16_ps0, acc0);
        }
        {
            const __m512 b_16_ps1 = _mm512_loadu_ps(scaled_zp_b + BlockCountK);
            acc1 = _mm512_fmadd_ps(a_16_ps, b_16_ps1, acc1);
        }
        {
            const __m512 b_16_ps2 = _mm512_loadu_ps(scaled_zp_b + 2 * BlockCountK);
            acc2 = _mm512_fmadd_ps(a_16_ps, b_16_ps2, acc2);
        }
        {
            const __m512 b_16_ps3 = _mm512_loadu_ps(scaled_zp_b + 3 * BlockCountK);
            acc3 = _mm512_fmadd_ps(a_16_ps, b_16_ps3, acc3);
        }
        scaled_zp_a += PerAccuBlk16;
        scaled_zp_b += PerAccuBlk16;
    }

    if (k_blks_remaining > 0) {
      // TODO: the mask is fixed per gemm
        uint32_t mask = 0xffff >> (PerAccuBlk16 - k_blks_remaining);
        const __m512 a_16_ps = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_a);
        {
            const __m512 b_16_ps0 = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_b);
            acc0 = _mm512_fmadd_ps(a_16_ps, b_16_ps0, acc0);
        }
        {
            const __m512 b_16_ps1 = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_b + BlockCountK);
            acc1 = _mm512_fmadd_ps(a_16_ps, b_16_ps1, acc1);
        }
        {
            const __m512 b_16_ps2 = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_b + 2 * BlockCountK);
            acc2 = _mm512_fmadd_ps(a_16_ps, b_16_ps2, acc2);
        }
        {
            const __m512 b_16_ps3 = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_b + 3 * BlockCountK);
            acc3 = _mm512_fmadd_ps(a_16_ps, b_16_ps3, acc3);
        }
    }
}

static MLAS_FORCEINLINE void
accumulate_scaled_zp_prod_r1_c1(__m512& acc, const float* scaled_zp_a, const float* scaled_zp_b, size_t BlockCountK)
{
    constexpr size_t PerAccuBlk16 = 16;
    size_t k_blks_remaining = BlockCountK;
    // process 2 blks of 64 4b weights a time
    for (; k_blks_remaining >= PerAccuBlk16; k_blks_remaining -= PerAccuBlk16) {
        const __m512 a_16_ps = _mm512_loadu_ps(scaled_zp_a);
        const __m512 b_16_ps = _mm512_loadu_ps(scaled_zp_b);
        acc = _mm512_fmadd_ps(a_16_ps, b_16_ps, acc);
        scaled_zp_a += PerAccuBlk16;
        scaled_zp_b += PerAccuBlk16;
    }

    if (k_blks_remaining > 0) {
        // TODO: the mask is fixed per gemm
        uint32_t mask = 0xffff >> (PerAccuBlk16 - k_blks_remaining);
        const __m512 a_16_ps = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_a);
        const __m512 b_16_ps = _mm512_maskz_loadu_ps(__mmask16(mask), scaled_zp_b);
        acc = _mm512_fmadd_ps(a_16_ps, b_16_ps, acc);
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR2xC4BlkLen32Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ScaledZPA,
    const float* ScaledZPB
)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen32;
    const size_t StrideQuantBData = PerAccuBlk4 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM % NRows2 == 0);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        const float* scaled_zp_b = ScaledZPB;
        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc[NCols4 * NRows2] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            const float* scaled_zp_a = ScaledZPA + m * BlockCountK;
            accumulate_scaled_zp_prod_r1_c4(acc[0], acc[1], acc[2], acc[3], scaled_zp_a, scaled_zp_b, BlockCountK);
            accumulate_scaled_zp_prod_r1_c4(acc[4], acc[5], acc[6], acc[7], scaled_zp_a + BlockCountK, scaled_zp_b, BlockCountK);
            scaled_zp_b += 4 * BlockCountK;


            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));
                const __m512i av_10_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                const __m512i av_11_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + 64));

                if constexpr (vnni) {
                    accumulate_blklen32_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr,
                        acc[0], acc[NCols4]
                    );
                    accumulate_blklen32_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + PerAccuBlk4,
                        acc[1], acc[NCols4 + 1]
                    );
                    accumulate_blklen32_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * PerAccuBlk4,
                        acc[2], acc[NCols4 + 2]
                    );
                    accumulate_blklen32_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * PerAccuBlk4,
                        acc[3], acc[NCols4 + 3]
                    );
                } else {
                    accumulate_blklen32_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr,
                        acc[0], acc[NCols4]
                    );
                    accumulate_blklen32_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + PerAccuBlk4,
                        acc[1], acc[NCols4 + 1]
                    );
                    accumulate_blklen32_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * PerAccuBlk4,
                        acc[2], acc[NCols4 + 2]
                    );
                    accumulate_blklen32_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * PerAccuBlk4,
                        acc[3], acc[NCols4 + 3]
                    );
                }

                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += StrideQuantBData * NCols4;
                QuantBScalePtr += PerAccuBlk4 * NCols4;
            }  // k_blks_remaining

            __m256 acc2[NCols4 * NRows2] = {
                h_add_512(acc[0]),
                h_add_512(acc[1]),
                h_add_512(acc[2]),
                h_add_512(acc[3]),
                h_add_512(acc[4]),
                h_add_512(acc[5]),
                h_add_512(acc[6]),
                h_add_512(acc[7])
            };

            while (k_blks_remaining-- > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantABlk0 + lda));

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                {
                    // Col0
                    const float scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr)[0];
                    accumulate_blklen32_r2c1blk1_avx512<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr, scale_00, scale_10, acc2[0], acc2[NCols4]);
                }

                {
                    // Col1
                    const float scale_00 = scale_a00 * (QuantBScalePtr + 1)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + 1)[0];
                    accumulate_blklen32_r2c1blk1_avx512<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + BlkDataSizeInBytes16, scale_00, scale_10, acc2[1], acc2[NCols4 + 1]);
                }

                {
                    // Col2
                    const float scale_00 = scale_a00 * (QuantBScalePtr + 2)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + 2)[0];
                    accumulate_blklen32_r2c1blk1_avx512<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + 2 * BlkDataSizeInBytes16, scale_00, scale_10, acc2[2], acc2[NCols4 + 2]);
                }

                {
                    // Col3
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3)[0];
                    const float& scale_10 = scale_a10 * (QuantBScalePtr + 3)[0];
                    accumulate_blklen32_r2c1blk1_avx512<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + 3 * BlkDataSizeInBytes16, scale_00, scale_10, acc2[3], acc2[NCols4 + 3]);
                }
                QuantAPtr += BlkLen32;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes16 * NCols4;
                QuantBScalePtr += NCols4;
            }  // k_blks_remaining

            __m128 acc_r0 = FoldAccumulators(acc2[0], acc2[1], acc2[2], acc2[3]);
            __m128 acc_r1 = FoldAccumulators(acc2[NCols4 + 0], acc2[NCols4 + 1], acc2[NCols4 + 2], acc2[NCols4 + 3]);
            if (BiasPtr != nullptr) {
                const __m128 bias_4_ps = _mm_loadu_ps(BiasPtr);
                acc_r0 = _mm_add_ps(acc_r0, bias_4_ps);
                acc_r1 = _mm_add_ps(acc_r1, bias_4_ps);
            }
            _mm_storeu_ps(SumPtr, acc_r0);
            _mm_storeu_ps(SumPtr + ldc, acc_r1);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * BlockCountK * BlkDataSizeInBytes16;
            QuantBScaleColPtr += NCols4 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
void MLAS_FORCEINLINE
Q4Int8GemmR2C1BlkLen32Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ScaledZPA,
    const float* ScaledZPB
)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen32;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountM % NRows2 == 0);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        const float* scaled_zp_b = ScaledZPB;
        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();

            const float* scaled_zp_a = ScaledZPA + m * BlockCountK;
            accumulate_scaled_zp_prod_r1_c1(acc0, scaled_zp_a, scaled_zp_b, BlockCountK);
            accumulate_scaled_zp_prod_r1_c1(acc1, scaled_zp_a + BlockCountK, scaled_zp_b, BlockCountK);
            scaled_zp_b += BlockCountK;

            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));
                const __m512i av_10_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                const __m512i av_11_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + 64));

                if constexpr (vnni) {
                    accumulate_blklen32_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1
                    );
                } else {
                    accumulate_blklen32_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1
                    );
                }

                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            __m256 acc20 = h_add_512(acc0);
            __m256 acc21 = h_add_512(acc1);
            while (k_blks_remaining-- > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantABlk0 + lda));

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                const float& scale_10 = scale_a10 * (QuantBScalePtr)[0];
                accumulate_blklen32_r2c1blk1_avx512<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr, scale_00, scale_10, acc20, acc21);

                QuantAPtr += BlkLen32;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes16;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc20);
            *(SumPtr + ldc) = hsum_float_8(acc21);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
                *(SumPtr + ldc) += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR1xC4BlkLen32Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ScaledZPA,
    const float* ScaledZPB
)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen32;
    //const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        const float* scaled_zp_b = ScaledZPB;
        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc[NCols4] = {
              _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            const float* scaled_zp_a = ScaledZPA;
            accumulate_scaled_zp_prod_r1_c4(acc[0], acc[1], acc[2], acc[3], scaled_zp_a, scaled_zp_b, BlockCountK);
            scaled_zp_b += 4 * BlockCountK;

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));

                if constexpr (vnni) {
                    accumulate_blklen32_r1c1blk4_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen32_r1c1blk4_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + PerAccuBlk4, acc[1]);
                    accumulate_blklen32_r1c1blk4_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + 2 * PerAccuBlk4, acc[2]);
                    accumulate_blklen32_r1c1blk4_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + 3 * PerAccuBlk4, acc[3]);
                } else {
                    accumulate_blklen32_r1c1blk4_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen32_r1c1blk4_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + PerAccuBlk4, acc[1]);
                    accumulate_blklen32_r1c1blk4_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + 2 * PerAccuBlk4, acc[2]);
                    accumulate_blklen32_r1c1blk4_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * PerAccuBlk4 * BlkDataSizeInBytes16, QuantAScalePtr, QuantBScalePtr + 3 * PerAccuBlk4, acc[3]);
                }

                QuantAPtr += BlkLen32 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk4 * NCols4;
                QuantBScalePtr += PerAccuBlk4 * NCols4;
            }

            __m256 acc2[NCols4] = {
                h_add_512(acc[0]), h_add_512(acc[1]), h_add_512(acc[2]), h_add_512(acc[3])
            };

            while (k_blks_remaining-- > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

                const float& scale_a00 = *QuantAScalePtr;
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    accumulate_blklen32_r1c1blk1_avx512<vnni>(av_00_epi8, QuantBDataPtr, scale_00, acc2[0]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 1)[0];
                    accumulate_blklen32_r1c1blk1_avx512<vnni>(av_00_epi8, QuantBDataPtr + BlkDataSizeInBytes16, scale_00, acc2[1]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 2)[0];
                    accumulate_blklen32_r1c1blk1_avx512<vnni>(av_00_epi8, QuantBDataPtr + 2 * BlkDataSizeInBytes16, scale_00, acc2[2]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3)[0];
                    accumulate_blklen32_r1c1blk1_avx512<vnni>(av_00_epi8, QuantBDataPtr + 3 * BlkDataSizeInBytes16, scale_00, acc2[3]);
                }

                QuantAPtr += BlkLen32;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes16 * NCols4;
                QuantBScalePtr += NCols4;

            }

            __m128 acc_r0 = FoldAccumulators(acc2[0], acc2[1], acc2[2], acc2[3]);
            if (BiasPtr != nullptr) {
                acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
            }

            _mm_storeu_ps(SumPtr, acc_r0);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * BlockCountK * BlkDataSizeInBytes16;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR1xC1BlkLen32Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ScaledZPA,
    const float* ScaledZPB
)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen32;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    assert(CountM < NRows2);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        const float* scaled_zp_b = ScaledZPB;
        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m512 acc0 = _mm512_setzero_ps();
            const float* scaled_zp_a = ScaledZPA;
            accumulate_scaled_zp_prod_r1_c1(acc0, scaled_zp_a, scaled_zp_b, BlockCountK);
            scaled_zp_b += BlockCountK;

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));

                if constexpr (vnni) {
                    accumulate_blklen32_r1c1blk4_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0);
                }
                else {
                    accumulate_blklen32_r1c1blk4_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0);
                }

                QuantAPtr += BlkLen32 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            __m256 acc2 = h_add_512(acc0);
            while (k_blks_remaining-- > 0) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                accumulate_blklen32_r1c1blk1_avx512<vnni>(av_00_epi8, QuantBDataPtr, scale_00, acc2);

                QuantAPtr += BlkLen32;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes16;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc2);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template<bool vnni>
MLAS_FORCEINLINE
size_t
MlasQ4Int8GemmKernelBlkLen32Avx512(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ScaledZPA,
    const float* ScaledZPB
    )
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;

    const size_t lda = BlockCountK * BlkLen32 * sizeof(int8_t);
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q4Int8GemmR2xC4BlkLen32Avx512<vnni>(
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            multipleRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc,
            ScaledZPA,
            ScaledZPB
        );
    }
    if (remainingCols > 0 && multipleRows > 0) {
        Q4Int8GemmR2C1BlkLen32Avx512<vnni>(
            QuantA,
            QuantAScale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleCols,
            multipleRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc,
            ScaledZPA,
            ScaledZPB + multipleCols * BlockCountK
        );
    }

    if (remainingRows > 0 && multipleCols > 0) {
        Q4Int8GemmR1xC4BlkLen32Avx512<vnni>(
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData,
            QuantBScale,
            C + multipleRows * ldc,
            remainingRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc,
            ScaledZPA + multipleRows * BlockCountK,
            ScaledZPB
        );
    }

    if (remainingCols > 0 && remainingRows > 0) {
        Q4Int8GemmR1xC1BlkLen32Avx512<vnni>(
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleRows * ldc + multipleCols,
            remainingRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc,
            ScaledZPA + multipleRows * BlockCountK,
            ScaledZPB + multipleCols * BlockCountK
        );
    }

    return CountM;
}
