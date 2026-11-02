// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*++

Module Name:

    t8gemm_kernel_avx2.cpp

Abstract:

    AVX2 T8 ternary GEMM kernel.

    Two groups (eight plane words) are decoded per 256-bit vector. To use a
    decoded word on eight consecutive inputs, the word is broadcast and each
    lane shifts its own bit into the sign position: an arithmetic shift turns
    the nonzero bit into a lane mask, and the negative bit becomes the sign
    flipped into A.

--*/

#include "t8gemm_kernel.h"

namespace
{

struct T8OpsAvx2 {
    using V = __m256i;

    static MLAS_FORCEINLINE V Not(V a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

    static MLAS_FORCEINLINE V Nt(V a, V b, V c)
    {
        return _mm256_and_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(b, c));
    }
    static MLAS_FORCEINLINE V Pt(V a, V b, V c, V nt)
    {
        return _mm256_andnot_si256(nt, _mm256_xor_si256(_mm256_xor_si256(a, b), c));
    }
    static MLAS_FORCEINLINE V Bq(V a, V b, V c)
    {
        return _mm256_and_si256(_mm256_xor_si256(a, b), _mm256_or_si256(a, c));
    }
    static MLAS_FORCEINLINE V Neg(V a, V b, V c)
    {
        return _mm256_xor_si256(b, _mm256_and_si256(_mm256_xor_si256(a, b), c));
    }
    static MLAS_FORCEINLINE V Pos(V a, V b, V c) { return _mm256_andnot_si256(b, _mm256_xor_si256(a, c)); }
    static MLAS_FORCEINLINE V Zero(V a, V b, V c) { return Not(_mm256_or_si256(a, _mm256_xor_si256(b, c))); }
    static MLAS_FORCEINLINE V Carry(V a, V b, V c) { return _mm256_or_si256(a, _mm256_and_si256(b, c)); }
    static MLAS_FORCEINLINE V And(V a, V b) { return _mm256_and_si256(a, b); }
    static MLAS_FORCEINLINE V NotOr(V a, V b) { return Not(_mm256_or_si256(a, b)); }
};

constexpr size_t GroupsPerBlock = 2;

//
// Loads plane b of groups [g0, g0 + ng) of one unit; lanes of a missing group
// are zero.
//

MLAS_FORCEINLINE __m256i
LoadPlane(const uint32_t* B, MLAS_T8_WEIGHT_LAYOUT Layout, size_t K, size_t unit, size_t g0, size_t ng, size_t b)
{
    const uint32_t* p = T8PlaneWords(B, Layout, K, unit, g0, b);
    if (Layout == MLAS_T8_WEIGHT_LAYOUT::Planes && ng == GroupsPerBlock) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    const __m256i lo = _mm256_zextsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    if (ng == 1) {
        return lo;
    }
    const size_t stride = Layout == MLAS_T8_WEIGHT_LAYOUT::Planes ? T8WordsPerGroup : T8TiledWordsPerGroup;
    return _mm256_inserti128_si256(lo, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride)), 1);
}

MLAS_FORCEINLINE float
ReduceAdd(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

//
// Accumulates one 128-input group for Rows rows. Nz[j] and Neg[j] point to the
// four nonzero and negative mask words of trit j.
//

template <size_t Rows>
MLAS_FORCEINLINE void
ComputeGroup(
    const float* A,
    size_t K,
    const uint32_t* const* Nz,
    const uint32_t* const* Neg,
    const MLAS_FP16* Scales,
    __m256 (&Totals)[Rows][T8TritsPerUnit]
)
{
    // Lane l of chunk c holds input 8 * c + l of a word; shifting by
    // 31 - (8 * c + l) moves that input's bit into the sign position.
    const __m256i shifts[4] = {
        _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24),
        _mm256_setr_epi32(23, 22, 21, 20, 19, 18, 17, 16),
        _mm256_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8),
        _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    };
    const __m256i sign = _mm256_set1_epi32(static_cast<int>(0x80000000u));

    for (size_t j = 0; j < T8TritsPerUnit; ++j) {
        __m256 sums[Rows][2];
        for (size_t r = 0; r < Rows; ++r) {
            sums[r][0] = _mm256_setzero_ps();
            sums[r][1] = _mm256_setzero_ps();
        }
        for (size_t w = 0; w < T8WordsPerGroup; ++w) {
            const __m256i nonzero_bits = _mm256_set1_epi32(static_cast<int>(Nz[j][w]));
            const __m256i negative_bits = _mm256_set1_epi32(static_cast<int>(Neg[j][w]));
            for (size_t c = 0; c < 4; ++c) {
                const __m256 nonzero = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_sllv_epi32(nonzero_bits, shifts[c]), 31));
                const __m256 negative = _mm256_castsi256_ps(_mm256_and_si256(_mm256_sllv_epi32(negative_bits, shifts[c]), sign));
                for (size_t r = 0; r < Rows; ++r) {
                    const __m256 x = _mm256_loadu_ps(A + r * K + w * 32 + c * 8);
                    sums[r][c & 1] = _mm256_add_ps(sums[r][c & 1], _mm256_and_ps(_mm256_xor_ps(x, negative), nonzero));
                }
            }
        }
        const __m256 scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(Scales[j].val)));
        for (size_t r = 0; r < Rows; ++r) {
            Totals[r][j] = _mm256_fmadd_ps(_mm256_add_ps(sums[r][0], sums[r][1]), scale, Totals[r][j]);
        }
    }
}

template <size_t Rows>
void
T8GemmUnits(
    size_t N,
    size_t K,
    size_t UnitBegin,
    size_t UnitEnd,
    const float* A,
    const uint32_t* B,
    const MLAS_FP16* Scales,
    float* C,
    MLAS_T8_WEIGHT_LAYOUT Layout
)
{
    const size_t groups = K / T8GroupSize;
    alignas(32) uint32_t nz_words[T8TritsPerUnit][GroupsPerBlock * T8WordsPerGroup];
    alignas(32) uint32_t neg_words[T8TritsPerUnit][GroupsPerBlock * T8WordsPerGroup];

    for (size_t unit = UnitBegin; unit < UnitEnd; ++unit) {
        __m256 totals[Rows][T8TritsPerUnit];
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t j = 0; j < T8TritsPerUnit; ++j) totals[r][j] = _mm256_setzero_ps();
        }
        const MLAS_FP16* scales = Scales + unit * groups * T8TritsPerUnit;

        for (size_t g0 = 0; g0 < groups; g0 += GroupsPerBlock) {
            const size_t ng = std::min(GroupsPerBlock, groups - g0);
            if (Layout != MLAS_T8_WEIGHT_LAYOUT::Masks) {
                __m256i z[T8TritsPerUnit], p[T8TritsPerUnit];
                T8Decode<T8OpsAvx2>(
                    LoadPlane(B, Layout, K, unit, g0, ng, 0), LoadPlane(B, Layout, K, unit, g0, ng, 1),
                    LoadPlane(B, Layout, K, unit, g0, ng, 2), LoadPlane(B, Layout, K, unit, g0, ng, 3),
                    LoadPlane(B, Layout, K, unit, g0, ng, 4), LoadPlane(B, Layout, K, unit, g0, ng, 5),
                    LoadPlane(B, Layout, K, unit, g0, ng, 6), LoadPlane(B, Layout, K, unit, g0, ng, 7), z, p
                );
                for (size_t j = 0; j < T8TritsPerUnit; ++j) {
                    const __m256i nonzero = T8OpsAvx2::Not(z[j]);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(nz_words[j]), nonzero);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(neg_words[j]), _mm256_andnot_si256(p[j], nonzero));
                }
            }
            for (size_t gg = 0; gg < ng; ++gg) {
                const size_t g = g0 + gg;
                const uint32_t* nz[T8TritsPerUnit];
                const uint32_t* neg[T8TritsPerUnit];
                for (size_t j = 0; j < T8TritsPerUnit; ++j) {
                    if (Layout == MLAS_T8_WEIGHT_LAYOUT::Masks) {
                        const uint32_t* block = B + (unit * groups + g) * T8MaskWordsPerGroup;
                        nz[j] = block + j * 8;
                        neg[j] = block + j * 8 + 4;
                    } else {
                        nz[j] = nz_words[j] + gg * T8WordsPerGroup;
                        neg[j] = neg_words[j] + gg * T8WordsPerGroup;
                    }
                }
                ComputeGroup<Rows>(A + g * T8GroupSize, K, nz, neg, scales + g * T8TritsPerUnit, totals);
            }
        }

        const size_t count = std::min(T8TritsPerUnit, N - unit * T8TritsPerUnit);
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t j = 0; j < count; ++j) {
                C[r * N + unit * T8TritsPerUnit + j] = ReduceAdd(totals[r][j]);
            }
        }
    }
}

}  // namespace

void
    MLASCALL
    MlasT8GemmKernelAvx2(
        size_t Rows,
        size_t N,
        size_t K,
        size_t UnitBegin,
        size_t UnitEnd,
        const float* A,
        const uint32_t* B,
        const MLAS_FP16* Scales,
        float* C,
        MLAS_T8_WEIGHT_LAYOUT Layout
    )
{
    if (Rows == 2) {
        T8GemmUnits<2>(N, K, UnitBegin, UnitEnd, A, B, Scales, C, Layout);
    } else {
        T8GemmUnits<1>(N, K, UnitBegin, UnitEnd, A, B, Scales, C, Layout);
    }
}
