// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*++

Module Name:

    t8gemm_kernel_avx512f.cpp

Abstract:

    AVX-512F T8 ternary GEMM kernel.

    Four groups (sixteen plane words) are decoded per 512-bit vector with
    vpternlogd gates. Each 16-bit half of a decoded word is used directly as an
    opmask: the sign mask flips A with a masked xor and the nonzero mask selects
    the lanes added into the accumulator.

--*/

#include "t8gemm_kernel.h"

namespace
{

struct T8OpsAvx512 {
    using V = __m512i;

    template <uint8_t Imm>
    static MLAS_FORCEINLINE V Tern(V a, V b, V c)
    {
        return _mm512_ternarylogic_epi32(a, b, c, Imm);
    }

    static MLAS_FORCEINLINE V Nt(V a, V b, V c)
    {
        return Tern<uint8_t((T8TernA ^ T8TernB) & (T8TernB ^ T8TernC))>(a, b, c);
    }
    static MLAS_FORCEINLINE V Pt(V a, V b, V c, V nt)
    {
        return _mm512_andnot_si512(nt, Tern<uint8_t(T8TernA ^ T8TernB ^ T8TernC)>(a, b, c));
    }
    static MLAS_FORCEINLINE V Bq(V a, V b, V c)
    {
        return Tern<uint8_t((T8TernA ^ T8TernB) & (T8TernA | T8TernC))>(a, b, c);
    }
    static MLAS_FORCEINLINE V Neg(V a, V b, V c)
    {
        return Tern<uint8_t(T8TernB ^ ((T8TernA ^ T8TernB) & T8TernC))>(a, b, c);
    }
    static MLAS_FORCEINLINE V Pos(V a, V b, V c)
    {
        return Tern<uint8_t((T8TernA ^ T8TernC) & ~T8TernB)>(a, b, c);
    }
    static MLAS_FORCEINLINE V Zero(V a, V b, V c)
    {
        return Tern<uint8_t(~(T8TernA | (T8TernB ^ T8TernC)))>(a, b, c);
    }
    static MLAS_FORCEINLINE V Carry(V a, V b, V c)
    {
        return Tern<uint8_t(T8TernA | (T8TernB & T8TernC))>(a, b, c);
    }
    static MLAS_FORCEINLINE V And(V a, V b) { return _mm512_and_si512(a, b); }
    static MLAS_FORCEINLINE V NotOr(V a, V b) { return Tern<uint8_t(~(T8TernA | T8TernB))>(a, b, b); }
};

constexpr size_t GroupsPerBlock = 4;

//
// Loads plane b of groups [g0, g0 + ng) of one unit; lanes of missing groups
// are zero.
//

MLAS_FORCEINLINE __m512i
LoadPlane(const uint32_t* B, MLAS_T8_WEIGHT_LAYOUT Layout, size_t K, size_t unit, size_t g0, size_t ng, size_t b)
{
    const uint32_t* p = T8PlaneWords(B, Layout, K, unit, g0, b);
    if (Layout == MLAS_T8_WEIGHT_LAYOUT::Planes) {
        if (ng == GroupsPerBlock) {
            return _mm512_loadu_si512(p);
        }
        return _mm512_maskz_loadu_epi32(static_cast<__mmask16>((1u << (ng * T8WordsPerGroup)) - 1), p);
    }
    __m512i v = _mm512_zextsi128_si512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    if (ng > 1) v = _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + T8TiledWordsPerGroup)), 1);
    if (ng > 2) v = _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2 * T8TiledWordsPerGroup)), 2);
    if (ng > 3) v = _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 3 * T8TiledWordsPerGroup)), 3);
    return v;
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
    __m512 (&Totals)[Rows][T8TritsPerUnit]
)
{
    // Two accumulators per trit hide the add latency. Every row tile uses the
    // same split, so results do not depend on the row tile.
    constexpr size_t Chains = 2;
    const __m512i sign = _mm512_set1_epi32(static_cast<int>(0x80000000u));
    __m512 sums[Rows][T8TritsPerUnit][Chains];
    for (size_t r = 0; r < Rows; ++r) {
        for (size_t j = 0; j < T8TritsPerUnit; ++j) {
            for (size_t c = 0; c < Chains; ++c) sums[r][j][c] = _mm512_setzero_ps();
        }
    }
    for (size_t w = 0; w < T8WordsPerGroup; ++w) {
        for (size_t h = 0; h < 2; ++h) {
            __m512i x[Rows];
            for (size_t r = 0; r < Rows; ++r) {
                x[r] = _mm512_castps_si512(_mm512_loadu_ps(A + r * K + w * 32 + h * 16));
            }
            for (size_t j = 0; j < T8TritsPerUnit; ++j) {
                const __mmask16 nonzero = static_cast<__mmask16>(Nz[j][w] >> (16 * h));
                const __mmask16 negative = static_cast<__mmask16>(Neg[j][w] >> (16 * h));
                for (size_t r = 0; r < Rows; ++r) {
                    const __m512 xs = _mm512_castsi512_ps(_mm512_mask_xor_epi32(x[r], negative, x[r], sign));
                    __m512& sum = sums[r][j][h];
                    sum = _mm512_mask_add_ps(sum, nonzero, sum, xs);
                }
            }
        }
    }
    for (size_t j = 0; j < T8TritsPerUnit; ++j) {
        const __m512 scale = _mm512_cvtph_ps(_mm256_set1_epi16(static_cast<short>(Scales[j].val)));
        for (size_t r = 0; r < Rows; ++r) {
            const __m512 sum = _mm512_add_ps(sums[r][j][0], sums[r][j][1]);
            Totals[r][j] = _mm512_fmadd_ps(sum, scale, Totals[r][j]);
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
    alignas(64) uint32_t nz_words[T8TritsPerUnit][GroupsPerBlock * T8WordsPerGroup];
    alignas(64) uint32_t neg_words[T8TritsPerUnit][GroupsPerBlock * T8WordsPerGroup];

    for (size_t unit = UnitBegin; unit < UnitEnd; ++unit) {
        __m512 totals[Rows][T8TritsPerUnit];
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t j = 0; j < T8TritsPerUnit; ++j) totals[r][j] = _mm512_setzero_ps();
        }
        const MLAS_FP16* scales = Scales + unit * groups * T8TritsPerUnit;

        for (size_t g0 = 0; g0 < groups; g0 += GroupsPerBlock) {
            const size_t ng = std::min(GroupsPerBlock, groups - g0);
            if (Layout != MLAS_T8_WEIGHT_LAYOUT::Masks) {
                __m512i z[T8TritsPerUnit], p[T8TritsPerUnit];
                T8Decode<T8OpsAvx512>(
                    LoadPlane(B, Layout, K, unit, g0, ng, 0), LoadPlane(B, Layout, K, unit, g0, ng, 1),
                    LoadPlane(B, Layout, K, unit, g0, ng, 2), LoadPlane(B, Layout, K, unit, g0, ng, 3),
                    LoadPlane(B, Layout, K, unit, g0, ng, 4), LoadPlane(B, Layout, K, unit, g0, ng, 5),
                    LoadPlane(B, Layout, K, unit, g0, ng, 6), LoadPlane(B, Layout, K, unit, g0, ng, 7), z, p
                );
                for (size_t j = 0; j < T8TritsPerUnit; ++j) {
                    const __m512i nonzero = _mm512_ternarylogic_epi32(z[j], z[j], z[j], uint8_t(~T8TernA));
                    _mm512_store_si512(nz_words[j], nonzero);
                    _mm512_store_si512(neg_words[j], _mm512_andnot_si512(p[j], nonzero));
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
                C[r * N + unit * T8TritsPerUnit + j] = _mm512_reduce_add_ps(totals[r][j]);
            }
        }
    }
}

}  // namespace

void
    MLASCALL
    MlasT8GemmKernelAvx512F(
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
