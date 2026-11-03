/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemv_avx2.cpp

Abstract:

    This module implements AVX2/F16C kernels for M=1 half precision GEMM.

--*/

#include <cmath>

#include "../../mlasi.h"

namespace
{

MLAS_FORCEINLINE
__m256
LoadHalf8(
    const MLAS_FP16* Source
)
{
    return _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(Source))
    );
}

MLAS_FORCEINLINE
void
StoreHalf8(
    MLAS_FP16* Destination,
    __m256 Value
)
{
    const __m128i HalfValue = _mm256_cvtps_ph(Value, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(Destination), HalfValue);
}

MLAS_FORCEINLINE
float
LoadHalf1(
    const MLAS_FP16* Source
)
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(Source->val)));
}

MLAS_FORCEINLINE
__m256
ApplyScale(
    __m256 Sum,
    const MLAS_FP16* C,
    float Alpha,
    float Beta
)
{
    if (Alpha != 1.0f) {
        Sum = _mm256_mul_ps(Sum, _mm256_set1_ps(Alpha));
    }

    if (Beta == 0.0f) {
        return Sum;
    }

    const __m256 Existing = LoadHalf8(C);
    if (Beta == 1.0f) {
        return _mm256_add_ps(Sum, Existing);
    }

    return _mm256_fmadd_ps(Existing, _mm256_set1_ps(Beta), Sum);
}

MLAS_FORCEINLINE
float
ApplyScale(
    float Sum,
    const MLAS_FP16* C,
    float Alpha,
    float Beta
)
{
    const float Existing = Beta == 0.0f ? 0.0f : float(*C);
    return std::fma(Beta, Existing, Alpha * Sum);
}

void
HalfGemmDecodeNoTrans(
    size_t K,
    const MLAS_HGEMM_DATA_PARAMS* Data,
    size_t StartN,
    size_t CountN
)
{
    const MLAS_FP16* B = Data->B + StartN;
    MLAS_FP16* C = Data->C + StartN;
    const size_t ldb = Data->ldb;
    const float Alpha = float(MLAS_FP16::FromBits(Data->alpha));
    const float Beta = float(MLAS_FP16::FromBits(Data->beta));

    size_t n = 0;
    while (n + 64 <= CountN) {
        __m256 Sum0 = _mm256_setzero_ps();
        __m256 Sum1 = _mm256_setzero_ps();
        __m256 Sum2 = _mm256_setzero_ps();
        __m256 Sum3 = _mm256_setzero_ps();
        __m256 Sum4 = _mm256_setzero_ps();
        __m256 Sum5 = _mm256_setzero_ps();
        __m256 Sum6 = _mm256_setzero_ps();
        __m256 Sum7 = _mm256_setzero_ps();

        const MLAS_FP16* BRow = B + n;
        for (size_t k = 0; k < K; ++k) {
            const __m256 AValue = _mm256_set1_ps(LoadHalf1(Data->A + k));
            Sum0 = _mm256_fmadd_ps(LoadHalf8(BRow), AValue, Sum0);
            Sum1 = _mm256_fmadd_ps(LoadHalf8(BRow + 8), AValue, Sum1);
            Sum2 = _mm256_fmadd_ps(LoadHalf8(BRow + 16), AValue, Sum2);
            Sum3 = _mm256_fmadd_ps(LoadHalf8(BRow + 24), AValue, Sum3);
            Sum4 = _mm256_fmadd_ps(LoadHalf8(BRow + 32), AValue, Sum4);
            Sum5 = _mm256_fmadd_ps(LoadHalf8(BRow + 40), AValue, Sum5);
            Sum6 = _mm256_fmadd_ps(LoadHalf8(BRow + 48), AValue, Sum6);
            Sum7 = _mm256_fmadd_ps(LoadHalf8(BRow + 56), AValue, Sum7);
            BRow += ldb;
        }

        StoreHalf8(C + n, ApplyScale(Sum0, C + n, Alpha, Beta));
        StoreHalf8(C + n + 8, ApplyScale(Sum1, C + n + 8, Alpha, Beta));
        StoreHalf8(C + n + 16, ApplyScale(Sum2, C + n + 16, Alpha, Beta));
        StoreHalf8(C + n + 24, ApplyScale(Sum3, C + n + 24, Alpha, Beta));
        StoreHalf8(C + n + 32, ApplyScale(Sum4, C + n + 32, Alpha, Beta));
        StoreHalf8(C + n + 40, ApplyScale(Sum5, C + n + 40, Alpha, Beta));
        StoreHalf8(C + n + 48, ApplyScale(Sum6, C + n + 48, Alpha, Beta));
        StoreHalf8(C + n + 56, ApplyScale(Sum7, C + n + 56, Alpha, Beta));
        n += 64;
    }

    while (n + 8 <= CountN) {
        __m256 Sum = _mm256_setzero_ps();
        const MLAS_FP16* BRow = B + n;
        for (size_t k = 0; k < K; ++k) {
            const __m256 AValue = _mm256_set1_ps(LoadHalf1(Data->A + k));
            Sum = _mm256_fmadd_ps(LoadHalf8(BRow), AValue, Sum);
            BRow += ldb;
        }
        StoreHalf8(C + n, ApplyScale(Sum, C + n, Alpha, Beta));
        n += 8;
    }

    while (n < CountN) {
        float Sum = 0.0f;
        const MLAS_FP16* BValue = B + n;
        for (size_t k = 0; k < K; ++k) {
            Sum = std::fma(LoadHalf1(Data->A + k), LoadHalf1(BValue), Sum);
            BValue += ldb;
        }
        C[n] = MLAS_FP16(ApplyScale(Sum, C + n, Alpha, Beta));
        ++n;
    }
}

MLAS_FORCEINLINE
float
ReduceAdd(
    __m256 Value
)
{
    __m128 Sum = _mm_add_ps(
        _mm256_castps256_ps128(Value),
        _mm256_extractf128_ps(Value, 1)
    );
    Sum = _mm_hadd_ps(Sum, Sum);
    Sum = _mm_hadd_ps(Sum, Sum);
    return _mm_cvtss_f32(Sum);
}

void
HalfGemmDecodeTrans(
    size_t K,
    const MLAS_HGEMM_DATA_PARAMS* Data,
    size_t StartN,
    size_t CountN
)
{
    MLAS_FP16* C = Data->C + StartN;
    const float Alpha = float(MLAS_FP16::FromBits(Data->alpha));
    const float Beta = float(MLAS_FP16::FromBits(Data->beta));

    for (size_t n = 0; n < CountN; ++n) {
        const MLAS_FP16* BRow = Data->B + (StartN + n) * Data->ldb;
        __m256 Sum0 = _mm256_setzero_ps();
        __m256 Sum1 = _mm256_setzero_ps();
        __m256 Sum2 = _mm256_setzero_ps();
        __m256 Sum3 = _mm256_setzero_ps();

        size_t k = 0;
        while (k + 32 <= K) {
            Sum0 = _mm256_fmadd_ps(LoadHalf8(Data->A + k), LoadHalf8(BRow + k), Sum0);
            Sum1 = _mm256_fmadd_ps(LoadHalf8(Data->A + k + 8), LoadHalf8(BRow + k + 8), Sum1);
            Sum2 = _mm256_fmadd_ps(LoadHalf8(Data->A + k + 16), LoadHalf8(BRow + k + 16), Sum2);
            Sum3 = _mm256_fmadd_ps(LoadHalf8(Data->A + k + 24), LoadHalf8(BRow + k + 24), Sum3);
            k += 32;
        }

        while (k + 8 <= K) {
            Sum0 = _mm256_fmadd_ps(LoadHalf8(Data->A + k), LoadHalf8(BRow + k), Sum0);
            k += 8;
        }

        float Sum = ReduceAdd(
            _mm256_add_ps(_mm256_add_ps(Sum0, Sum1), _mm256_add_ps(Sum2, Sum3))
        );
        while (k < K) {
            Sum = std::fma(LoadHalf1(Data->A + k), LoadHalf1(BRow + k), Sum);
            ++k;
        }
        C[n] = MLAS_FP16(ApplyScale(Sum, C + n, Alpha, Beta));
    }
}

}  // namespace

void
    MLASCALL
    MlasHalfGemmDecodeKernelAvx2(
        CBLAS_TRANSPOSE TransB,
        size_t N,
        size_t K,
        const MLAS_HGEMM_DATA_PARAMS* Data,
        size_t StartN,
        size_t CountN
    )
{
    MLAS_UNREFERENCED_PARAMETER(N);

    if (TransB == CblasNoTrans) {
        HalfGemmDecodeNoTrans(K, Data, StartN, CountN);
    } else {
        HalfGemmDecodeTrans(K, Data, StartN, CountN);
    }

    _mm256_zeroupper();
}
