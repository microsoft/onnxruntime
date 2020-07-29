/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qladd_avx2.cpp

Abstract:

    This module implements routines to quantize linear add using avx2 intrinsics.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "../../mlasi.h"
#include "../../qladd.h"

template <typename DataType>
MLAS_FORCEINLINE
static
__m256i
MlasShiftRight24Epi32(
    __m256i v
    );

template <>
__m256i
MlasShiftRight24Epi32<int8_t>(
    __m256i v
    )
{
    return _mm256_srai_epi32(v, 24);
}

template <>
__m256i
MlasShiftRight24Epi32<uint8_t>(
    __m256i v
    )
{
    return _mm256_srli_epi32(v, 24);
}

template <typename DataType>
MLAS_FORCEINLINE
static
__m256i
MlasPackS16_256(
    __m256i a,
    __m256i b
    );

template <>
__m256i
MlasPackS16_256<uint8_t>(
    __m256i a,
    __m256i b
    )
{
    return _mm256_packus_epi16(a, b);
}

template <>
__m256i
MlasPackS16_256<int8_t>(
    __m256i a,
    __m256i b
    )
{
    return _mm256_packs_epi16(a, b);
}

MLAS_FORCEINLINE
static
__m256i
MlasLoad32Bytes(const uint8_t* buffer, int64_t N)
{
    if (N >= 32) {
        return _mm256_lddqu_si256((const __m256i*)buffer);
    } else {
        uint8_t dup[32];
        MlasCopyTailBytes(dup, buffer, (size_t)N);
        return _mm256_lddqu_si256((const __m256i*)dup);
    }
}

template <typename DataType, bool IsScalarB>
static
void
MlasQLinearAddKernelAvx2Helper(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    const float ScaleRatio_AC = ScaleA / ScaleC;
    const float ScaleRatio_BC = ScaleB / ScaleC;
    const __m256 VectorScaleRatio_AC = _mm256_set1_ps(ScaleRatio_AC);
    const __m256 VectorScaleRatio_BC = _mm256_set1_ps(ScaleRatio_BC);
    __m256 VectorFixedPart = _mm256_set1_ps((float)ZeroPointC - (ScaleRatio_AC * ZeroPointA + ScaleRatio_BC * ZeroPointB));

    if (IsScalarB) {
        const auto vb_f32x8 = _mm256_set1_ps((float)(int32_t)*InputB);
        VectorFixedPart = _mm256_add_ps(VectorFixedPart, _mm256_mul_ps(vb_f32x8, VectorScaleRatio_BC));
    }

    int64_t n = static_cast<int64_t>(N);
    __m256i vc = _mm256_setzero_si256();
    while (n > 0) {
        __m256i va_i8x32, vb_i8x32;
        va_i8x32 = MlasLoad32Bytes((const uint8_t*)InputA, n);
        InputA += 32;

        if (!IsScalarB) {
            vb_i8x32 = MlasLoad32Bytes((const uint8_t*)InputB, n);
            InputB += 32;
        }

        __m256 lolo_f32x8, lohi_f32x8, hilo_f32x8, hihi_f32x8;
        if (IsScalarB) {
            const auto alo_i16x16 = _mm256_unpacklo_epi8(va_i8x32, va_i8x32);
            const auto ahi_i16x16 = _mm256_unpackhi_epi8(va_i8x32, va_i8x32);
            lolo_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(alo_i16x16, alo_i16x16)));
            lohi_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(alo_i16x16, alo_i16x16)));
            hilo_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(ahi_i16x16, ahi_i16x16)));
            hihi_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(ahi_i16x16, ahi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(lolo_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            lohi_f32x8 = _mm256_fmadd_ps(lohi_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            hilo_f32x8 = _mm256_fmadd_ps(hilo_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            hihi_f32x8 = _mm256_fmadd_ps(hihi_f32x8, VectorScaleRatio_AC, VectorFixedPart);
        } else {
            const auto blo_i16x16 = _mm256_unpacklo_epi8(vb_i8x32, vb_i8x32);
            const auto bhi_i16x16 = _mm256_unpackhi_epi8(vb_i8x32, vb_i8x32);
            lolo_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(blo_i16x16, blo_i16x16)));
            lohi_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(blo_i16x16, blo_i16x16)));
            hilo_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(bhi_i16x16, bhi_i16x16)));
            hihi_f32x8 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(bhi_i16x16, bhi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(lolo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            lohi_f32x8 = _mm256_fmadd_ps(lohi_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hilo_f32x8 = _mm256_fmadd_ps(hilo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hihi_f32x8 = _mm256_fmadd_ps(hihi_f32x8, VectorScaleRatio_BC, VectorFixedPart);

            const auto alo_i16x16 = _mm256_unpacklo_epi8(va_i8x32, va_i8x32);
            const auto alolo_8xfp32 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(alo_i16x16, alo_i16x16)));
            const auto alohi_8xfp32 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(alo_i16x16, alo_i16x16)));
            const auto ahi_i16x16 = _mm256_unpackhi_epi8(va_i8x32, va_i8x32);
            const auto ahilo_8xfp32 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(ahi_i16x16, ahi_i16x16)));
            const auto ahihi_8xfp32 = _mm256_cvtepi32_ps(MlasShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(ahi_i16x16, ahi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(alolo_8xfp32, VectorScaleRatio_AC, lolo_f32x8);
            lohi_f32x8 = _mm256_fmadd_ps(alohi_8xfp32, VectorScaleRatio_AC, lohi_f32x8);
            hilo_f32x8 = _mm256_fmadd_ps(ahilo_8xfp32, VectorScaleRatio_AC, hilo_f32x8);
            hihi_f32x8 = _mm256_fmadd_ps(ahihi_8xfp32, VectorScaleRatio_AC, hihi_f32x8);
        }

        const auto vc02 = _mm256_packs_epi32(_mm256_cvtps_epi32(lolo_f32x8), _mm256_cvtps_epi32(lohi_f32x8));
        const auto vc13 = _mm256_packs_epi32(_mm256_cvtps_epi32(hilo_f32x8), _mm256_cvtps_epi32(hihi_f32x8));
        vc = MlasPackS16_256<DataType>(vc02, vc13);

        n -= 32;
        if (n < 0) break;

        _mm256_storeu_si256((__m256i*)OutputC, vc);
        OutputC += 32;
    }

    if (n < 0) {
        n += 32;
        int k = static_cast<int>(n / 4);
        if (k > 0) {
            const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(k), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            _mm256_maskstore_epi32((int*)OutputC, mask, vc);
            OutputC += k * 4;
        }

        int r = static_cast<int>(n % 4);
        if (r > 0) {
            auto permuted = _mm256_permutevar8x32_epi32(vc, _mm256_set1_epi32(k));
            uint32_t PackedValueC = (uint32_t)_mm256_extract_epi32(permuted, 0);
            for (int i = 0; i < r; ++i) {
                *((uint8_t*)OutputC + i) = (uint8_t)PackedValueC;
                PackedValueC >>= 8;
            }
        }
    }
}

void
MLASCALL
MlasQLinearAddS8KernelAvx2(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t N,
    bool IsScalarB
    )
{
    if (IsScalarB) {
        MlasQLinearAddKernelAvx2Helper<int8_t, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearAddKernelAvx2Helper<int8_t, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}

void
MLASCALL
MlasQLinearAddU8KernelAvx2(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t N,
    bool IsScalarB
    )
{
    if (IsScalarB) {
        MlasQLinearAddKernelAvx2Helper<uint8_t, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearAddKernelAvx2Helper<uint8_t, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}
