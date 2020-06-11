#include "qladd.h"

template <typename DataType>
static
MLAS_FORCEINLINE
__m256i
ShiftRight24Epi32(__m256i v);

template <> __m256i ShiftRight24Epi32<int8_t>(__m256i v) { return _mm256_srai_epi32(v, 24); }
template <> __m256i ShiftRight24Epi32<uint8_t>(__m256i v) { return _mm256_srli_epi32(v, 24); }

template <typename DataType>
MLAS_FORCEINLINE
static
__m256i
PackS16(__m256i a, __m256i b);

template <> __m256i PackS16<uint8_t>(__m256i a, __m256i b) { return _mm256_packus_epi16(a, b); }
template <> __m256i PackS16<int8_t>(__m256i a, __m256i b) { return _mm256_packs_epi16(a, b); }

template<typename DataType, bool IsScalarA, bool IsScalarB>
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

    if (IsScalarA) {
        const auto va_f32x8 = _mm256_set1_ps((float)(int32_t)*InputA);
        VectorFixedPart = _mm256_add_ps(VectorFixedPart, _mm256_mul_ps(va_f32x8, VectorScaleRatio_AC));
    }
    if (IsScalarB) {
        const auto vb_f32x8 = _mm256_set1_ps((float)(int32_t)*InputB);
        VectorFixedPart = _mm256_add_ps(VectorFixedPart, _mm256_mul_ps(vb_f32x8, VectorScaleRatio_BC));
    }

    int64_t n = static_cast<int64_t>(N);
    __m256i vc = _mm256_setzero_si256();
    while (n > 0) {
        __m256i va_i8x32, vb_i8x32, vc02, vc13;

        if (!IsScalarA) {
            va_i8x32 = _mm256_lddqu_si256((const __m256i*)InputA);
            InputA += 32;
        }
        if (!IsScalarB) {
            vb_i8x32 = _mm256_lddqu_si256((const __m256i*)InputB);
            InputB += 32;
        }

        __m256 lolo_f32x8, lohi_f32x8, hilo_f32x8, hihi_f32x8;

        if (IsScalarA) {
            const auto blo_i16x16 = _mm256_unpacklo_epi8(vb_i8x32, vb_i8x32);
            const auto bhi_i16x16 = _mm256_unpackhi_epi8(vb_i8x32, vb_i8x32);
            lolo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(blo_i16x16, blo_i16x16)));
            lohi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(blo_i16x16, blo_i16x16)));
            hilo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(bhi_i16x16, bhi_i16x16)));
            hihi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(bhi_i16x16, bhi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(lolo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            lohi_f32x8 = _mm256_fmadd_ps(lohi_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hilo_f32x8 = _mm256_fmadd_ps(hilo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hihi_f32x8 = _mm256_fmadd_ps(hihi_f32x8, VectorScaleRatio_BC, VectorFixedPart);
        } else if (IsScalarB) {
            const auto alo_i16x16 = _mm256_unpacklo_epi8(va_i8x32, va_i8x32);
            const auto ahi_i16x16 = _mm256_unpackhi_epi8(va_i8x32, va_i8x32);
            lolo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(alo_i16x16, alo_i16x16)));
            lohi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(alo_i16x16, alo_i16x16)));
            hilo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(ahi_i16x16, ahi_i16x16)));
            hihi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(ahi_i16x16, ahi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(lolo_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            lohi_f32x8 = _mm256_fmadd_ps(lohi_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            hilo_f32x8 = _mm256_fmadd_ps(hilo_f32x8, VectorScaleRatio_AC, VectorFixedPart);
            hihi_f32x8 = _mm256_fmadd_ps(hihi_f32x8, VectorScaleRatio_AC, VectorFixedPart);
        } else {
            const auto blo_i16x16 = _mm256_unpacklo_epi8(vb_i8x32, vb_i8x32);
            const auto bhi_i16x16 = _mm256_unpackhi_epi8(vb_i8x32, vb_i8x32);

            lolo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(blo_i16x16, blo_i16x16)));
            lohi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(blo_i16x16, blo_i16x16)));
            hilo_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(bhi_i16x16, bhi_i16x16)));
            hihi_f32x8 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(bhi_i16x16, bhi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(lolo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            lohi_f32x8 = _mm256_fmadd_ps(lohi_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hilo_f32x8 = _mm256_fmadd_ps(hilo_f32x8, VectorScaleRatio_BC, VectorFixedPart);
            hihi_f32x8 = _mm256_fmadd_ps(hihi_f32x8, VectorScaleRatio_BC, VectorFixedPart);

            const auto alo_i16x16 = _mm256_unpacklo_epi8(va_i8x32, va_i8x32);
            const auto alolo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(alo_i16x16, alo_i16x16)));
            const auto alohi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(alo_i16x16, alo_i16x16)));
            const auto ahi_i16x16 = _mm256_unpackhi_epi8(va_i8x32, va_i8x32);
            const auto ahilo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(ahi_i16x16, ahi_i16x16)));
            const auto ahihi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(ahi_i16x16, ahi_i16x16)));
            lolo_f32x8 = _mm256_fmadd_ps(alolo_8xfp32, VectorScaleRatio_AC, lolo_f32x8);
            lohi_f32x8 = _mm256_fmadd_ps(alohi_8xfp32, VectorScaleRatio_AC, lohi_f32x8);
            hilo_f32x8 = _mm256_fmadd_ps(ahilo_8xfp32, VectorScaleRatio_AC, hilo_f32x8);
            hihi_f32x8 = _mm256_fmadd_ps(ahihi_8xfp32, VectorScaleRatio_AC, hihi_f32x8);
        }

        vc02 = _mm256_packs_epi32(_mm256_cvtps_epi32(lolo_f32x8), _mm256_cvtps_epi32(lohi_f32x8));
        vc13 = _mm256_packs_epi32(_mm256_cvtps_epi32(hilo_f32x8), _mm256_cvtps_epi32(hihi_f32x8));

        vc = PackS16<DataType>(vc02, vc13);

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

