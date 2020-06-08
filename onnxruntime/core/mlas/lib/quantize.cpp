/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

uint32_t BitsOfFp32(float f) {
    union {
        uint32_t u32;
        float    fp32;
    } uf;
    uf.fp32 = f;
    return uf.u32;
}

float Fp32FromBits(uint32_t u) {
    union {
        uint32_t u32;
        float    fp32;
    } uf = { u };
    return uf.fp32;
}

#if defined(MLAS_NEON64_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS)

//
// QuantizeLinear implementation using NEON or SSE2 intrinsics.
//

MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearVector(
    MLAS_FLOAT32X4 FloatVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasDivideFloat32x4(FloatVector, ScaleVector);

#if defined(MLAS_NEON64_INTRINSICS)
    // N.B. FMINNM and FMAXNM returns the numeric value if either of the values
    // is a NaN.
    FloatVector = vmaxnmq_f32(FloatVector, MinimumValueVector);
    FloatVector = vminnmq_f32(FloatVector, MaximumValueVector);
#else
    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);
#endif

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

#if defined(MLAS_NEON64_INTRINSICS)
    auto IntegerVector = vcvtnq_s32_f32(FloatVector);
    IntegerVector = vaddq_s32(IntegerVector, ZeroPointVector);
#else
    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    auto IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);
#endif

    return IntegerVector;
}

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    );

#if defined(MLAS_NEON64_INTRINSICS)

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    )
{
    //
    // Swizzle the least significant byte from each int32_t element to the
    // bottom four bytes of the vector register.
    //

    uint16x8_t WordVector = vreinterpretq_u16_s32(IntegerVector);
    WordVector = vuzp1q_u16(WordVector, WordVector);
    uint8x16_t ByteVector = vreinterpretq_u8_u16(WordVector);
    ByteVector = vuzp1q_u8(ByteVector, ByteVector);

    return vreinterpretq_s32_u8(ByteVector);
}

#else

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

#endif

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = MlasBroadcastFloat32x4(Scale);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);

    while (N >= 4) {

        auto FloatVector = MlasLoadFloat32x4(Input);
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

        IntegerVector = MlasQuantizeLinearPackBytes<OutputType>(IntegerVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)Output, IntegerVector, 0);
#else
        *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);
#endif

        Input += 4;
        Output += 4;
        N -= 4;
    }

    for (size_t n = 0; n < N; n++) {

#if defined(MLAS_NEON64_INTRINSICS)
        auto FloatVector = vld1q_dup_f32(Input + n);
#else
        auto FloatVector = _mm_load_ss(Input + n);
#endif
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_u8((uint8_t*)Output + n, vreinterpretq_u8_s32(IntegerVector), 0);
#else
        *((uint8_t*)Output + n) = (uint8_t)_mm_cvtsi128_si32(IntegerVector);
#endif
    }
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasDequantizeLinearVector(
    MLAS_INT32X4 IntVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    return MlasMultiplyFloat32x4(vcvtq_f32_s32(MlasSubtractInt32x4(IntVector, ZeroPointVector)), ScaleVector);
#else
    return MlasMultiplyFloat32x4(_mm_cvtepi32_ps(MlasSubtractInt32x4(IntVector, ZeroPointVector)), ScaleVector);
#endif
}

template<typename DataType>
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes(
    MLAS_INT32X4 IntegerVector
    );

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    uint16x8_t vl = vmovl_u8(vget_low_u8(vreinterpretq_u8_s32(IntegerVector)));
    IntegerVector = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vl)));
#else
    IntegerVector = _mm_unpacklo_epi8(IntegerVector, IntegerVector);
    IntegerVector = _mm_unpacklo_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_srli_epi32(IntegerVector, 24);
#endif
    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    int16x8_t vl = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(IntegerVector)));
    IntegerVector = vmovl_s16(vget_low_s16(vl));
#else
    IntegerVector = _mm_unpacklo_epi8(IntegerVector, IntegerVector);
    IntegerVector = _mm_unpacklo_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_srai_epi32(IntegerVector, 24);
#endif
    return IntegerVector;
}

template<typename DataType, bool IsScalarA, bool IsScalarB>
void
MlasQLinearAddKernelHelper(
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
    constexpr int32_t MinimumValue = std::numeric_limits<DataType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<DataType>::max();

    const auto ScaleVectorA = MlasBroadcastFloat32x4(ScaleA);
    const auto ScaleVectorB = MlasBroadcastFloat32x4(ScaleB);
    const auto ScaleVectorC = MlasBroadcastFloat32x4(ScaleC);
    const auto ZeroPointVectorA = MlasBroadcastInt32x4(ZeroPointA);
    const auto ZeroPointVectorB = MlasBroadcastInt32x4(ZeroPointB);
    const auto ZeroPointVectorC = MlasBroadcastInt32x4(ZeroPointC);
    const auto MinimumValueVectorC = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPointC));
    const auto MaximumValueVectorC = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPointC));

    MLAS_FLOAT32X4 FloatVectorA;
    MLAS_FLOAT32X4 FloatVectorB;

    if (IsScalarA) {
        auto IntegerVectorA = MlasBroadcastInt32x4((int32_t)*InputA);
        FloatVectorA = MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA);
    }
    if (IsScalarB) {
        auto IntegerVectorB = MlasBroadcastInt32x4((int32_t)*InputB);
        FloatVectorB = MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB);
    }

    while (N >= 4) {
        if (!IsScalarA) {
            auto IntegerVectorA = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputA)));
            FloatVectorA = MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA);
        }
        if (!IsScalarB) {
            auto IntegerVectorB = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputB)));
            FloatVectorB = MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB);
        }
        auto FloatVectorC = MlasAddFloat32x4(FloatVectorA, FloatVectorB);
        auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
        IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)OutputC, IntegerVectorC, 0);
#else
        *((int32_t*)OutputC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

        if (!IsScalarA) {
            InputA += 4;
        }
        if (!IsScalarB) {
            InputB += 4;
        }
        OutputC += 4;
        N -= 4;
    }

    if (N > 0) {
        auto IntegerVectorA = (IsScalarA) ?
                MlasBroadcastInt32x4((int32_t)*InputA) :
                MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputA)));
        auto IntegerVectorB = (IsScalarB) ?
                MlasBroadcastInt32x4((int32_t)*InputB) :
                MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputB)));
        auto FloatVectorC = MlasAddFloat32x4(
                MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA),
                MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB));
        auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
        IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

        uint32_t PackedValueC = 0;
#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)&PackedValueC, IntegerVectorC, 0);
#else
        *((int32_t*)&PackedValueC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

        for (size_t n = 0; n < N; ++n) {
            *((uint8_t*)OutputC + n) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#else

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

#if defined(MLAS_NEON32_INTRINSICS)

template <typename DataType>
class SignedUnsignedIntOps;

template <>
class SignedUnsignedIntOps<uint8_t>
{
public:
    typedef uint8_t T;
    typedef uint8x8_t i8x8_t;
    typedef uint8x16_t i8x16_t;
    typedef uint16x8_t i16x8_t;

    static inline i8x8_t vmov_n_i8(T value) { return vmov_n_u8(value); }
    static inline i8x8_t vld1_dup_i8(T const* ptr) { return vld1_dup_u8(ptr); }
    static inline i8x8_t vld1_i8(T const * ptr) { return vld1_u8(ptr); }
    static inline i8x16_t vld1q_i8(T const * ptr) { return vld1q_u8(ptr); }
    static inline void vst1q_i8(T* ptr, i8x16_t a) { vst1q_u8(ptr, a); }

    static inline i8x8_t vget_low_i8(i8x16_t a) { return vget_low_u8(a); }
    static inline i8x8_t vget_high_i8(i8x16_t a) { return vget_high_u8(a); }

    static inline i16x8_t vsubl_i8(i8x8_t a, i8x8_t b) { return vsubl_u8(a, b); }
    static inline int16x8_t vreinterpretq_s16(i16x8_t a) { return vreinterpretq_s16_u16(a); }
    static inline i8x8_t vqmovn_s16(int16x8_t a) { return vqmovun_s16(a); }
    static inline i8x16_t vcombine_i8(i8x8_t a, i8x8_t b) { return vcombine_u8(a, b); }
    static inline uint64x2_t vreinterpretq_u64_i8(i8x16_t a) { return vreinterpretq_u64_u8(a); }
    static inline uint32x4_t vreinterpretq_u32_i8(i8x16_t a) { return vreinterpretq_u32_u8(a); }
    static inline uint16x8_t vreinterpretq_u16_i8(i8x16_t a) { return vreinterpretq_u16_u8(a); }

    template <int n>
    static inline void vst1q_lane_i8(T* ptr, i8x16_t a) { vst1q_lane_u8(ptr, a, n); }

    template <int n>
    static inline i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi) { return vextq_u8(lo, hi, n); }
};

template <>
class SignedUnsignedIntOps<int8_t>
{
public:
    typedef int8_t T;
    typedef int8x8_t i8x8_t;
    typedef int8x16_t i8x16_t;
    typedef int16x8_t i16x8_t;

    static inline i8x8_t vmov_n_i8(T value) { return vmov_n_s8(value); }
    static inline i8x8_t vld1_dup_i8(T const* ptr) { return vld1_dup_s8(ptr); }
    static inline i8x8_t vld1_i8(T const * ptr) { return vld1_s8(ptr); }
    static inline i8x16_t vld1q_i8(T const * ptr) { return vld1q_s8(ptr); }

    static inline void vst1q_i8(T* ptr, i8x16_t a) { vst1q_s8(ptr, a); }
    static inline i8x8_t vget_low_i8(i8x16_t a) { return vget_low_s8(a); }
    static inline i8x8_t vget_high_i8(i8x16_t a) { return vget_high_s8(a); }

    static inline i16x8_t vsubl_i8(i8x8_t a, i8x8_t b) { return vsubl_s8(a, b); }
    static inline int16x8_t vreinterpretq_s16(i16x8_t a) { return a; }

    static inline i8x8_t vqmovn_s16(int16x8_t a) { return vqmovn_s16(a); }
    static inline i8x16_t vcombine_i8(i8x8_t a, i8x8_t b) { return vcombine_s8(a, b); }
    static inline uint64x2_t vreinterpretq_u64_i8(i8x16_t a) { return vreinterpretq_u64_s8(a); }
    static inline uint32x4_t vreinterpretq_u32_i8(i8x16_t a) { return vreinterpretq_u32_s8(a); }
    static inline uint16x8_t vreinterpretq_u16_i8(i8x16_t a) { return vreinterpretq_u16_s8(a); }

    template <int n>
    static inline void vst1q_lane_i8(T* ptr, i8x16_t a) { vst1q_lane_s8(ptr, a, n); }

    template <int n>
    static inline i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi) { return vextq_s8(lo, hi, n); }
};

#include <cassert>

// DataType : int8_t or uint8_t,
template<typename DataType, bool IsScalarA, bool IsScalarB>
void
MlasQLinearAddKernelHelper(
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
    typedef SignedUnsignedIntOps<DataType> SUI;

    float ScaleRatio_AC = ScaleA / ScaleC;
    float ScaleRatio_BC = ScaleB / ScaleC;

    constexpr float MinScaleRatio = 6.103515625e-05f; // std::stof("0x1.0p-14f");
    constexpr float MaxScaleRatio = 256.0f; //std::stof("0x1.0p+8f");
    (void)MinScaleRatio; // Unused in Release build
    (void)MaxScaleRatio; // Unused in Release build
    assert(ScaleRatio_AC >= MinScaleRatio && ScaleRatio_AC < MaxScaleRatio);
    assert(ScaleRatio_BC >= MinScaleRatio && ScaleRatio_BC < MaxScaleRatio);

    const float GreaterScaleRatio = std::max(ScaleRatio_AC, ScaleRatio_BC);
    const int32_t GreaterExponent = (int32_t)(BitsOfFp32(GreaterScaleRatio) >> 23) - 127;
    const uint32_t Shift = (uint32_t)(21 - GreaterExponent); // Shift is in [13, 31] range.
    assert(Shift <= 31 && Shift >= 13);

    const float MultiplierForMantissa = Fp32FromBits((uint32_t)(21 - GreaterExponent + 127) << 23);
    const int32_t MultiplierA = (int32_t) lrintf(ScaleRatio_AC * MultiplierForMantissa);
    const int32_t MultiplierB = (int32_t) lrintf(ScaleRatio_BC * MultiplierForMantissa);
    assert(MultiplierA < 0x00400000 && MultiplierB < 0x00400000);
    assert(MultiplierA >= 0x00200000 || MultiplierB >= 0x00200000); // the bigger one must fullfil this check

    const int32x4_t VectorMultiplierA = vld1q_dup_s32(&MultiplierA);
    const int32x4_t VectorMultiplierB = vld1q_dup_s32(&MultiplierB);
    const typename SUI::i8x8_t VectorZeroPointA = SUI::vmov_n_i8((DataType)ZeroPointA);
    const typename SUI::i8x8_t VectorZeroPointB = SUI::vmov_n_i8((DataType)ZeroPointB);
    const int16x8_t VectorZeroPointC = vmovq_n_s16((int16_t)ZeroPointC);
    const int32x4_t vright_shift = vmovq_n_s32(- static_cast<int32_t>(Shift)); // vld1q_dup_s32(&right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));

     typename SUI::i8x16_t vc;
    auto n = static_cast<int64_t>(N);
    while (n > 0) {
        const typename SUI::i8x16_t VectorA = SUI::vld1q_i8(InputA); InputA += 16;
        const typename SUI::i8x16_t VectorB = SUI::vld1q_i8(InputB); InputB += 16;
        const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA), VectorZeroPointA));
        const int16x8_t vb0_s16x8 = SUI::vreinterpretq_s16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB), VectorZeroPointB));
        const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA), VectorZeroPointA));
        const int16x8_t vb1_s16x8 = SUI::vreinterpretq_s16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB), VectorZeroPointB));

        int32x4_t vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
        int32x4_t vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
        int32x4_t vacc0_hi = vmulq_s32(vmovl_s16(vget_high_s16(va0_s16x8)), VectorMultiplierA);
        int32x4_t vacc1_hi = vmulq_s32(vmovl_s16(vget_high_s16(va1_s16x8)), VectorMultiplierA);
        vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vb0_s16x8)), VectorMultiplierB);
        vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vb1_s16x8)), VectorMultiplierB);
        vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_s16(vget_high_s16(vb0_s16x8)), VectorMultiplierB);
        vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_s16(vget_high_s16(vb1_s16x8)), VectorMultiplierB);
        
        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), VectorZeroPointC);

        vc = SUI::vcombine_i8(SUI::vqmovn_s16(vacc0), SUI::vqmovn_s16(vacc1));
        n -= 16;
        if (n < 0) break;

        SUI::vst1q_i8(OutputC, vc);
        OutputC += 16;
    }

    if (n < 0) {
        n += 16;
        if (n & 8) {
            vst1_u64((uint64_t*)OutputC, vget_low_u64(SUI::vreinterpretq_u64_i8(vc)));
            OutputC += 8;
            vc = SUI::template vextq_i8<8>(vc, vc);
        }
        if (n & 4) {
            vst1q_lane_u32((uint32_t*)OutputC, SUI::vreinterpretq_u32_i8(vc), 0);
            OutputC += 4;
            vc = SUI::template vextq_i8<4>(vc, vc);
        }
        if (n & 2) {
            vst1q_lane_u16((uint16_t*)OutputC, SUI::vreinterpretq_u16_i8(vc), 0);
            OutputC += 2;
            vc = SUI::template vextq_i8<2>(vc, vc);
        }
        if (n & 1) {
            SUI::template vst1q_lane_i8<0>(OutputC, vc);
        }
    }
}

#else

template<typename DataType, bool IsScalarA, bool IsScalarB>
void
MlasQLinearAddKernelHelper(
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
    constexpr int32_t MinimumValue = std::numeric_limits<DataType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<DataType>::max();

    float ValueA;
    float ValueB;

    if (IsScalarA) {
        ValueA = ScaleA * (int32_t(InputA[0]) - ZeroPointA);
    }
    if (IsScalarB) {
        ValueB = ScaleB * (int32_t(InputB[0]) - ZeroPointB);
    }

    for (size_t n = 0; n < N; n++) {
        if (!IsScalarA) {
            ValueA = ScaleA * (int32_t(InputA[n]) - ZeroPointA);
        }
        if (!IsScalarB) {
            ValueB = ScaleB * (int32_t(InputB[n]) - ZeroPointB);
        }
        int32_t IntValueC = (int32_t)std::nearbyintf((ValueA + ValueB) / ScaleC) + ZeroPointC;
        IntValueC = std::max(IntValueC, MinimumValue);
        IntValueC = std::min(IntValueC, MaximumValue);
        OutputC[n] = (DataType)IntValueC;
    }
}

#endif

#endif

template<typename DataType>
void
MLASCALL
MlasQLinearAddKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    size_t N = std::max(LengthA, LengthB);
    if (N > 0) {
        if (LengthA == 1) {
            MlasQLinearAddKernelHelper<DataType, true, false>(
                InputA, ScaleA, ZeroPointA,
                InputB, ScaleB, ZeroPointB,
                ScaleC, ZeroPointC, OutputC, N);
        } else if (LengthB == 1) {
            MlasQLinearAddKernelHelper<DataType, false, true>(
                InputA, ScaleA, ZeroPointA,
                InputB, ScaleB, ZeroPointB,
                ScaleC, ZeroPointC, OutputC, N);
        } else {
            MlasQLinearAddKernelHelper<DataType, false, false>(
                InputA, ScaleA, ZeroPointA,
                InputB, ScaleB, ZeroPointB,
                ScaleC, ZeroPointC, OutputC, N);
        }
    }
}

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

#if defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
MLAS_INT32X4
MlasRequantizeOutputVector(
    MLAS_INT32X4 IntegerVector,
    MLAS_INT32X4 BiasVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    IntegerVector = _mm_add_epi32(IntegerVector, BiasVector);
    MLAS_FLOAT32X4 FloatVector = _mm_cvtepi32_ps(IntegerVector);

    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);

    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);

    return IntegerVector;
}

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        if (Bias != nullptr) {
            BiasVector = MlasBroadcastInt32x4(*Bias++);
        }

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

#endif

void
MLASCALL
MlasFindMinMaxElement(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
/*++

Routine Description:

    This routine finds the minimum and maximum values of the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    Min - Returns the minimum value of the supplied buffer.

    Max - Returns the maximum value of the supplied buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.ReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#else
    MlasReduceMinimumMaximumF32Kernel(Input, Min, Max, N);
#endif
}

void
MLASCALL
MlasQLinearAddS8Kernel(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    MlasQLinearAddKernel<int8_t>(
        InputA, ScaleA, ZeroPointA,
        InputB, ScaleB, ZeroPointB,
        ScaleC, ZeroPointC, OutputC,
        LengthA, LengthB
    );
}

void
MLASCALL
MlasQLinearAddU8Kernel(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    MlasQLinearAddKernel<uint8_t>(
        InputA, ScaleA, ZeroPointA,
        InputB, ScaleB, ZeroPointB,
        ScaleC, ZeroPointC, OutputC,
        LengthA, LengthB
    );
}

template<>
void
MLASCALL
MlasQLinearAdd<int8_t>(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
#if defined(MLAS_TARGET_AMD64)
        MlasPlatform.QLinearAddS8Kernel(
#else
        MlasQLinearAddKernel<int8_t>(
#endif
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, LengthA, LengthB);
}

template<>
void
MLASCALL
MlasQLinearAdd<uint8_t>(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
#if defined(MLAS_TARGET_AMD64)
        MlasPlatform.QLinearAddU8Kernel(
#else
        MlasQLinearAddKernel<uint8_t>(
#endif
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, LengthA, LengthB);
}

#if defined(MLAS_TARGET_AMD64)
MLAS_INTERNAL_DATA  MLAS_DECLSPEC_ALIGN(const uint8_t MlasPackBytesMM256VpshufbControl[32], 32) = {
    0,4,8,12,        255,255,255,255, 255,255,255,255, 255,255,255,255,
    255,255,255,255, 0,4,8,12,        255,255,255,255, 255,255,255,255
};

MLAS_INTERNAL_DATA  MLAS_DECLSPEC_ALIGN(const int32_t MlasPackBytesMM256VpermpsControl[8], 32) = {
    0, 5, 2, 3, 4, 1, 6, 7
};
#endif
