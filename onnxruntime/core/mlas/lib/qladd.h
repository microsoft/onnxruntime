/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    qladd.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for QLinearAdd function usage .

--*/

#pragma once

#include "mlasi.h"

MLAS_FORCEINLINE
static
void
MlasCopyTailBytes(
    uint8_t* target,
    const uint8_t* src,
    size_t N)
{
    while (N >= sizeof(uint32_t)) {
        *(uint32_t*)(target) = *(uint32_t*)(src);
        N -= sizeof(uint32_t);
        target += sizeof(uint32_t);
        src += sizeof(uint32_t);
    }
    while (N > 0) {
        *target++ = *src++;
        --N;
    }
}

bool
MlasCalcQLinearAddParameters(
    float ScaleRatio_AC,
    float ScaleRatio_BC,
    int32_t& Shift,
    int32_t& MultiplierA,
    int32_t& MultiplierB
    );

#if defined(MLAS_NEON_INTRINSICS)

#if ! defined(_MSC_VER)

#define vld1q_s8_ex(pD, align) vld1q_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)))
#define vst1_s8_ex(pD, D, align) vst1_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1q_s8_ex(pD, D, align) vst1q_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vld1q_u8_ex(pD, align) vld1q_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)))
#define vst1_u8_ex(pD, D, align) vst1_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1q_u8_ex(pD, D, align) vst1q_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1_lane_u32_ex(pD, D, lane, align) vst1_lane_u32((uint32_t*)__builtin_assume_aligned(pD, ((align)/8)), D, lane)
#define vst1_lane_u16_ex(pD, D, lane, align) vst1_lane_u16((uint16_t*)__builtin_assume_aligned(pD, ((align)/8)), D, lane)

#endif

template <typename DataType>
class MLAS_SignedUnsignedIntOps;

template <>
class MLAS_SignedUnsignedIntOps<uint8_t>
{
public:
    typedef uint8_t T;
    typedef uint8x8_t i8x8_t;
    typedef uint8x16_t i8x16_t;
    typedef uint16x8_t i16x8_t;

    static MLAS_FORCEINLINE i8x8_t vmov_n_i8(T value)
    {
        return vmov_n_u8(value);
    }

    static MLAS_FORCEINLINE i8x8_t vget_low_i8(i8x16_t a)
    {
        return vget_low_u8(a);
    }

    static MLAS_FORCEINLINE i8x8_t vget_high_i8(i8x16_t a)
    {
        return vget_high_u8(a);
    }

    static MLAS_FORCEINLINE i16x8_t vsubl_i8(i8x8_t a, i8x8_t b)
    {
        return vsubl_u8(a, b);
    }

    static MLAS_FORCEINLINE int16x8_t vreinterpretq_s16_i16(i16x8_t a)
    {
        return vreinterpretq_s16_u16(a);
    }

    static MLAS_FORCEINLINE uint32x4_t vreinterpretq_u32_i8(i8x16_t a)
    {
        return vreinterpretq_u32_u8(a);
    }

    static MLAS_FORCEINLINE uint16x8_t vreinterpretq_u16_i8(i8x16_t a)
    {
        return vreinterpretq_u16_u8(a);
    }

    static MLAS_FORCEINLINE uint32x2_t vreinterpret_u32_i8(i8x8_t a)
    {
        return vreinterpret_u32_u8(a);
    }

    static MLAS_FORCEINLINE uint16x4_t vreinterpret_u16_i8(i8x8_t a)
    {
        return vreinterpret_u16_u8(a);
    }

    static MLAS_FORCEINLINE i8x16_t vld1q_i8(T const * ptr)
    {
        return vld1q_u8_ex(ptr, 8);
    }

    static MLAS_FORCEINLINE void vst1_i8(T* ptr, i8x8_t a)
    {
        vst1_u8_ex(ptr, a, 8);
    }

    static MLAS_FORCEINLINE void vst1q_i8(T* ptr, i8x16_t a)
    {
        vst1q_u8_ex(ptr, a, 8);
    }

    template <int n>
    static MLAS_FORCEINLINE void vst1_lane_i8(T* ptr, i8x8_t a)
    {
        vst1_lane_u8(ptr, a, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi)
    {
        return vextq_u8(lo, hi, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x8_t vext_i8(i8x8_t lo, i8x8_t hi)
    {
        return vext_u8(lo, hi, n);
    }

    static MLAS_FORCEINLINE i8x16_t combine_i8_s16(int16x8_t v0, int16x8_t v1)
    {

#if defined(MLAS_NEON64_INTRINSICS)
         return vqmovun_high_s16(vqmovun_s16(v0), v1);
#else
         return vcombine_u8(vqmovun_s16(v0), vqmovun_s16(v1));
#endif

    }
};

template <>
class MLAS_SignedUnsignedIntOps<int8_t>
{
public:
    typedef int8_t T;
    typedef int8x8_t i8x8_t;
    typedef int8x16_t i8x16_t;
    typedef int16x8_t i16x8_t;

    static MLAS_FORCEINLINE i8x8_t vmov_n_i8(T value)
    {
        return vmov_n_s8(value);
    }

    static MLAS_FORCEINLINE i8x8_t vget_low_i8(i8x16_t a)
    {
        return vget_low_s8(a);
    }

    static MLAS_FORCEINLINE i8x8_t vget_high_i8(i8x16_t a)
    {
        return vget_high_s8(a);
    }

    static MLAS_FORCEINLINE i16x8_t vsubl_i8(i8x8_t a, i8x8_t b)
    {
        return vsubl_s8(a, b);
    }

    static MLAS_FORCEINLINE int16x8_t vreinterpretq_s16_i16(i16x8_t a)
    {
        return a;
    }

    static MLAS_FORCEINLINE uint32x4_t vreinterpretq_u32_i8(i8x16_t a)
    {
        return vreinterpretq_u32_s8(a);
    }

    static MLAS_FORCEINLINE uint16x8_t vreinterpretq_u16_i8(i8x16_t a)
    {
        return vreinterpretq_u16_s8(a);
    }

    static MLAS_FORCEINLINE uint32x2_t vreinterpret_u32_i8(i8x8_t a)
    {
        return vreinterpret_u32_s8(a);
    }

    static MLAS_FORCEINLINE uint16x4_t vreinterpret_u16_i8(i8x8_t a)
    {
        return vreinterpret_u16_s8(a);
    }

    static MLAS_FORCEINLINE i8x16_t vld1q_i8(T const * ptr)
    {
        return vld1q_s8_ex(ptr, 8);
    }

    static MLAS_FORCEINLINE void vst1_i8(T* ptr, i8x8_t a)
    {
        vst1_s8_ex(ptr, a, 8);
    }

    static MLAS_FORCEINLINE void vst1q_i8(T* ptr, i8x16_t a)
    {
        vst1q_s8_ex(ptr, a, 8);
    }

    template <int n>
    static MLAS_FORCEINLINE void vst1_lane_i8(T* ptr, i8x8_t a)
    {
        vst1_lane_s8(ptr, a, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi)
    {
        return vextq_s8(lo, hi, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x8_t vext_i8(i8x8_t lo, i8x8_t hi)
    {
        return vext_s8(lo, hi, n);
    }

    static MLAS_FORCEINLINE i8x16_t combine_i8_s16(int16x8_t v0, int16x8_t v1)
    {

#if defined(MLAS_NEON64_INTRINSICS)
         return vqmovn_high_s16(vqmovn_s16(v0), v1);
#else
         return vcombine_s8(vqmovn_s16(v0), vqmovn_s16(v1));
#endif

    }
};

#if defined(MLAS_NEON64_INTRINSICS)

#define MlasMoveHighS16S32(s16x8) vmovl_high_s16(s16x8)
#define MlasCombineS16S32(lo, hi) vqmovn_high_s32(vqmovn_s32(lo), hi)

#else

#define MlasMoveHighS16S32(s16x8) vmovl_s16(vget_high_s16(s16x8))
#define MlasCombineS16S32(lo, hi) vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))

#endif

#elif defined(MLAS_SSE2_INTRINSICS)

template <typename DataType>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt32(
    MLAS_INT32X4 v,
    int imm
    );

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt32<int8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srai_epi32(v, imm);
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt32<uint8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srli_epi32(v, imm);
}

template <typename DataType>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt16(
    MLAS_INT32X4 v,
    int imm
    );

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt16<int8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srai_epi16(v, imm);
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasShiftRightInt16<uint8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srli_epi16(v, imm);
}

template <typename DataType>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasPackS16_128(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    );

template <>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasPackS16_128<uint8_t>(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    )
{
    return _mm_packus_epi16(a, b);
}

template <>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasPackS16_128<int8_t>(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    )
{
    return _mm_packs_epi16(a, b);
}

#endif
