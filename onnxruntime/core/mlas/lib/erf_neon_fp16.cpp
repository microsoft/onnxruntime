/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   erf_neon_fp16.cpp

Abstract:

    This module contains the procedure prototypes for the ERF NEON FP16 intrinsics.

--*/

#include "erf_neon_fp16.h"

// Helpers to safely convert between float and FP16-bit representation
static float
fp16_to_float(uint16_t h)
{
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(h));
    return (float)tmp;
}

static uint16_t
float_to_fp16(float f)
{
    __fp16 tmp = (__fp16)f;
    uint16_t h;
    memcpy(&h, &tmp, sizeof(h));
    return h;
}

static inline MLAS_FLOAT16X8
exp_neg_rational_approx_f16(MLAS_FLOAT16X8 x)
{
    const float16_t a0 = 6.0f;
    MLAS_FLOAT16X8 max_x = MlasBroadcastF16Float16x8(a0);
    x = MlasMinimumFloat16(x, max_x);

    const float16_t c0 = 1.330f;
    const float16_t c1 = -0.390f;
    const float16_t c2 = 0.0288f;

    const float16_t d0 = 1.338f;
    const float16_t d1 = 0.848f;
    const float16_t d2 = 0.467f;

    MLAS_FLOAT16X8 c0v = MlasBroadcastF16Float16x8(c0);
    MLAS_FLOAT16X8 c1v = MlasBroadcastF16Float16x8(c1);
    MLAS_FLOAT16X8 c2v = MlasBroadcastF16Float16x8(c2);

    MLAS_FLOAT16X8 d0v = MlasBroadcastF16Float16x8(d0);
    MLAS_FLOAT16X8 d1v = MlasBroadcastF16Float16x8(d1);
    MLAS_FLOAT16X8 d2v = MlasBroadcastF16Float16x8(d2);
    MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(x, x);
    MLAS_FLOAT16X8 num = MlasMultiplyAddFloat16(c1v, x, c0v);
    num = MlasMultiplyAddFloat16(c2v, x2, num);
    MLAS_FLOAT16X8 den = MlasMultiplyAddFloat16(d1v, x, d0v);
    den = MlasMultiplyAddFloat16(d2v, x2, den);
    MLAS_FLOAT16X8 recip = MlasApproximateReciprocalFloat16(den);
    recip = MlasMultiplyFloat16(recip, MlasReciprocalSqrtFloat16(den, recip));
    recip = MlasMultiplyFloat16(recip, MlasReciprocalSqrtFloat16(den, recip));
    MLAS_FLOAT16X8 result = MlasMultiplyFloat16(num, recip);
    return result;
}

void
MlasNeonErfF16Kernel(const _mlas_fp16_* Input, _mlas_fp16_* Output, size_t N)
{
    const float16_t p = 0.328f;
    const float16_t a1 = 0.2505f;
    const float16_t a2 = -0.2881f;
    const float16_t a3 = 1.4102f;
    const float16_t a4 = -1.423f;
    const float16_t a5 = 1.0547f;

    MLAS_FLOAT16X8 vp = MlasBroadcastF16Float16x8(p);
    MLAS_FLOAT16X8 va1 = MlasBroadcastF16Float16x8(a1);
    MLAS_FLOAT16X8 va2 = MlasBroadcastF16Float16x8(a2);
    MLAS_FLOAT16X8 va3 = MlasBroadcastF16Float16x8(a3);
    MLAS_FLOAT16X8 va4 = MlasBroadcastF16Float16x8(a4);
    MLAS_FLOAT16X8 va5 = MlasBroadcastF16Float16x8(a5);

    constexpr float16_t one_fp16 = 1.0f;
    constexpr float16_t neg_one_fp16 = -1.0f;
    constexpr float16_t zero_fp16 = 0.0f;
    constexpr float16_t four_fp16 = 4.0f;

    MLAS_FLOAT16X8 vone = MlasBroadcastF16Float16x8(one_fp16);
    MLAS_FLOAT16X8 vneg_one = MlasBroadcastF16Float16x8(neg_one_fp16);
    MLAS_FLOAT16X8 vzero = MlasBroadcastF16Float16x8(zero_fp16);
    MLAS_FLOAT16X8 vth = MlasBroadcastF16Float16x8(four_fp16);

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        MLAS_FLOAT16X8 x = MlasLoadFloat16x8(&Input[i]);
        MLAS_UINT16X8 neg_mask = MlasCompareLessThanFloat16(x, vzero);
        MLAS_FLOAT16X8 sign = MlasSelectFloat16(neg_mask, vneg_one, vone);
        MLAS_FLOAT16X8 absx = MlasAbsFloat16(x);
        MLAS_UINT16X8 use_mask = MlasCompareLessThanFloat16(absx, vth);
        MLAS_FLOAT16X8 absx_clamped = MlasMinimumFloat16(absx, vth);
        MLAS_FLOAT16X8 denom = MlasMultiplyAddFloat16(vp, absx_clamped, vone);
        MLAS_FLOAT16X8 t = MlasApproximateReciprocalFloat16(denom);
        t = MlasMultiplyFloat16(t, MlasReciprocalSqrtFloat16(denom, t));
        t = MlasMultiplyFloat16(t, MlasReciprocalSqrtFloat16(denom, t));
        MLAS_FLOAT16X8 t2 = MlasMultiplyFloat16(t, t);
        MLAS_FLOAT16X8 t3 = MlasMultiplyFloat16(t2, t);
        MLAS_FLOAT16X8 t4 = MlasMultiplyFloat16(t3, t);
        MLAS_FLOAT16X8 t5 = MlasMultiplyFloat16(t4, t);
        MLAS_FLOAT16X8 poly = MlasMultiplyFloat16(va1, t);
        poly = MlasMultiplyAddFloat16(va2, t2, poly);
        poly = MlasMultiplyAddFloat16(va3, t3, poly);
        poly = MlasMultiplyAddFloat16(va4, t4, poly);
        poly = MlasMultiplyAddFloat16(va5, t5, poly);
        MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(absx_clamped, absx_clamped);
        MLAS_FLOAT16X8 exp_neg_x2 = exp_neg_rational_approx_f16(x2);
        MLAS_FLOAT16X8 poly_mul_exp = MlasMultiplyFloat16(poly, exp_neg_x2);
        MLAS_FLOAT16X8 one_minus_term = MlasSubtractFloat16(vone, poly_mul_exp);
        MLAS_FLOAT16X8 erf_approx = MlasMultiplyFloat16(sign, one_minus_term);
        erf_approx = MlasMinimumFloat16(erf_approx, vone);
        erf_approx = MlasMaximumFloat16(erf_approx, vneg_one);
        MLAS_FLOAT16X8 result = MlasSelectFloat16(use_mask, erf_approx, sign);
        MlasStoreFloat16x8(&Output[i], result);
    }

    for (; i < N; i++) {
        float x = fp16_to_float(Input[i]);
        float sign = (x < 0) ? -1.0f : 1.0f;
        float absx = fabsf(x);

        if (absx > 4.0f) {
            Output[i] = float_to_fp16(sign);
            continue;
        }

        float t = 1.0f / (1.0f + p * absx);
        float poly = a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
        float exp_neg_x2 = expf(-absx * absx);
        float erf_approx = sign * (1.0f - poly * exp_neg_x2);
        if (erf_approx > 1.0f) erf_approx = 1.0f;
        if (erf_approx < -1.0f) erf_approx = -1.0f;

        Output[i] = float_to_fp16(erf_approx);
    }
}
