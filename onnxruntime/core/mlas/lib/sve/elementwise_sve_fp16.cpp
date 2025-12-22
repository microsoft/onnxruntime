/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   Elementwise_sve_fp16.cpp

Abstract:

    This module contains the  SVE Elementwise functions .

--*/
#include "mlas_sve_fp16.h"

struct MlasTanhConstants_fp16_scalar {
    __fp16 LowerRange;
    __fp16 UpperRange;
    __fp16 alpha_7;
    __fp16 alpha_5;
    __fp16 alpha_3;
    __fp16 alpha_1;
    __fp16 beta_6;
    __fp16 beta_4;
    __fp16 beta_2;
    __fp16 beta_0;
};

constexpr MlasTanhConstants_fp16_scalar TanhConstantsFp16 = {
    -3.515625f,
    3.515625f,
    5.960464477539063e-08f,
    1.4841556549072266e-05f,
    0.000637054443359375f,
    0.004894256591796875f,
    1.1920928955078125e-06f,
    0.00011855363845825195f,
    0.0022678375244140625f,
    0.004894256591796875f
};

static inline MLAS_SVFLOAT16
Tanh_Vector_SVE_fp16(MLAS_SVFLOAT16 x, MLAS_SVBOOL pg)
{
    MLAS_SVFLOAT16 g_LowerRange_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.LowerRange);
    MLAS_SVFLOAT16 g_UpperRange_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.UpperRange);
    MLAS_SVFLOAT16 g_alpha_7_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.alpha_7);
    MLAS_SVFLOAT16 g_alpha_5_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.alpha_5);
    MLAS_SVFLOAT16 g_alpha_3_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.alpha_3);
    MLAS_SVFLOAT16 g_alpha_1_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.alpha_1);
    MLAS_SVFLOAT16 g_beta_6_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.beta_6);
    MLAS_SVFLOAT16 g_beta_4_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.beta_4);
    MLAS_SVFLOAT16 g_beta_2_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.beta_2);
    MLAS_SVFLOAT16 g_beta_0_vec = MlasSveBroadcastfloat16(TanhConstantsFp16.beta_0);

    x = MlasSveMinfloat16(pg, x, g_UpperRange_vec);
    x = MlasSveMaxfloat16(pg, x, g_LowerRange_vec);

    MLAS_SVFLOAT16 x2 = MlasSveMulfloat16(pg, x, x);
    MLAS_SVFLOAT16 p = MlasSveMLAfloat16(pg, g_alpha_5_vec, g_alpha_7_vec, x2);
    p = MlasSveMLAfloat16(pg, g_alpha_3_vec, p, x2);
    p = MlasSveMLAfloat16(pg, g_alpha_1_vec, p, x2);
    p = MlasSveMulfloat16(pg, p, x);

    svfloat16_t q = MlasSveMLAfloat16(pg, g_beta_4_vec, g_beta_6_vec, x2);
    q = MlasSveMLAfloat16(pg, g_beta_2_vec, q, x2);
    q = MlasSveMLAfloat16(pg, g_beta_0_vec, q, x2);

    MLAS_SVFLOAT16 res = MlasSveDivfloat16(pg, p, q);

    return res;
}

void
MlasSveTanhF16Kernel(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N)
{
    size_t offset = 0;
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    while (offset < N) {
        MLAS_SVBOOL pg = MlasSveSelPredictefloat16(offset, N);
        MLAS_SVFLOAT16 x = MlasSvereinterpretf16_u16(MlasSveLoadUint16(pg, &input[offset]));
        MLAS_SVFLOAT16 y = Tanh_Vector_SVE_fp16(x, pg);
        MlasSveStoreUint16(pg, &output[offset], MlasSvereinterpretu16_f16(y));
        offset += svcnth();
    }
}

static inline MLAS_SVFLOAT16
exp_neg_rational_approx_f16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x)
{
    const __fp16 a0 = 6.0f;
    MLAS_SVFLOAT16 max_x = MlasSveBroadcastfloat16(a0);
    x = MlasSveMinfloat16(pg, x, max_x);

    const __fp16 c0 = 1.330f;
    const __fp16 c1 = -0.390f;
    const __fp16 c2 = 0.0288f;

    const __fp16 d0 = 1.338f;
    const __fp16 d1 = 0.848f;
    const __fp16 d2 = 0.467f;

    MLAS_SVFLOAT16 c0v = MlasSveBroadcastfloat16(c0);
    MLAS_SVFLOAT16 c1v = MlasSveBroadcastfloat16(c1);
    MLAS_SVFLOAT16 c2v = MlasSveBroadcastfloat16(c2);
    MLAS_SVFLOAT16 d0v = MlasSveBroadcastfloat16(d0);
    MLAS_SVFLOAT16 d1v = MlasSveBroadcastfloat16(d1);
    MLAS_SVFLOAT16 d2v = MlasSveBroadcastfloat16(d2);
    MLAS_SVFLOAT16 x2 = MlasSveMulfloat16(pg, x, x);

    MLAS_SVFLOAT16 num = MlasSveMLAfloat16(pg, c0v, c1v, x);
    num = MlasSveMLAfloat16(pg, num, c2v, x2);

    MLAS_SVFLOAT16 den = MlasSveMLAfloat16(pg, d0v, d1v, x);
    den = MlasSveMLAfloat16(pg, den, d2v, x2);

    MLAS_SVFLOAT16 recip = MlasSveReciprocalfloat16(den);
    recip = MlasSveMulfloat16(pg, recip, MlasSveReciprocalStepfloat16(den, recip));
    recip = MlasSveMulfloat16(pg, recip, MlasSveReciprocalStepfloat16(den, recip));

    MLAS_SVFLOAT16 result = MlasSveMulfloat16(pg, num, recip);
    return result;
}

void MLASCALL
MlasSveErfF16Kernel(const _mlas_fp16_* Input, _mlas_fp16_* Output, size_t N)
{
    const __fp16 p = 0.328f;
    const __fp16 a1 = 0.2505f;
    const __fp16 a2 = -0.2881f;
    const __fp16 a3 = 1.4102f;
    const __fp16 a4 = -1.423f;
    const __fp16 a5 = 1.0547f;

    MLAS_SVFLOAT16 vp = MlasSveBroadcastfloat16(p);
    MLAS_SVFLOAT16 va1 = MlasSveBroadcastfloat16(a1);
    MLAS_SVFLOAT16 va2 = MlasSveBroadcastfloat16(a2);
    MLAS_SVFLOAT16 va3 = MlasSveBroadcastfloat16(a3);
    MLAS_SVFLOAT16 va4 = MlasSveBroadcastfloat16(a4);
    MLAS_SVFLOAT16 va5 = MlasSveBroadcastfloat16(a5);

    const __fp16 v1 = 1.0f;
    const __fp16 v2 = -1.0f;
    const __fp16 v3 = 0.0f;
    const __fp16 v4 = 4.0f;
    MLAS_SVFLOAT16 vone = MlasSveBroadcastfloat16(v1);
    MLAS_SVFLOAT16 vneg_one = MlasSveBroadcastfloat16(v2);
    MLAS_SVFLOAT16 vzero = MlasSveBroadcastfloat16(v3);
    MLAS_SVFLOAT16 vth = MlasSveBroadcastfloat16(v4);

    size_t i = 0;
    while (i < N) {
        MLAS_SVBOOL pg = MlasSveSelPredictefloat16(i, N);
        MLAS_SVFLOAT16 x = MlasSvereinterpretf16_u16(MlasSveLoadUint16(pg, &Input[i]));
        MLAS_SVBOOL neg_mask = MlasSveComparelessthanfloat16(pg, x, vzero);
        MLAS_SVFLOAT16 sign = MlasSveSelectfloat16(neg_mask, vneg_one, vone);
        MLAS_SVFLOAT16 absx = MlasSveAbsolutefloat16(MlasSveBroadcastfloat16(v3), pg, x);
        svbool_t use_mask = MlasSveComparelessthanfloat16(pg, absx, vth);
        MLAS_SVFLOAT16 absx_clamped = MlasSveMinfloat16(pg, absx, vth);
        MLAS_SVFLOAT16 denom = MlasSveMLAfloat16(pg, vone, vp, absx_clamped);
        MLAS_SVFLOAT16 t = MlasSveReciprocalfloat16(denom);
        t = MlasSveMulfloat16(pg, t, MlasSveReciprocalStepfloat16(denom, t));
        t = MlasSveMulfloat16(pg, t, MlasSveReciprocalStepfloat16(denom, t));
        MLAS_SVFLOAT16 t2 = MlasSveMulfloat16(pg, t, t);
        MLAS_SVFLOAT16 t3 = MlasSveMulfloat16(pg, t2, t);
        MLAS_SVFLOAT16 t4 = MlasSveMulfloat16(pg, t3, t);
        MLAS_SVFLOAT16 t5 = MlasSveMulfloat16(pg, t4, t);
        svfloat16_t poly = MlasSveMulfloat16(pg, va1, t);
        poly = MlasSveMLAfloat16(pg, poly, va2, t2);
        poly = MlasSveMLAfloat16(pg, poly, va3, t3);
        poly = MlasSveMLAfloat16(pg, poly, va4, t4);
        poly = MlasSveMLAfloat16(pg, poly, va5, t5);
        MLAS_SVFLOAT16 x2 = MlasSveMulfloat16(pg, absx_clamped, absx_clamped);
        MLAS_SVFLOAT16 exp_neg_x2 = exp_neg_rational_approx_f16(pg, x2);
        MLAS_SVFLOAT16 poly_mul_exp = MlasSveMulfloat16(pg, poly, exp_neg_x2);
        MLAS_SVFLOAT16 one_minus_term = MlasSveSubtractfloat16(pg, vone, poly_mul_exp);
        MLAS_SVFLOAT16 erf_approx = MlasSveMulfloat16(pg, sign, one_minus_term);
        erf_approx = MlasSveMinfloat16(pg, erf_approx, vone);
        erf_approx = MlasSveMaxfloat16(pg, erf_approx, vneg_one);
        MLAS_SVFLOAT16 result = MlasSveSelectfloat16(use_mask, erf_approx, sign);
        MlasSveStoreUint16(pg, &Output[i], MlasSvereinterpretu16_f16(result));
        i += svcntp_b16(svptrue_b16(), pg);
    }
}

void MLASCALL
MlasSveGeluF16Kernel(const MLAS_FP16* input, MLAS_FP16* output, MLAS_FP16* temp, int64_t count, const std::string& algo)
{
    const __fp16 r1 = 0.5f;
    const __fp16 r2 = 1.0f;
    const __fp16 r3 = static_cast<float>(M_SQRT1_2);
    const __fp16 r4 = 0.7979f;
    const __fp16 r5 = 0.03568f;

    const MLAS_SVFLOAT16 v_half = MlasSveBroadcastfloat16(r1);
    const MLAS_SVFLOAT16 v_one = MlasSveBroadcastfloat16(r2);
    const MLAS_SVFLOAT16 v_sqrt1_2 = MlasSveBroadcastfloat16(r3);
    const MLAS_SVFLOAT16 v_B = MlasSveBroadcastfloat16(r4);
    const MLAS_SVFLOAT16 v_C = MlasSveBroadcastfloat16(r5);

    const __fp16 c1 = -5.0f;
    const __fp16 c2 = 5.0f;
    if (algo == "tanh") {
        int64_t i = 0;
        while (i < (count)) {
            svbool_t pg = MlasSveSelPredictefloat16(i, count);
            MLAS_SVFLOAT16 v_x = MlasSveLoadFloat16(pg, &input[i]);
            MLAS_SVFLOAT16 v_x2 = MlasSveMulfloat16(pg, v_x, v_x);
            MLAS_SVFLOAT16 v_inner = MlasSveMLAfloat16(pg, v_B, v_C, v_x2);
            MLAS_SVFLOAT16 v_tanh_arg = MlasSveMulfloat16(pg, v_x, v_inner);
            v_tanh_arg = MlasSveMaxfloat16(pg, MlasSveBroadcastfloat16(c1), MlasSveMinfloat16(pg, v_tanh_arg, MlasSveBroadcastfloat16(c2)));
            MlasSveStoreF16(pg, &temp[i], v_tanh_arg);
            i += svcnth();
        }

        MlasSveTanhF16Kernel(reinterpret_cast<const MLAS_FP16*>(temp), reinterpret_cast<MLAS_FP16*>(temp), count);

        int64_t j = 0;
        while (j < (count)) {
            svbool_t pg = MlasSveSelPredictefloat16(j, count);
            MLAS_SVFLOAT16 v_x = MlasSveLoadFloat16(pg, &input[j]);
            MLAS_SVFLOAT16 v_tanh = MlasSveLoadFloat16(pg, &temp[j]);
            MLAS_SVFLOAT16 v_result = MlasSveMulfloat16(pg, v_half, MlasSveMulfloat16(pg, v_x, svadd_f16_m(pg, v_one, v_tanh)));
            MlasSveStoreF16(pg, &output[j], v_result);
            j += svcnth();
        }
    } else if (algo == "none") {
        int64_t i = 0;
        while (i < (count)) {
            svbool_t pg = MlasSveSelPredictefloat16(i, count);
            MLAS_SVFLOAT16 v_x = MlasSveLoadFloat16(pg, &input[i]);
            MLAS_SVFLOAT16 v_scaled = MlasSveMulfloat16(pg, v_x, v_sqrt1_2);
            MlasSveStoreF16(pg, &temp[i], v_scaled);
            i += svcnth();
        }

        MlasSveErfF16Kernel(reinterpret_cast<const _mlas_fp16_*>(temp), reinterpret_cast<_mlas_fp16_*>(temp), count);

        int64_t j = 0;
        while (j < (count)) {
            svbool_t pg = MlasSveSelPredictefloat16(j, count);
            MLAS_SVFLOAT16 v_x = MlasSveLoadFloat16(pg, &input[j]);
            MLAS_SVFLOAT16 v_erf = MlasSveLoadFloat16(pg, &temp[j]);
            MLAS_SVFLOAT16 v_result = MlasSveMulfloat16(pg, v_half, MlasSveMulfloat16(pg, v_x, MlasSveAddfloat16(pg, v_one, v_erf)));
            MlasSveStoreF16(pg, &output[j], v_result);
            j += svcnth();
        }
    }
}
