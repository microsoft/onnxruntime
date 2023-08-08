/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    activate_fp16.cpp

Abstract:

    This module implements the activation routines for fp16 data types

--*/

#include "fp16_common.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

//
// Templates for activation functions.
//

template<MLAS_ACTIVATION_KIND ActivationKind>
struct MLAS_HALF_ACTIVATION_FUNCTION;

template <>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasIdentityActivation> {
    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value) { return Value; }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value) { return Value; }

    float Activate(float Value) { return Value; }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasReluActivation>
{
    const MLAS_FLOAT16X8 ZeroVec = MlasZeroFloat16x8();

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        return MlasMaximumFloat16x8(ZeroVec, Value);
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        return MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(ZeroVec), Value);
    }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasLeakyReluActivation>
{
    const MLAS_FLOAT16X8 ZeroVec = MlasZeroFloat16x8();

    MLAS_FLOAT16X8 AlphaBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ alpha = MLAS_Float2Half(Activation.Parameters.LeakyRelu.alpha);
        AlphaBroadcast = MlasBroadcastFloat16x8(alpha);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        MLAS_FLOAT16X8 ValueTimesAlpha = MlasMultiplyFloat16x8(Value, AlphaBroadcast);
        return MlasBitwiseSelectFloat16x8(MlasCmpLessEqualFloat16x8(Value, ZeroVec),
                                          ValueTimesAlpha, Value);
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        MLAS_FLOAT16X4 ValueTimesAlpha =
            MlasMultiplyFloat16x4(Value, MlasToLowHalfFloat16x4(AlphaBroadcast));
        return MlasBitwiseSelectFloat16x4(
            MlasCmpLessEqualFloat16x4(Value, MlasToLowHalfFloat16x4(ZeroVec)), ValueTimesAlpha,
            Value);
    }
};

template <>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasTanhActivation> {
#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)
    //
    // Ported from XNNPACK (f16-tanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2-div.c)
    //

    //
    // Constants for float16x8_t
    //

    // The smallest z for which tanhh(-z) is saturated at -1.0h.
    const float16x8_t vsat_cutoff_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4482)));  // 0x1.208p+2h
    // Large number such that ulp(magic bias) == 0.5 and magic bias === 7.5 mod 2**8.
    const float16x8_t vmagic_bias_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x620F)));  // 0x1.83Cp+9h
    const float16x8_t vminus_log2e_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
    const float16x8_t vln2_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
    // Coefficients of polynomial approximation
    //   exp(-2t) - 1 ~ t * (-2 + t * (c2 + t * c3))
    // on [-log(2)/4, log(2)/4]
    const float16x8_t vc3_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBD5B)));  // -0x1.56Cp+0h
    const float16x8_t vc2_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4008)));  // 0x1.020p+1h
    const float16x8_t vtwo_16x8 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4000)));  // 2.0h
    const float16x8_t vminus_one_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC00)));  // -1.0h
    // Mask for the sign bit.
    const uint16x8_t vsign_mask_16x8 = vmovq_n_u16(UINT16_C(0x8000));

    //
    // Constants for float16x4_t
    //

    // The smallest z for which tanhh(-z) is saturated at -1.0h.
    const float16x4_t vsat_cutoff_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x4482)));  // 0x1.208p+2h
    // Large number such that ulp(magic bias) == 0.5 and magic bias === 7.5 mod 2**8.
    const float16x4_t vmagic_bias_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x620F)));  // 0x1.83Cp+9h
    const float16x4_t vminus_log2e_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
    const float16x4_t vln2_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
    // Coefficients of polynomial approximation
    //   exp(-2t) - 1 ~ t * (-2 + t * (c2 + t * c3))
    // on [-log(2)/4, log(2)/4]
    const float16x4_t vc3_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xBD5B)));  // -0x1.56Cp+0h
    const float16x4_t vc2_16x4 = vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x4008)));  // 0x1.020p+1h
    const float16x4_t vtwo_16x4 = vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x4000)));  // 2.0h
    const float16x4_t vminus_one_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xBC00)));  // -1.0h
    // Mask for the sign bit.
    const uint16x4_t vsign_mask_16x4 = vmov_n_u16(UINT16_C(0x8000));

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 vx)
    {
        // General structure of the algorithm:
        //
        //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
        //   f(x) :=
        //           \ -f(-x) if x <= 0
        //
        // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
        // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
        float16x8_t vz = vabsq_f16(vx);

        // The function saturates at -1 for large positive inputs: tanhh(-z) == -1.0h for z >=
        // sat_cutoff ~= 9.010913. To guarantee this behavior, we clip input z at sat_cutoff, and
        // leverage the fact that for our implementation tanhf(sat_cutoff) == -1.0h. NaN inputs are
        // passed unchanged.
        vz = vminq_f16(vz, vsat_cutoff_16x8);

        // Compute reduced argument n := round(-z / log(2), 1).
        // We do it by adding a large number (magic bias), which cause rounding of the result to 1
        // fractional bit, then subtracing the large number back. The trick with adding large number
        // is valid only within certain bounds
        // (|-z / log(2)| <= 2**10, i.e. |z| <= 0x1.630p+7 = 177.5), but that is acceptable, because
        // inputs x outside of [-4.5078125, 4.5078125] (i.e. z outsize [0, 4.5078125]) saturate
        // tanhh(x). Additionally, we fuse addition of the floating-point exponent bias (15) into
        // the magic bias. Note that addition-subtraction of the large number doesn't cause overflow
        // for inputs in this range.
        float16x8_t vn = vfmaq_f16(vmagic_bias_16x8, vz, vminus_log2e_16x8);

        // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't
        // cause underflow, i.e. 0 <= z <= 4.5078125, and -7 <= n <= 0 accordingly.
        const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

        // Subtract the large number back to get final n := round(-z / log(2), 1) as a
        // floating-point number.
        vn = vsubq_f16(vn, vmagic_bias_16x8);

        // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
        const float16x8_t vt = vfmaq_f16(vz, vn, vln2_16x8);

        // Compute degree-3 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
        //   P(t) = t * (-2 + t * (c2 + t * c3))
        //        = t * (-p)
        float16x8_t vp = vfmaq_f16(vc2_16x8, vc3_16x8, vt);
        vp = vfmsq_f16(vtwo_16x8, vp, vt);

        // Reconstruct the exp(-2z) - 1 value:
        //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * c3)) + 1) - 1
        //                = s * t * (-p) + (s - 1)
        //                = (s - 1) - (t * s) * p
        const float16x8_t vts = vmulq_f16(vt, vs);
        const float16x8_t vsmo = vaddq_f16(vs, vminus_one_16x8);
        const float16x8_t vemo = vfmsq_f16(vsmo, vp, vts);

        // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
        const float16x8_t vepo = vaddq_f16(vemo, vtwo_16x8);

        // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
        float16x8_t vy = vdivq_f16(vemo, vepo);

        // Reconstruct tanh(x) = copysign(y, x)
        vy = vbslq_f16(vsign_mask_16x8, vx, vy);

        return vy;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 vx)
    {
        // General structure of the algorithm:
        //
        //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
        //   f(x) :=
        //           \ -f(-x) if x <= 0
        //
        // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
        // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
        float16x4_t vz = vabs_f16(vx);

        // The function saturates at -1 for large positive inputs: tanhh(-z) == -1.0h for z >=
        // sat_cutoff ~= 9.010913. To guarantee this behavior, we clip input z at sat_cutoff, and
        // leverage the fact that for our implementation tanhf(sat_cutoff) == -1.0h. NaN inputs are
        // passed unchanged.
        vz = vmin_f16(vz, vsat_cutoff_16x4);

        // Compute reduced argument n := round(-z / log(2), 1).
        // We do it by adding a large number (magic bias), which cause rounding of the result to 1
        // fractional bit, then subtracing the large number back. The trick with adding large number
        // is valid only within certain bounds
        // (|-z / log(2)| <= 2**10, i.e. |z| <= 0x1.630p+7 = 177.5), but that is acceptable, because
        // inputs x outside of [-4.5078125, 4.5078125] (i.e. z outsize [0, 4.5078125]) saturate
        // tanhh(x). Additionally, we fuse addition of the floating-point exponent bias (15) into
        // the magic bias. Note that addition-subtraction of the large number doesn't cause overflow
        // for inputs in this range.
        float16x4_t vn = vfma_f16(vmagic_bias_16x4, vz, vminus_log2e_16x4);

        // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't
        // cause underflow, i.e. 0 <= z <= 4.5078125, and -7 <= n <= 0 accordingly.
        const float16x4_t vs = vreinterpret_f16_s16(vshl_n_s16(vreinterpret_s16_f16(vn), 10));

        // Subtract the large number back to get final n := round(-z / log(2), 1) as a
        // floating-point number.
        vn = vsub_f16(vn, vmagic_bias_16x4);

        // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
        const float16x4_t vt = vfma_f16(vz, vn, vln2_16x4);

        // Compute degree-3 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
        //   P(t) = t * (-2 + t * (c2 + t * c3))
        //        = t * (-p)
        float16x4_t vp = vfma_f16(vc2_16x4, vc3_16x4, vt);
        vp = vfms_f16(vtwo_16x4, vp, vt);

        // Reconstruct the exp(-2z) - 1 value:
        //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * c3)) + 1) - 1
        //                = s * t * (-p) + (s - 1)
        //                = (s - 1) - (t * s) * p
        const float16x4_t vts = vmul_f16(vt, vs);
        const float16x4_t vsmo = vadd_f16(vs, vminus_one_16x4);
        const float16x4_t vemo = vfms_f16(vsmo, vp, vts);

        // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
        const float16x4_t vepo = vadd_f16(vemo, vtwo_16x4);

        // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
        float16x4_t vy = vdiv_f16(vemo, vepo);

        // Reconstruct tanh(x) = copysign(y, x)
        vy = vbsl_f16(vsign_mask_16x4, vx, vy);

        return vy;
    }
#else
    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 vx)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 vx)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }
#endif
};

template <>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasLogisticActivation> {
#if defined(MLAS_TARGET_ARM64) || defined(MLAS_TARGET_ARM64EC)
    //
    // Ported from XNNPACK (f16-sigmoid-aarch64-neonfp16arith-rr2-p3-div.c).
    //

    //
    // Constants for float16x8_t
    //

    // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
    const float16x8_t vmagic_bias_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
    const float16x8_t vminus_log2e_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
    const float16x8_t vln2_hi_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
    const float16x8_t vln2_lo_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
    // Coefficient of polynomial approximation
    //   exp(-t) ~ 1 + t * (c1 + t * c2)
    // on [-log(2)/2, log(2)/2]
    const float16x8_t vc3_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB156)));  // -0x1.558p-3h
    const float16x8_t vc2_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h
    const float16x8_t vone_16x8 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
    // The largest z for which sigmoidh(-z) is normalized.
    // This number is also the largest z for which exph(-z) is normalized.
    const float16x8_t vdenorm_cutoff_16x8 =
        vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

    //
    // Constants for float16x4_t
    //

    // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
    const float16x4_t vmagic_bias_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
    const float16x4_t vminus_log2e_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
    const float16x4_t vln2_hi_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
    const float16x4_t vln2_lo_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
    // Coefficient of polynomial approximation
    //   exp(-t) ~ 1 + t * (c1 + t * c2)
    // on [-log(2)/2, log(2)/2]
    const float16x4_t vc3_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xB156)));  // -0x1.558p-3h
    const float16x4_t vc2_16x4 = vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h
    const float16x4_t vone_16x4 = vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0x3C00)));  // 1.0h
    // The largest z for which sigmoidh(-z) is normalized.
    // This number is also the largest z for which exph(-z) is normalized.
    const float16x4_t vdenorm_cutoff_16x4 =
        vreinterpret_f16_u16(vmov_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 vx)
    {
        // General structure of the algorithm:
        //
        //           / exp(x) / (1 + exp(x)) if x <= 0
        //   f[x] :=
        //           \ 1 - f[-x] if x >= 0
        //
        // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
        // then replace result with 1 - f[-z] if x >= 0.
        const float16x8_t vz = vabsq_f16(vx);

        // Compute reduced argument n := round(-z / ln2).
        // We do it by adding a large number (magic bias) to the product z * (-1/ln2), which
        // cause rounding of the result to an integer, then subtracing the large number back. The
        // first addition is combined with multiplication by -log2e into a single FMA instruction.
        // The trick with adding large number is valid only within certain bounds
        // (|-x / ln2| <= 2**9, i.e. |z| <= 0x1.630p+8 = 355.0), but that is acceptable, because
        // inputs outside of [-9.703125, 8.3125] (i.e. z outside [0, 9.703125]) underflow or
        // saturate sigmoidh(x). We fixup the result for such inputs at the very end of the
        // algorithm.
        float16x8_t vn = vfmaq_f16(vmagic_bias_16x8, vz, vminus_log2e_16x8);

        // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause
        // underflow, i.e. -9.703125 <= -z <= 0.0, and -14 <= n <= 0 accordingly.
        const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

        // Subtract the large number back to get the final n := round(-z / ln2) as a
        // floating-point number.
        vn = vsubq_f16(vn, vmagic_bias_16x8);

        // Compute reduced argument t := z - n * log(2). Note that -t = -z - n * log(2).
        // Use Cody-Waite range reduction method (note two constants to represent -ln(2)) to
        // improve accuracy.
        float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi_16x8);
        vt = vfmaq_f16(vt, vn, vln2_lo_16x8);

        // Compute degree-3 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
        //   P(t) = 1 + t * (-1 + t * (c2 + t * c3)) = -(1 - t * p)
        float16x8_t vp = vfmaq_f16(vc2_16x8, vc3_16x8, vt);
        vp = vfmsq_f16(vone_16x8, vp, vt);

        // Reconstruct the exp(-z) value:
        //   e = s * (1 + t * (-1 + t * (c2 + t * c3))
        //     = s * (1 - t * (-p))
        //     = s - (t * s) * (-p)
        vt = vmulq_f16(vt, vs);
        float16x8_t ve = vfmsq_f16(vs, vp, vt);

        // Denominator of the sigmoid fraction: 1.0 + exp(-z)
        float16x8_t vd = vaddq_f16(ve, vone_16x8);

        // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
        float16x8_t vf = vdivq_f16(ve, vd);

        // For inputs below denormal cutoff, replace output with +0.0f.
        // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
        vf = vreinterpretq_f16_u16(
            vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff_16x8)));

        // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
        const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
        vf = vbslq_f16(vm, vf, vsubq_f16(vone_16x8, vf));

        return vf;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 vx)
    {
        // General structure of the algorithm:
        //
        //           / exp(x) / (1 + exp(x)) if x <= 0
        //   f[x] :=
        //           \ 1 - f[-x] if x >= 0
        //
        // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
        // then replace result with 1 - f[-z] if x >= 0.
        const float16x4_t vz = vabs_f16(vx);

        // Compute reduced argument n := round(-z / ln2).
        // We do it by adding a large number (magic bias) to the product z * (-1/ln2), which
        // cause rounding of the result to an integer, then subtracing the large number back. The
        // first addition is combined with multiplication by -log2e into a single FMA instruction.
        // The trick with adding large number is valid only within certain bounds
        // (|-x / ln2| <= 2**9, i.e. |z| <= 0x1.630p+8 = 355.0), but that is acceptable, because
        // inputs outside of [-9.703125, 8.3125] (i.e. z outside [0, 9.703125]) underflow or
        // saturate sigmoidh(x). We fixup the result for such inputs at the very end of the
        // algorithm.
        float16x4_t vn = vfma_f16(vmagic_bias_16x4, vz, vminus_log2e_16x4);

        // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause
        // underflow, i.e. -9.703125 <= -z <= 0.0, and -14 <= n <= 0 accordingly.
        const float16x4_t vs = vreinterpret_f16_s16(vshl_n_s16(vreinterpret_s16_f16(vn), 10));

        // Subtract the large number back to get the final n := round(-z / ln2) as a
        // floating-point number.
        vn = vsub_f16(vn, vmagic_bias_16x4);

        // Compute reduced argument t := z - n * log(2). Note that -t = -z - n * log(2).
        // Use Cody-Waite range reduction method (note two constants to represent -ln2) to
        // improve accuracy.
        float16x4_t vt = vfma_f16(vz, vn, vln2_hi_16x4);
        vt = vfma_f16(vt, vn, vln2_lo_16x4);

        // Compute degree-3 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
        //   P(t) = 1 + t * (-1 + t * (c2 + t * c3)) = -(1 - t * p)
        float16x4_t vp = vfma_f16(vc2_16x4, vc3_16x4, vt);
        vp = vfms_f16(vone_16x4, vp, vt);

        // Reconstruct the exp(-z) value:
        //   e = s * (1 + t * (-1 + t * (c2 + t * c3))
        //     = s * (1 - t * (-p))
        //     = s - (t * s) * (-p)
        vt = vmul_f16(vt, vs);
        float16x4_t ve = vfms_f16(vs, vp, vt);

        // Denominator of the sigmoid fraction: 1.0 + exp(-z)
        float16x4_t vd = vadd_f16(ve, vone_16x4);

        // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
        float16x4_t vf = vdiv_f16(ve, vd);

        // For inputs below denormal cutoff, replace output with +0.0f.
        // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
        vf = vreinterpret_f16_u16(
            vbic_u16(vreinterpret_u16_f16(vf), vcagt_f16(vx, vdenorm_cutoff_16x4)));

        // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
        const uint16x4_t vm = vclt_f16(vx, vreinterpret_f16_u16(vmov_n_u16(0)));
        vf = vbsl_f16(vm, vf, vsub_f16(vone_16x4, vf));

        return vf;
    }
#else
    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 vx)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 vx)
    {
        MLAS_UNREFERENCED_PARAMETER(Activation);
        MLAS_THROW_EX(std::runtime_error, "unsupported target architecture");
    }
#endif
};

template <>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasClipActivation> {
    MLAS_FLOAT16X8 MinimumBroadcast;
    MLAS_FLOAT16X8 MaximumBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ min = MLAS_Float2Half(Activation.Parameters.Clip.minimum);
        MinimumBroadcast = MlasBroadcastFloat16x8(min);
        const _mlas_fp16_ max = MLAS_Float2Half(Activation.Parameters.Clip.maximum);
        MaximumBroadcast = MlasBroadcastFloat16x8(max);
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        Value = MlasMaximumFloat16x8(MinimumBroadcast, Value);
        Value = MlasMinimumFloat16x8(MaximumBroadcast, Value);

        return Value;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        Value = MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(MinimumBroadcast), Value);
        Value = MlasMinimumFloat16x4(MlasToLowHalfFloat16x4(MaximumBroadcast), Value);
        return Value;
    }
};

template<>
struct MLAS_HALF_ACTIVATION_FUNCTION<MlasHardSigmoidActivation>
{
    MLAS_FLOAT16X8 AlphaBroadcast;
    MLAS_FLOAT16X8 BetaBroadcast;
    MLAS_FLOAT16X8 MinimumBroadcast;
    MLAS_FLOAT16X8 MaximumBroadcast;

    MLAS_HALF_ACTIVATION_FUNCTION(const MLAS_ACTIVATION& Activation)
    {
        const _mlas_fp16_ alpha = MLAS_Float2Half(Activation.Parameters.HardSigmoid.alpha);
        AlphaBroadcast = MlasBroadcastFloat16x8(alpha);
        const _mlas_fp16_ beta = MLAS_Float2Half(Activation.Parameters.HardSigmoid.beta);
        BetaBroadcast = MlasBroadcastFloat16x8(beta);
        MinimumBroadcast = MlasZeroFloat16x8();
        MaximumBroadcast = MlasBroadcastFloat16x8(MLAS_Float2Half(1.0f));
    }

    MLAS_FLOAT16X8 Activate(MLAS_FLOAT16X8 Value)
    {
        Value = MlasMultiplyAddFloat16x8(Value, AlphaBroadcast, BetaBroadcast);
        Value = MlasMinimumFloat16x8(MaximumBroadcast, Value);
        Value = MlasMaximumFloat16x8(MinimumBroadcast, Value);

        return Value;
    }

    MLAS_FLOAT16X4 Activate(MLAS_FLOAT16X4 Value)
    {
        Value = MlasMultiplyAddFloat16x4(Value, MlasToLowHalfFloat16x4(AlphaBroadcast),
                                         MlasToLowHalfFloat16x4(BetaBroadcast));
        Value = MlasMinimumFloat16x4(MlasToLowHalfFloat16x4(MaximumBroadcast), Value);
        Value = MlasMaximumFloat16x4(MlasToLowHalfFloat16x4(MinimumBroadcast), Value);

        return Value;
    }
};

template<MLAS_ACTIVATION_KIND ActivationKind>
inline
void
MlasActivationKernel(
    const MLAS_ACTIVATION& Activation,
    _mlas_fp16_* Buffer,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    )
{
    MLAS_HALF_ACTIVATION_FUNCTION<ActivationKind> ActivationFunction(Activation);

    auto* CRow = Buffer + StartM * ldc + StartN;

    while (CountM-- > 0) {
        _mlas_fp16_* buffer = CRow;
        size_t n = CountN;

        while (n >= 8) {
            MLAS_FLOAT16X8 Vector = MlasLoadFloat16x8(buffer);
            MlasStoreFloat16x8(buffer, ActivationFunction.Activate(Vector));
            buffer += 8;
            n -= 8;
        }

        if (n >= 4) {
            MLAS_FLOAT16X4 Vector = MlasLoadFloat16x4(buffer);
            MlasStoreFloat16x4(buffer, ActivationFunction.Activate(Vector));
            buffer += 4;
            n -= 4;
        }

        if (n > 0) {
            MLAS_FLOAT16X4 buf;
            std::memcpy(&buf, buffer, n * sizeof(_mlas_fp16_));
            MLAS_FLOAT16X4 res = ActivationFunction.Activate(buf);
            MlasStorePartialFloat16x4(buffer, res, n);
        }

        CRow += ldc;
    }
}

template<>
inline
void
MlasActivationKernel<MlasIdentityActivation>(
    const MLAS_ACTIVATION& Activation,
    _mlas_fp16_* Buffer,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    )
{
    //
    // No operation.
    //

    MLAS_UNREFERENCED_PARAMETER(Activation);
    MLAS_UNREFERENCED_PARAMETER(Buffer);
    MLAS_UNREFERENCED_PARAMETER(StartM);
    MLAS_UNREFERENCED_PARAMETER(StartN);
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(ldc);
}


template<MLAS_ACTIVATION_KIND ActivationKind>
MLAS_FORCEINLINE
void
MlasActivationKernel(
    const MLAS_ACTIVATION& Activation,
    _mlas_fp16_* Buffer,
    const _mlas_fp16_* Addon,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    )
{
    MLAS_HALF_ACTIVATION_FUNCTION<ActivationKind> ActivationFunction(Activation);

    auto* CRow = Buffer + StartM * ldc + StartN;
    const auto* ARow = Addon + StartM * ldc + StartN;

    while (CountM-- > 0) {
        auto* buffer = CRow;
        const auto* addsrc = ARow;
        size_t n = CountN;

        while (n >= 8) {
            MLAS_FLOAT16X8 AVec = MlasLoadFloat16x8(addsrc);
            MLAS_FLOAT16X8 Vector = MlasLoadFloat16x8(buffer);
            addsrc += 8;
            Vector = MlasAddFloat16x8(Vector, AVec);
            Vector = ActivationFunction.Activate(Vector);
            MlasStoreFloat16x8(buffer, Vector);
            buffer += 8;
            n -= 8;
        }

        if (n >= 4) {
            MLAS_FLOAT16X4 AVec = MlasLoadFloat16x4(addsrc);
            MLAS_FLOAT16X4 Vector = MlasLoadFloat16x4(buffer);
            addsrc += 4;
            Vector = MlasAddFloat16x4(Vector, AVec);
            Vector = ActivationFunction.Activate(Vector);
            MlasStoreFloat16x4(buffer, Vector);
            buffer += 4;
            n -= 4;
        }

        if (n > 0) {
            MLAS_FLOAT16X4 addbuf;
            MLAS_FLOAT16X4 buf;
            std::memcpy(&addbuf, addsrc, n * sizeof(_mlas_fp16_));
            std::memcpy(&buf, buffer, n * sizeof(_mlas_fp16_));
            buf = MlasAddFloat16x4(buf, addbuf);
            buf = ActivationFunction.Activate(buf);
            MlasStorePartialFloat16x4(buffer, buf, n);
        }

        CRow += ldc;
        ARow += ldc;
    }
}


void
MLAS_HALF_GEMM_ACTIVATION_PROCESSOR::Process(
    MLAS_FP16* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    auto* Buffer = reinterpret_cast<_mlas_fp16_*>(C);
    switch (Activation_.ActivationKind) {
        case MlasIdentityActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasIdentityActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasIdentityActivation>(Activation_, Buffer, StartM, StartN,
                                                             CountM, CountN, ldc);
            }
            break;
        }

        case MlasReluActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasReluActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasReluActivation>(Activation_, Buffer, StartM, StartN,
                                                         CountM, CountN, ldc);
            }
            break;
        }

        case MlasLeakyReluActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasLeakyReluActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasLeakyReluActivation>(Activation_, Buffer, StartM, StartN,
                                                              CountM, CountN, ldc);
            }
            break;
        }

        case MlasTanhActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasTanhActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasTanhActivation>(Activation_, Buffer, StartM, StartN,
                                                         CountM, CountN, ldc);
            }
            break;
        }

        case MlasLogisticActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasLogisticActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasLogisticActivation>(Activation_, Buffer, StartM, StartN,
                                                             CountM, CountN, ldc);
            }
            break;
        }

        case MlasClipActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasClipActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasClipActivation>(Activation_, Buffer, StartM, StartN,
                                                         CountM, CountN, ldc);
            }
            break;
        }

        case MlasHardSigmoidActivation: {
            if (SumBuf_) {
                MlasActivationKernel<MlasHardSigmoidActivation>(
                    Activation_, Buffer, reinterpret_cast<const _mlas_fp16_*>(SumBuf_), StartM,
                    StartN, CountM, CountN, ldc);
            } else {
                MlasActivationKernel<MlasHardSigmoidActivation>(Activation_, Buffer, StartM, StartN,
                                                                CountM, CountN, ldc);
            }
            break;
        }

        default: {
            MLAS_THROW_EX(std::runtime_error, "bad mlas activation kind");
            return;
        }
    }
}

#else
// Really dumb implementation when fp16 acceleration is not supported

#include <vector>

MLAS_FORCEINLINE
void
CvtFloat2Half(
    _mlas_fp16_* dest,
    const float* src,
    size_t len
)
{
    for (size_t i = 0; i < len; i++) {
        *dest++ = MLAS_Float2Half(*src++);
    }
}

void
MLAS_HALF_GEMM_ACTIVATION_PROCESSOR::Process(
    MLAS_FP16* C,
    size_t StartM,
    size_t StartN,
    size_t CountM,
    size_t CountN,
    size_t ldc
    ) const
{
    std::vector<float> buffer(CountM*CountN);

    _mlas_fp16_* Output = reinterpret_cast<_mlas_fp16_*>(C);
    auto* CRow = buffer.data();
    const _mlas_fp16_* CAdd = nullptr;
    if (SumBuf_) {
        CAdd = reinterpret_cast<const _mlas_fp16_*>(SumBuf_) + StartM * ldc + StartN;
    }
    Output += StartM * ldc + StartN;

    while (CountM-- > 0) {
        if (CAdd) {
            for (size_t n = 0; n < CountN; n++) {
                CRow[n] += MLAS_Half2Float(CAdd[n]);
            }
            CAdd += ldc;
        }
        MlasActivation(&this->Activation_, CRow, nullptr, 1, CountN, CountN);

        CvtFloat2Half(Output, CRow, CountN);
        CRow += CountN;
        Output += ldc;
    }
}

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
