/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    activation_kernel_rvv.cpp

Abstract:

    RVV unary activation kernels for riscv64: erf, tanh, logistic (sigmoid),
    exp, silu, gelu(erf). Wired through MLAS_PLATFORM kernel routine fields
    on builds with RVV support (MLAS_USE_RVV).

    LMUL=m4 throughout (32 floats per vector at VLEN=256), scaling with VLEN
    via dynamic vsetvli.

--*/

#include <riscv_vector.h>
#include <stddef.h>
#include "mlasi.h"

#if defined(MLAS_USE_RVV)

// vfmin/vfmax follow IEEE-754 minNum/maxNum, so a NaN input would be clamped to
// a finite limit. To let NaN round-trip, each kernel captures an is-NaN mask of
// the raw input and merges those lanes back into the result before the store.

namespace {

// Cody-Waite constants for exp(x).
constexpr float LOG2E         = 1.4426950408889634f;
constexpr float LN2_HI        = 0.6931381225585938f;
constexpr float LN2_LO        = 9.05800061733627e-6f;
// Lower bound applied by the activation kernels before their internal exp().
constexpr float EXP_CLAMP_MIN = -87.0f;
// Input guard for exp_f32m4's two-step 2^n reconstruction. +/-128 sits past
// float32 exp()'s saturation points (+inf above ln(FLT_MAX) ~= 88.7, 0 below
// -150*ln2 ~= -104) so clamping never alters a representable result, yet stays
// inside |x| ~< 174 where the split halves of n keep the exponent-bit trick in
// its valid [-126,127] range.
constexpr float EXP_RECON_MIN = -128.0f;
constexpr float EXP_RECON_MAX = 128.0f;
// SiLU clamps its sigmoid argument to this range -- mirrors the generic
// MlasLogisticConstants [-18, 18] so the sigmoid stays strictly inside (0, 1).
constexpr float LOGISTIC_CLAMP = 18.0f;
// Smallest positive normal float; used to floor 1+erf in GELU so the final
// multiply by x cannot evaluate to the NaN-producing (+/-inf) * 0.
constexpr float SMALLEST_NORMAL_F32 = 1.17549435e-38f;

// 6th-order minimax polynomial for exp(r), |r| <= ln2/2.
constexpr float C1 = 1.0f;
constexpr float C2 = 0.5f;
constexpr float C3 = 0.16666667f;
constexpr float C4 = 0.04166667f;
constexpr float C5 = 0.00833333f;
constexpr float C6 = 0.00138889f;

MLAS_FORCEINLINE
vfloat32m4_t exp_f32m4(vfloat32m4_t x, size_t vl)
{
    x = __riscv_vfmin_vf_f32m4(x, EXP_RECON_MAX, vl);
    x = __riscv_vfmax_vf_f32m4(x, EXP_RECON_MIN, vl);

    // n = round(x / ln2)
    const vfloat32m4_t xlog2e = __riscv_vfmul_vf_f32m4(x, LOG2E, vl);
    const vint32m4_t   n      = __riscv_vfcvt_x_f_v_i32m4(xlog2e, vl);
    const vfloat32m4_t nf     = __riscv_vfcvt_f_x_v_f32m4(n, vl);

    // r = x - n*ln2 (Cody-Waite two-step)
    vfloat32m4_t r = __riscv_vfnmsac_vf_f32m4(x, LN2_HI, nf, vl);
    r              = __riscv_vfnmsac_vf_f32m4(r, LN2_LO, nf, vl);

    // Horner polynomial for exp(r) = 1 + r*(1 + r*(C2 + r*(C3 + r*(C4 + r*(C5 + r*C6)))))
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(C6, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(C5, vl), r, p, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(C4, vl), r, p, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(C3, vl), r, p, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(C2, vl), r, p, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(C1, vl), r, p, vl);
    p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), r, p, vl);

    // 2^n via a two-step IEEE-754 exponent-bit construction: split n = n1 + n2
    // so each 2^ni stays a representable normal float. The single-step trick is
    // valid only for n in [-126, 127]; splitting spans exp's full range, so it
    // yields denormals near the underflow edge and +inf on overflow rather than
    // saturating valid finite results.
    const vint32m4_t   n1 = __riscv_vsra_vx_i32m4(n, 1, vl);
    const vint32m4_t   n2 = __riscv_vsub_vv_i32m4(n, n1, vl);
    const vfloat32m4_t s1 = __riscv_vreinterpret_v_i32m4_f32m4(
        __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n1, 127, vl), 23, vl));
    const vfloat32m4_t s2 = __riscv_vreinterpret_v_i32m4_f32m4(
        __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n2, 127, vl), 23, vl));
    return __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(p, s1, vl), s2, vl);
}

// Abramowitz & Stegun erf approximation (max error ~2.5e-5, sufficient for GELU)
constexpr float ERF_P  =  0.3275911f;
constexpr float ERF_A1 =  0.254829592f;
constexpr float ERF_A2 = -0.284496736f;
constexpr float ERF_A3 =  1.421413741f;
constexpr float ERF_A4 = -1.453152027f;
constexpr float ERF_A5 =  1.061405429f;

}  // namespace

extern "C"
void
MLASCALL
MlasErfKernelRvv(const float* Input, float* Output, size_t N)
{
    while (N > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t x     = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask  = __riscv_vmfne_vv_f32m4_b8(x, x, vl);
        const vfloat32m4_t abs_x = __riscv_vfabs_v_f32m4(x, vl);

        // t = 1 / (1 + p*|x|)
        const vfloat32m4_t t_den = __riscv_vfmacc_vf_f32m4(
            __riscv_vfmv_v_f_f32m4(1.0f, vl), ERF_P, abs_x, vl);
        const vfloat32m4_t t = __riscv_vfrdiv_vf_f32m4(t_den, 1.0f, vl);

        // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(ERF_A5, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A4, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A3, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A2, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A1, vl), t, p, vl);
        p = __riscv_vfmul_vv_f32m4(p, t, vl);

        // exp(-x^2)
        vfloat32m4_t neg_x2 = __riscv_vfneg_v_f32m4(
            __riscv_vfmul_vv_f32m4(abs_x, abs_x, vl), vl);
        neg_x2 = __riscv_vfmax_vf_f32m4(neg_x2, EXP_CLAMP_MIN, vl);
        const vfloat32m4_t exp_neg_x2 = exp_f32m4(neg_x2, vl);

        // erf = (1 - poly * exp(-x^2)) * sign(x)
        vfloat32m4_t result = __riscv_vfnmsac_vv_f32m4(
            __riscv_vfmv_v_f_f32m4(1.0f, vl), p, exp_neg_x2, vl);
        result = __riscv_vfsgnj_vv_f32m4(result, x, vl);
        result = __riscv_vmerge_vvm_f32m4(result, x, nan_mask, vl);

        __riscv_vse32_v_f32m4(Output, result, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

extern "C"
void
MLASCALL
MlasTanhKernelRvv(const float* Input, float* Output, size_t N)
{
    while (N > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t x_in = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(x_in, x_in, vl);

        // tanh saturates for |x| >= 9
        vfloat32m4_t x = __riscv_vfmin_vf_f32m4(x_in, 9.0f, vl);
        x = __riscv_vfmax_vf_f32m4(x, -9.0f, vl);

        vfloat32m4_t two_x = __riscv_vfmul_vf_f32m4(x, 2.0f, vl);
        two_x              = __riscv_vfmax_vf_f32m4(two_x, EXP_CLAMP_MIN, vl);
        const vfloat32m4_t e2x = exp_f32m4(two_x, vl);

        const vfloat32m4_t num = __riscv_vfsub_vf_f32m4(e2x, 1.0f, vl);
        const vfloat32m4_t den = __riscv_vfadd_vf_f32m4(e2x, 1.0f, vl);
        vfloat32m4_t v   = __riscv_vfdiv_vv_f32m4(num, den, vl);
        v = __riscv_vmerge_vvm_f32m4(v, x_in, nan_mask, vl);

        __riscv_vse32_v_f32m4(Output, v, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

extern "C"
void
MLASCALL
MlasLogisticKernelRvv(const float* Input, float* Output, size_t N)
{
    while (N > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t x = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(x, x, vl);
        vfloat32m4_t neg_x   = __riscv_vfneg_v_f32m4(x, vl);
        neg_x                = __riscv_vfmax_vf_f32m4(neg_x, EXP_CLAMP_MIN, vl);
        const vfloat32m4_t exp_neg = exp_f32m4(neg_x, vl);
        const vfloat32m4_t den     = __riscv_vfadd_vf_f32m4(exp_neg, 1.0f, vl);
        vfloat32m4_t result        = __riscv_vfrdiv_vf_f32m4(den, 1.0f, vl);
        result = __riscv_vmerge_vvm_f32m4(result, x, nan_mask, vl);
        __riscv_vse32_v_f32m4(Output, result, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

extern "C"
void
MLASCALL
MlasComputeExpF32KernelRvv(const float* Input, float* Output, size_t N)
{
    while (N > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t v_in = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(v_in, v_in, vl);
        vfloat32m4_t v = exp_f32m4(v_in, vl);
        v = __riscv_vmerge_vvm_f32m4(v, v_in, nan_mask, vl);
        __riscv_vse32_v_f32m4(Output, v, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Fused in a single pass to halve load/store traffic vs a two-kernel
// composition of logistic + multiply.
extern "C"
void
MLASCALL
MlasSiluKernelRvv(const float* Input, float* Output, size_t N)
{
    while (N > 0) {
        const size_t vl        = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t x   = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(x, x, vl);
        vfloat32m4_t neg_x     = __riscv_vfneg_v_f32m4(x, vl);
        // Clamp the sigmoid argument to the generic logistic range [-18, 18];
        // otherwise exp(-x) overflows to +inf, sigmoid becomes exactly 0, and
        // x*sigmoid produces NaN for x = -inf and -0 for large negative x.
        neg_x                  = __riscv_vfmin_vf_f32m4(
            __riscv_vfmax_vf_f32m4(neg_x, -LOGISTIC_CLAMP, vl), LOGISTIC_CLAMP, vl);
        const vfloat32m4_t en  = exp_f32m4(neg_x, vl);
        const vfloat32m4_t den = __riscv_vfadd_vf_f32m4(en, 1.0f, vl);
        const vfloat32m4_t sig = __riscv_vfrdiv_vf_f32m4(den, 1.0f, vl);
        vfloat32m4_t out       = __riscv_vfmul_vv_f32m4(x, sig, vl);
        out = __riscv_vmerge_vvm_f32m4(out, x, nan_mask, vl);
        __riscv_vse32_v_f32m4(Output, out, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

// GELU (erf variant): 0.5 * x * (1 + erf(x / sqrt(2)))
// Reuses the Abramowitz & Stegun erf approximation inlined for fusion,
// sharing the ERF_A1..A5 / ERF_P constants with MlasErfKernelRvv.
extern "C"
void
MLASCALL
MlasGeluErfKernelRvv(const float* Input, float* Output, size_t N)
{
    constexpr float INV_SQRT_2 = 0.70710678118654752f;

    while (N > 0) {
        const size_t vl           = __riscv_vsetvl_e32m4(N);
        const vfloat32m4_t x      = __riscv_vle32_v_f32m4(Input, vl);
        const vbool8_t nan_mask   = __riscv_vmfne_vv_f32m4_b8(x, x, vl);
        const vfloat32m4_t sx     = __riscv_vfmul_vf_f32m4(x, INV_SQRT_2, vl);
        const vfloat32m4_t abs_sx = __riscv_vfabs_v_f32m4(sx, vl);

        // t = 1 / (1 + p * |sx|)
        const vfloat32m4_t t_den = __riscv_vfmacc_vf_f32m4(
            __riscv_vfmv_v_f_f32m4(1.0f, vl), ERF_P, abs_sx, vl);
        const vfloat32m4_t t = __riscv_vfrdiv_vf_f32m4(t_den, 1.0f, vl);

        // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(ERF_A5, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A4, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A3, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A2, vl), t, p, vl);
        p = __riscv_vfmacc_vv_f32m4(__riscv_vfmv_v_f_f32m4(ERF_A1, vl), t, p, vl);
        p = __riscv_vfmul_vv_f32m4(p, t, vl);

        // exp(-sx^2)
        vfloat32m4_t neg_x2 = __riscv_vfneg_v_f32m4(
            __riscv_vfmul_vv_f32m4(abs_sx, abs_sx, vl), vl);
        neg_x2 = __riscv_vfmax_vf_f32m4(neg_x2, EXP_CLAMP_MIN, vl);
        const vfloat32m4_t exp_neg_x2 = exp_f32m4(neg_x2, vl);

        // erf(sx) = (1 - poly * exp(-sx^2)) * sign(sx)
        vfloat32m4_t erf_val = __riscv_vfnmsac_vv_f32m4(
            __riscv_vfmv_v_f_f32m4(1.0f, vl), p, exp_neg_x2, vl);
        erf_val = __riscv_vfsgnj_vv_f32m4(erf_val, sx, vl);

        // 0.5 * x * (1 + erf(sx)). erf saturates to -1 for large negative x, so
        // 1+erf can be exactly 0; floor it to the smallest normal float so the
        // multiply by x cannot become the NaN-producing (+/-inf) * 0.
        vfloat32m4_t one_plus_erf = __riscv_vfadd_vf_f32m4(erf_val, 1.0f, vl);
        one_plus_erf = __riscv_vfmax_vf_f32m4(one_plus_erf, SMALLEST_NORMAL_F32, vl);
        vfloat32m4_t out = __riscv_vfmul_vv_f32m4(x, one_plus_erf, vl);
        out = __riscv_vfmul_vf_f32m4(out, 0.5f, vl);
        out = __riscv_vmerge_vvm_f32m4(out, x, nan_mask, vl);

        __riscv_vse32_v_f32m4(Output, out, vl);
        Input  += vl;
        Output += vl;
        N      -= vl;
    }
}

#endif  // MLAS_USE_RVV
