/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_kernel_rvv.cpp

Abstract:

    This module implements RVV kernels for the softmax critical path on
    riscv64. The implementation keeps the scope intentionally small and
    focuses on the float32 primitives used by Softmax and LogSoftmax:
    reduction, sum-exp, normalization, and log-softmax output.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace {

constexpr float kExpLowerRangeSumExp = -88.3762626647949f;
constexpr float kRoundingBias = MLAS_ROUNDING_BIAS_MAGIC;
constexpr float kLog2Reciprocal = 1.44269504088896341f;
constexpr float kLog2High = -6.93145752e-1f;
constexpr float kLog2Low = -1.42860677e-6f;
constexpr float kPoly0 = 0x1.694000p-10f;
constexpr float kPoly1 = 0x1.125edcp-7f;
constexpr float kPoly2 = 0x1.555b5ap-5f;
constexpr float kPoly3 = 0x1.555450p-3f;
constexpr float kPoly4 = 0x1.fffff6p-2f;
constexpr float kPoly56 = 0x1.000000p+0f;
constexpr int32_t kMaximumExponentBits = 0x3F800000;

MLAS_FORCEINLINE
vfloat32m1_t
MlasComputeExpVectorRvv(
    vfloat32m1_t value,
    size_t vl
    )
{
    value = __riscv_vfmax_vf_f32m1(value, kExpLowerRangeSumExp, vl);

    vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(value, kLog2Reciprocal, vl);
    vfloat32m1_t biased = __riscv_vfadd_vf_f32m1(scaled, kRoundingBias, vl);
    vfloat32m1_t reduced_m = __riscv_vfsub_vf_f32m1(biased, kRoundingBias, vl);
    vfloat32m1_t reduced = __riscv_vfadd_vv_f32m1(
        __riscv_vfmul_vf_f32m1(reduced_m, kLog2High, vl), value, vl);
    reduced = __riscv_vfadd_vv_f32m1(
        __riscv_vfmul_vf_f32m1(reduced_m, kLog2Low, vl), reduced, vl);

    vfloat32m1_t poly = __riscv_vfmv_v_f_f32m1(kPoly0, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly1, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly2, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly3, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly4, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly56, vl);
    poly = __riscv_vfadd_vf_f32m1(
        __riscv_vfmul_vv_f32m1(poly, reduced, vl), kPoly56, vl);

    vint32m1_t exponent_bits = __riscv_vreinterpret_v_f32m1_i32m1(biased);
    exponent_bits = __riscv_vsll_vx_i32m1(exponent_bits, 23, vl);
    exponent_bits = __riscv_vadd_vx_i32m1(exponent_bits, kMaximumExponentBits, vl);
    vfloat32m1_t scale = __riscv_vreinterpret_v_i32m1_f32m1(exponent_bits);

    return __riscv_vfmul_vv_f32m1(poly, scale, vl);
}

MLAS_FORCEINLINE
float
MlasReduceSumRvv(
    vfloat32m1_t value,
    size_t vl
    )
{
    vfloat32m1_t accumulator = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    accumulator = __riscv_vfredusum_vs_f32m1_f32m1(value, accumulator, vl);
    return __riscv_vfmv_f_s_f32m1_f32(accumulator);
}

MLAS_FORCEINLINE
float
MlasReduceMaxRvv(
    vfloat32m1_t value,
    size_t vl
    )
{
    vfloat32m1_t accumulator =
        __riscv_vfmv_s_f_f32m1(std::numeric_limits<float>::lowest(), 1);
    accumulator = __riscv_vfredmax_vs_f32m1_f32m1(value, accumulator, vl);
    return __riscv_vfmv_f_s_f32m1_f32(accumulator);
}

}  // namespace

float
MLASCALL
MlasReduceMaximumF32KernelRvv(
    const float* Input,
    size_t N
    )
{
    float maximum = std::numeric_limits<float>::lowest();

    while (N > 0) {
        size_t vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t input = __riscv_vle32_v_f32m1(Input, vl);
        input = __riscv_vfmax_vf_f32m1(input, maximum, vl);
        maximum = MlasReduceMaxRvv(input, vl);

        Input += vl;
        N -= vl;
    }

    return maximum;
}

float
MLASCALL
MlasComputeSumExpF32KernelRvv(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
    )
{
    const float negative_maximum = *NegativeMaximum;
    float accumulation = 0.0f;

    while (N > 0) {
        size_t vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t input = __riscv_vle32_v_f32m1(Input, vl);
        vfloat32m1_t shifted = __riscv_vfadd_vf_f32m1(input, negative_maximum, vl);
        vfloat32m1_t exp_value = MlasComputeExpVectorRvv(shifted, vl);

        if (Output != nullptr) {
            __riscv_vse32_v_f32m1(Output, exp_value, vl);
            Output += vl;
        }

        accumulation += MlasReduceSumRvv(exp_value, vl);

        Input += vl;
        N -= vl;
    }

    return accumulation;
}

void
MLASCALL
MlasComputeSoftmaxOutputF32KernelRvv(
    float* Output,
    size_t N,
    const float* Parameters
    )
{
    const float scale = Parameters[0];

    while (N > 0) {
        size_t vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t output = __riscv_vle32_v_f32m1(Output, vl);
        output = __riscv_vfmul_vf_f32m1(output, scale, vl);
        __riscv_vse32_v_f32m1(Output, output, vl);

        Output += vl;
        N -= vl;
    }
}

void
MLASCALL
MlasComputeLogSoftmaxOutputF32KernelRvv(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
    )
{
    const float negative_maximum = Parameters[0];
    const float logarithm = Parameters[1];

    while (N > 0) {
        size_t vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t input = __riscv_vle32_v_f32m1(Input, vl);
        input = __riscv_vfadd_vf_f32m1(input, negative_maximum, vl);
        input = __riscv_vfsub_vf_f32m1(input, logarithm, vl);
        __riscv_vse32_v_f32m1(Output, input, vl);

        Input += vl;
        Output += vl;
        N -= vl;
    }
}

#endif  // defined(MLAS_USE_RVV)
