/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_kernel_neon_fp16.cpp

Abstract:

    This module implements the fp16 softmax kernels for ARM NEON.

--*/
#include <arm_neon.h>
#include <cassert>

#include "fp16_common.h"
#include "softmax.h"
#include "softmax_kernel_neon.h"

// TODO(fajin): intra-loop parallelism
namespace softmax_neon {

template <typename T>
struct MlasExpConstants {
    T LowerRange;
    T UpperRange;
    T LowerRangeSumExp;
    T UpperRangeSumExp;
    T RoundingBias;
    T Log2Reciprocal;
    T Log2High;
    T Log2Mid;
    T Log2Low;
    T poly_0;
    T poly_1;
    T poly_2;
    T poly_3;
    T poly_4;
    T poly_56;
    T MinimumExponent;
    T MaximumExponent;
};

const MlasExpConstants<_mlas_fp16_> ExpConstantsFp16 = {
    0xcc55, // -25 * ln2
    0x498c, // 16 * ln2
    0xc95f, // -15.5 * ln2
    0x495f, // 15.5 * ln2
    0x6600, // 1.5 * 2^10
    0x3dc5, // 1/ln2
    0xb98b, // -6.9287109375e-1f16
    0x8c85, // -2.758502960205078e-4f16
    0x8004, // -2.384185791015625e-7f16
    0x15b0, // 1/6!
    0x2044, // 1/5!
    0x2955, // 1/4!
    0x3155, // 1/3!
    0x3800, // 1/2!
    0x3c00, // 1/1!
    0xC800, // -14
    0x3C00, // 15
};

const MlasExpConstants<float16x8_t> ExpConstantsFp16x8 = {
    MlasBroadcastFloat16x8(ExpConstantsFp16.LowerRange),
    MlasBroadcastFloat16x8(ExpConstantsFp16.UpperRange),
    MlasBroadcastFloat16x8(ExpConstantsFp16.LowerRangeSumExp),
    MlasBroadcastFloat16x8(ExpConstantsFp16.UpperRangeSumExp),
    MlasBroadcastFloat16x8(ExpConstantsFp16.RoundingBias),
    MlasBroadcastFloat16x8(ExpConstantsFp16.Log2Reciprocal),
    MlasBroadcastFloat16x8(ExpConstantsFp16.Log2High),
    MlasBroadcastFloat16x8(ExpConstantsFp16.Log2Mid),
    MlasBroadcastFloat16x8(ExpConstantsFp16.Log2Low),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_0),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_1),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_2),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_3),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_4),
    MlasBroadcastFloat16x8(ExpConstantsFp16.poly_56),
    MlasBroadcastFloat16x8(ExpConstantsFp16.MinimumExponent),
    MlasBroadcastFloat16x8(ExpConstantsFp16.MaximumExponent),
};

const MlasExpConstants<float16x4_t> ExpConstantsFp16x4 = {
    MlasBroadcastFloat16x4(ExpConstantsFp16.LowerRange),
    MlasBroadcastFloat16x4(ExpConstantsFp16.UpperRange),
    MlasBroadcastFloat16x4(ExpConstantsFp16.LowerRangeSumExp),
    MlasBroadcastFloat16x4(ExpConstantsFp16.UpperRangeSumExp),
    MlasBroadcastFloat16x4(ExpConstantsFp16.RoundingBias),
    MlasBroadcastFloat16x4(ExpConstantsFp16.Log2Reciprocal),
    MlasBroadcastFloat16x4(ExpConstantsFp16.Log2High),
    MlasBroadcastFloat16x4(ExpConstantsFp16.Log2Mid),
    MlasBroadcastFloat16x4(ExpConstantsFp16.Log2Low),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_0),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_1),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_2),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_3),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_4),
    MlasBroadcastFloat16x4(ExpConstantsFp16.poly_56),
    MlasBroadcastFloat16x4(ExpConstantsFp16.MinimumExponent),
    MlasBroadcastFloat16x4(ExpConstantsFp16.MaximumExponent),
};

template <typename T>
MLAS_FORCEINLINE
MlasExpConstants<T> Get_Exp_Constants();

template <>
MLAS_FORCEINLINE
MlasExpConstants<float16x8_t> Get_Exp_Constants<float16x8_t>() {
    return ExpConstantsFp16x8;
}

template <>
MLAS_FORCEINLINE
MlasExpConstants<float16x4_t> Get_Exp_Constants<float16x4_t>() {
    return ExpConstantsFp16x4;
}

// Range reduction + polynomial approximation. Refer algorithm details to MlasComputeExpVector.
template<typename T>
MLAS_FORCEINLINE
T Exp_Vector_Fp16(T x) {
    const auto constants = Get_Exp_Constants<T>();
    auto clamped_x = MlasClampFloat16(x, constants.LowerRange, constants.UpperRange);

    // integral
    auto biased = MlasMultiplyAddFloat16(clamped_x, constants.Log2Reciprocal, constants.RoundingBias);
    auto m = MlasSubtractFloat16(biased, constants.RoundingBias);

    // residual
    auto r = MlasMultiplyAddFloat16(m, constants.Log2High, clamped_x);
    r = MlasMultiplyAddFloat16(m, constants.Log2Mid, r);
    r = MlasMultiplyAddFloat16(m, constants.Log2Low, r);

    // handle overflow
    auto overflow = MlasShiftLeftInt16<10>(MlasReinterpretFloat16AsInt16(biased));
    auto normal = overflow;

    auto minimum_exponent = MlasReinterpretFloat16AsInt16(constants.MinimumExponent);
    auto maximum_exponent = MlasReinterpretFloat16AsInt16(constants.MaximumExponent);
    normal = MlasClampInt16(normal, minimum_exponent, maximum_exponent);

    overflow = MlasSubtractInt16(overflow, normal);
    overflow = MlasAddInt16(overflow, maximum_exponent);
    normal = MlasAddInt16(normal, maximum_exponent);

    // polynomial approximation
    auto p = MlasMultiplyAddFloat16(constants.poly_0, r, constants.poly_1);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_2);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_3);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_4);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_56);

    auto overflow_f = MlasReinterpretInt16AsFloat16(overflow);
    r = MlasMultiplyFloat16(r, overflow_f);
    p = MlasMultiplyAddFloat16(p, r, overflow_f);
    p = MlasMultiplyFloat16(p, MlasReinterpretInt16AsFloat16(normal));

    return p;
}

void Exp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        auto r0 = Exp_Vector_Fp16(v0);
        auto r1 = Exp_Vector_Fp16(v1);
        auto r2 = Exp_Vector_Fp16(v2);
        auto r3 = Exp_Vector_Fp16(v3);

        MlasStoreFloat16x8(output, r0);
        MlasStoreFloat16x8(output + 8, r1);
        MlasStoreFloat16x8(output + 16, r2);
        MlasStoreFloat16x8(output + 24, r3);

        input += 32;
        output += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        auto r0 = Exp_Vector_Fp16(v0);
        auto r1 = Exp_Vector_Fp16(v1);

        MlasStoreFloat16x8(output, r0);
        MlasStoreFloat16x8(output + 8, r1);

        input += 16;
        output += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        auto r0 = Exp_Vector_Fp16(v0);
        MlasStoreFloat16x8(output, r0);

        input += 8;
        output += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        auto r0 = Exp_Vector_Fp16(v0);
        MlasStoreFloat16x4(output, r0);

        input += 4;
        output += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        auto r0 = Exp_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 3);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        auto r0 = Exp_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 2);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        auto r0 = Exp_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 1);
    }
}

// assume no overflow
template<typename T>
MLAS_FORCEINLINE
T SumExp_Vector_Fp16(T x, T negative_maximum) {
    const auto constants = Get_Exp_Constants<T>();
    auto clamped_x = MlasMaximumFloat16(MlasAddFloat16(x, negative_maximum), constants.LowerRangeSumExp);

    // integral
    auto biased = MlasMultiplyAddFloat16(clamped_x, constants.Log2Reciprocal, constants.RoundingBias);
    auto m = MlasSubtractFloat16(biased, constants.RoundingBias);

    // residual
    auto r = MlasMultiplyAddFloat16(m, constants.Log2High, clamped_x);
    r = MlasMultiplyAddFloat16(m, constants.Log2Mid, r);
    r = MlasMultiplyAddFloat16(m, constants.Log2Low, r);

    // 2^m
    auto normal = MlasShiftLeftInt16<10>(MlasReinterpretFloat16AsInt16(biased));
    normal = MlasAddInt16(normal, MlasReinterpretFloat16AsInt16(constants.MaximumExponent));

    // polynomial approximation
    auto p = MlasMultiplyAddFloat16(constants.poly_0, r, constants.poly_1);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_2);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_3);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_4);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_56);
    p = MlasMultiplyAddFloat16(p, r, constants.poly_56);

    p = MlasMultiplyFloat16(p, MlasReinterpretInt16AsFloat16(normal));

    return p;
}

MLAS_FORCEINLINE
float16x8_t AddUp(float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3, float16x8_t v4) {
    v0 = MlasAddFloat16(v0, v1);
    v2 = MlasAddFloat16(v2, v3);
    return MlasAddFloat16(MlasAddFloat16(v0, v2), v4);
}

MLAS_FORCEINLINE
float16x8_t AddUp(float16x8_t v0, float16x8_t v1, float16x8_t v2) {
    return MlasAddFloat16(MlasAddFloat16(v0, v1), v2);
}

MLAS_FP16 SumExp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    float16x8_t negative_maximum8 = MlasBroadcastFloat16x8(NegativeMaximum.val);
    float16x4_t negative_maximum4 = MlasBroadcastFloat16x4(NegativeMaximum.val);
    const bool store_output = Output != nullptr;
    float16x8_t accumulator8 = MlasZeroFloat16x8();
    float16x4_t accumulator4 = MlasZeroFloat16x4();

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum8);
        auto r1 = SumExp_Vector_Fp16(v1, negative_maximum8);
        auto r2 = SumExp_Vector_Fp16(v2, negative_maximum8);
        auto r3 = SumExp_Vector_Fp16(v3, negative_maximum8);

        accumulator8 = AddUp(r0, r1, r2, r3, accumulator8);

        if (store_output) {
            MlasStoreFloat16x8(output, r0);
            MlasStoreFloat16x8(output + 8, r1);
            MlasStoreFloat16x8(output + 16, r2);
            MlasStoreFloat16x8(output + 24, r3);
            output += 32;
        }

        input += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum8);
        auto r1 = SumExp_Vector_Fp16(v1, negative_maximum8);

        accumulator8 = AddUp(r0, r1, accumulator8);

        if (store_output) {
            MlasStoreFloat16x8(output, r0);
            MlasStoreFloat16x8(output + 8, r1);
            output += 16;
        }

        input += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum8);
        accumulator8 = MlasAddFloat16(r0, accumulator8);

        if (store_output) {
            MlasStoreFloat16x8(output, r0);
            output += 8;
        }

        input += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);
        accumulator4 = MlasAddFloat16(r0, accumulator4);

        if (store_output) {
            MlasStoreFloat16x4(output, r0);
            output += 4;
        }

        input += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);

        if (store_output) {
            MlasStorePartialFloat16x4(output, r0, 3);
        }

        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 3));
        accumulator4 = MlasAddFloat16(r0, accumulator4);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);

        if (store_output) {
            MlasStorePartialFloat16x4(output, r0, 2);
        }

        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 3));
        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 2));
        accumulator4 = MlasAddFloat16(r0, accumulator4);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);

        if (store_output) {
            MlasStorePartialFloat16x4(output, r0, 1);
        }

        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 3));
        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 2));
        r0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0), MlasReinterpretFloat16AsInt16(r0), 1));
        accumulator4 = MlasAddFloat16(r0, accumulator4);
    }

    auto t = MlasAddFloat16(vget_low_f16(accumulator8), vget_high_f16(accumulator8));
    t = MlasAddFloat16(t, accumulator4);
    _mlas_fp16_ result = MlasReduceAddFloat16(t);
    return MLAS_FP16::FromBits(result);
}

template <typename T>
struct MlasTanhConstants {
    T LowerRange;
    T UpperRange;
    T alpha_7;
    T alpha_5;
    T alpha_3;
    T alpha_1;
    T beta_6;
    T beta_4;
    T beta_2;
    T beta_0;
};

const MlasTanhConstants<_mlas_fp16_> TanhConstantsFp16 = {
    0xc308, // -3.51562
    0x4308, // 3.51562
    0x0001,
    0x00f9,
    0x1138,
    0x1d03,
    0x0014,
    0x07c5,
    0x18a5,
    0x1d03,
};

const MlasTanhConstants<float16x8_t> TanhConstantsFp16x8 = {
    MlasBroadcastFloat16x8(TanhConstantsFp16.LowerRange),
    MlasBroadcastFloat16x8(TanhConstantsFp16.UpperRange),
    MlasBroadcastFloat16x8(TanhConstantsFp16.alpha_7),
    MlasBroadcastFloat16x8(TanhConstantsFp16.alpha_5),
    MlasBroadcastFloat16x8(TanhConstantsFp16.alpha_3),
    MlasBroadcastFloat16x8(TanhConstantsFp16.alpha_1),
    MlasBroadcastFloat16x8(TanhConstantsFp16.beta_6),
    MlasBroadcastFloat16x8(TanhConstantsFp16.beta_4),
    MlasBroadcastFloat16x8(TanhConstantsFp16.beta_2),
    MlasBroadcastFloat16x8(TanhConstantsFp16.beta_0),
};

const MlasTanhConstants<float16x4_t> TanhConstantsFp16x4 = {
    MlasBroadcastFloat16x4(TanhConstantsFp16.LowerRange),
    MlasBroadcastFloat16x4(TanhConstantsFp16.UpperRange),
    MlasBroadcastFloat16x4(TanhConstantsFp16.alpha_7),
    MlasBroadcastFloat16x4(TanhConstantsFp16.alpha_5),
    MlasBroadcastFloat16x4(TanhConstantsFp16.alpha_3),
    MlasBroadcastFloat16x4(TanhConstantsFp16.alpha_1),
    MlasBroadcastFloat16x4(TanhConstantsFp16.beta_6),
    MlasBroadcastFloat16x4(TanhConstantsFp16.beta_4),
    MlasBroadcastFloat16x4(TanhConstantsFp16.beta_2),
    MlasBroadcastFloat16x4(TanhConstantsFp16.beta_0),
};

template <typename T>
MLAS_FORCEINLINE
MlasTanhConstants<T> Get_Tanh_Constants();

template <>
MLAS_FORCEINLINE
MlasTanhConstants<float16x8_t> Get_Tanh_Constants<float16x8_t>() {
    return TanhConstantsFp16x8;
}

template <>
MLAS_FORCEINLINE
MlasTanhConstants<float16x4_t> Get_Tanh_Constants<float16x4_t>() {
    return TanhConstantsFp16x4;
}

// TODO(fajin): optimize polynomial coefficients
template <typename T>
MLAS_FORCEINLINE
T Tanh_Vector_Fp16(T x) {
    const auto constants = Get_Tanh_Constants<T>();
    x = MlasClampFloat16(x, constants.LowerRange, constants.UpperRange);

    T x_2 = MlasMultiplyFloat16(x, x);

    T p = MlasMultiplyAddFloat16(constants.alpha_7, x_2, constants.alpha_5);
    p = MlasMultiplyAddFloat16(p, x_2, constants.alpha_3);
    p = MlasMultiplyAddFloat16(p, x_2, constants.alpha_1);
    p = MlasMultiplyFloat16(p, x);

    T q = MlasMultiplyAddFloat16(constants.beta_6, x_2, constants.beta_4);
    q = MlasMultiplyAddFloat16(q, x_2, constants.beta_2);
    q = MlasMultiplyAddFloat16(q, x_2, constants.beta_0);

    return MlasDivideFloat16(p, q);
}

void Tanh_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        auto r0 = Tanh_Vector_Fp16(v0);
        auto r1 = Tanh_Vector_Fp16(v1);
        auto r2 = Tanh_Vector_Fp16(v2);
        auto r3 = Tanh_Vector_Fp16(v3);

        MlasStoreFloat16x8(output, r0);
        MlasStoreFloat16x8(output + 8, r1);
        MlasStoreFloat16x8(output + 16, r2);
        MlasStoreFloat16x8(output + 24, r3);

        input += 32;
        output += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        auto r0 = Tanh_Vector_Fp16(v0);
        auto r1 = Tanh_Vector_Fp16(v1);

        MlasStoreFloat16x8(output, r0);
        MlasStoreFloat16x8(output + 8, r1);

        input += 16;
        output += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        auto r0 = Tanh_Vector_Fp16(v0);
        MlasStoreFloat16x8(output, r0);

        input += 8;
        output += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        auto r0 = Tanh_Vector_Fp16(v0);
        MlasStoreFloat16x4(output, r0);

        input += 4;
        output += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        auto r0 = Tanh_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 3);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        auto r0 = Tanh_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 2);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        auto r0 = Tanh_Vector_Fp16(v0);
        MlasStorePartialFloat16x4(output, r0, 1);
    }
}

void Softcap_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Softcap) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    auto softcap8 = MlasBroadcastFloat16x8(Softcap.val);
    auto softcap4 = MlasBroadcastFloat16x4(Softcap.val);
    auto one8 = MlasBroadcastFloat16x8((_mlas_fp16_)0x3c00);
    auto one4 = MlasBroadcastFloat16x4((_mlas_fp16_)0x3c00);
    auto softcap_reciprocal8 = MlasDivideFloat16(one8, softcap8);
    auto softcap_reciprocal4 = MlasDivideFloat16(one4, softcap4);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal8);
        v1 = MlasMultiplyFloat16(v1, softcap_reciprocal8);
        v2 = MlasMultiplyFloat16(v2, softcap_reciprocal8);
        v3 = MlasMultiplyFloat16(v3, softcap_reciprocal8);

        v0 = Tanh_Vector_Fp16(v0);
        v1 = Tanh_Vector_Fp16(v1);
        v2 = Tanh_Vector_Fp16(v2);
        v3 = Tanh_Vector_Fp16(v3);

        v0 = MlasMultiplyFloat16(v0, softcap8);
        v1 = MlasMultiplyFloat16(v1, softcap8);
        v2 = MlasMultiplyFloat16(v2, softcap8);
        v3 = MlasMultiplyFloat16(v3, softcap8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);
        MlasStoreFloat16x8(output + 16, v2);
        MlasStoreFloat16x8(output + 24, v3);

        input += 32;
        output += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal8);
        v1 = MlasMultiplyFloat16(v1, softcap_reciprocal8);

        v0 = Tanh_Vector_Fp16(v0);
        v1 = Tanh_Vector_Fp16(v1);

        v0 = MlasMultiplyFloat16(v0, softcap8);
        v1 = MlasMultiplyFloat16(v1, softcap8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);

        input += 16;
        output += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal8);
        v0 = Tanh_Vector_Fp16(v0);
        v0 = MlasMultiplyFloat16(v0, softcap8);
        MlasStoreFloat16x8(output, v0);

        input += 8;
        output += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal4);
        v0 = Tanh_Vector_Fp16(v0);
        v0 = MlasMultiplyFloat16(v0, softcap4);
        MlasStoreFloat16x4(output, v0);

        input += 4;
        output += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal4);
        v0 = Tanh_Vector_Fp16(v0);
        v0 = MlasMultiplyFloat16(v0, softcap4);
        MlasStorePartialFloat16x4(output, v0, 3);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal4);
        v0 = Tanh_Vector_Fp16(v0);
        v0 = MlasMultiplyFloat16(v0, softcap4);
        MlasStorePartialFloat16x4(output, v0, 2);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        v0 = MlasMultiplyFloat16(v0, softcap_reciprocal4);
        v0 = Tanh_Vector_Fp16(v0);
        v0 = MlasMultiplyFloat16(v0, softcap4);
        MlasStorePartialFloat16x4(output, v0, 1);
    }
}

MLAS_FP16 ReduceMax_Kernel_Fp16(const MLAS_FP16* Input, size_t N) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto max8 = MlasBroadcastFloat16x8((_mlas_fp16_)0xfbff);
    auto max4 = MlasBroadcastFloat16x4((_mlas_fp16_)0xfbff);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        v0 = MlasMaximumFloat16(v0, v1);
        v2 = MlasMaximumFloat16(v2, v3);
        v0 = MlasMaximumFloat16(v0, v2);
        max8 = MlasMaximumFloat16(max8, v0);

        input += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        v0 = MlasMaximumFloat16(v0, v1);
        max8 = MlasMaximumFloat16(max8, v0);

        input += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        max8 = MlasMaximumFloat16(max8, v0);

        input += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        max4 = MlasMaximumFloat16(max4, v0);

        input += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 3));
        max4 = MlasMaximumFloat16(max4, v0);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 3));
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 2));
        max4 = MlasMaximumFloat16(max4, v0);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 3));
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 2));
        v0 = MlasReinterpretInt16AsFloat16(vset_lane_s16(static_cast<int16_t>(0xfbff), MlasReinterpretFloat16AsInt16(v0), 1));
        max4 = MlasMaximumFloat16(max4, v0);
    }

    auto t = MlasMaximumFloat16(vget_low_f16(max8), vget_high_f16(max8));
    t = MlasMaximumFloat16(t, max4);
    _mlas_fp16_ result = MlasReduceMaximumFloat16(t);

    return MLAS_FP16::FromBits(result);
}

void Softmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Sum) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    auto sum8 = MlasBroadcastFloat16x8(Sum.val);
    auto sum4 = MlasBroadcastFloat16x4(Sum.val);
    auto scale8 = MlasDivideFloat16(MlasBroadcastFloat16x8((_mlas_fp16_)0x3c00), sum8);
    auto scale4 = MlasDivideFloat16(MlasBroadcastFloat16x4((_mlas_fp16_)0x3c00), sum4);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        v0 = MlasMultiplyFloat16(v0, scale8);
        v1 = MlasMultiplyFloat16(v1, scale8);
        v2 = MlasMultiplyFloat16(v2, scale8);
        v3 = MlasMultiplyFloat16(v3, scale8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);
        MlasStoreFloat16x8(output + 16, v2);
        MlasStoreFloat16x8(output + 24, v3);

        input += 32;
        output += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        v0 = MlasMultiplyFloat16(v0, scale8);
        v1 = MlasMultiplyFloat16(v1, scale8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);

        input += 16;
        output += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        v0 = MlasMultiplyFloat16(v0, scale8);
        MlasStoreFloat16x8(output, v0);

        input += 8;
        output += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        v0 = MlasMultiplyFloat16(v0, scale4);
        MlasStoreFloat16x4(output, v0);

        input += 4;
        output += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        v0 = MlasMultiplyFloat16(v0, scale4);
        MlasStorePartialFloat16x4(output, v0, 3);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        v0 = MlasMultiplyFloat16(v0, scale4);
        MlasStorePartialFloat16x4(output, v0, 2);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        v0 = MlasMultiplyFloat16(v0, scale4);
        MlasStorePartialFloat16x4(output, v0, 1);
    }
}

void LogSoftmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    auto negative_maximum8 = MlasBroadcastFloat16x8(NegativeMaximum.val);
    auto negative_maximum4 = MlasBroadcastFloat16x4(NegativeMaximum.val);
    auto log_sum8 = MlasBroadcastFloat16x8(LogSum.val);
    auto log_sum4 = MlasBroadcastFloat16x4(LogSum.val);

    while (N >= 32) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);
        auto v2 = MlasLoadFloat16x8(input + 16);
        auto v3 = MlasLoadFloat16x8(input + 24);

        v0 = MlasAddFloat16(v0, negative_maximum8);
        v1 = MlasAddFloat16(v1, negative_maximum8);
        v2 = MlasAddFloat16(v2, negative_maximum8);
        v3 = MlasAddFloat16(v3, negative_maximum8);

        v0 = MlasSubtractFloat16(v0, log_sum8);
        v1 = MlasSubtractFloat16(v1, log_sum8);
        v2 = MlasSubtractFloat16(v2, log_sum8);
        v3 = MlasSubtractFloat16(v3, log_sum8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);
        MlasStoreFloat16x8(output + 16, v2);
        MlasStoreFloat16x8(output + 24, v3);

        input += 32;
        output += 32;
        N -= 32;
    }

    if (N & 16) {
        auto v0 = MlasLoadFloat16x8(input);
        auto v1 = MlasLoadFloat16x8(input + 8);

        v0 = MlasAddFloat16(v0, negative_maximum8);
        v1 = MlasAddFloat16(v1, negative_maximum8);

        v0 = MlasSubtractFloat16(v0, log_sum8);
        v1 = MlasSubtractFloat16(v1, log_sum8);

        MlasStoreFloat16x8(output, v0);
        MlasStoreFloat16x8(output + 8, v1);

        input += 16;
        output += 16;
        N -= 16;
    }

    if (N & 8) {
        auto v0 = MlasLoadFloat16x8(input);
        v0 = MlasAddFloat16(v0, negative_maximum8);
        v0 = MlasSubtractFloat16(v0, log_sum8);
        MlasStoreFloat16x8(output, v0);

        input += 8;
        output += 8;
        N -= 8;
    }

    if (N & 4) {
        auto v0 = MlasLoadFloat16x4(input);
        v0 = MlasAddFloat16(v0, negative_maximum4);
        v0 = MlasSubtractFloat16(v0, log_sum4);
        MlasStoreFloat16x4(output, v0);

        input += 4;
        output += 4;
        N -= 4;
    }

    if (N == 3) {
        auto v0 = MlasLoadPartialFloat16x4(input, 3);
        v0 = MlasAddFloat16(v0, negative_maximum4);
        v0 = MlasSubtractFloat16(v0, log_sum4);
        MlasStorePartialFloat16x4(output, v0, 3);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        v0 = MlasAddFloat16(v0, negative_maximum4);
        v0 = MlasSubtractFloat16(v0, log_sum4);
        MlasStorePartialFloat16x4(output, v0, 2);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        v0 = MlasAddFloat16(v0, negative_maximum4);
        v0 = MlasSubtractFloat16(v0, log_sum4);
        MlasStorePartialFloat16x4(output, v0, 1);
    }
}

}  // namespace rope_neon
