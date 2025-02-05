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
    auto clamped_x = MlasClamp(x, constants.LowerRange, constants.UpperRange);

    // integral
    auto biased = MlasMultiplyAdd(clamped_x, constants.Log2Reciprocal, constants.RoundingBias);
    auto m = MlasSubtract(biased, constants.RoundingBias);

    // residual
    auto r = MlasMultiplyAdd(m, constants.Log2High, clamped_x);
    r = MlasMultiplyAdd(m, constants.Log2Mid, r);
    r = MlasMultiplyAdd(m, constants.Log2Low, r);

    // handle overflow
    auto overflow = MlasShiftLeft<10>(MlasReinterpretAsInt16(biased));
    auto normal = overflow;

    auto minimum_exponent = MlasReinterpretAsInt16(constants.MinimumExponent);
    auto maximum_exponent = MlasReinterpretAsInt16(constants.MaximumExponent);
    normal = MlasClamp(normal, minimum_exponent, maximum_exponent);

    overflow = MlasSubtract(overflow, normal);
    overflow = MlasAdd(overflow, maximum_exponent);
    normal = MlasAdd(normal, maximum_exponent);

    // polynomial approximation
    auto p = MlasMultiplyAdd(constants.poly_0, r, constants.poly_1);
    p = MlasMultiplyAdd(p, r, constants.poly_2);
    p = MlasMultiplyAdd(p, r, constants.poly_3);
    p = MlasMultiplyAdd(p, r, constants.poly_4);
    p = MlasMultiplyAdd(p, r, constants.poly_56);

    auto overflow_f = MlasReinterpretAsFloat16(overflow);
    r = MlasMultiply(r, overflow_f);
    p = MlasMultiplyAdd(p, r, overflow_f);
    p = MlasMultiply(p, MlasReinterpretAsFloat16(normal));

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
    auto clamped_x = MlasMaximum(MlasAdd(x, negative_maximum), constants.LowerRangeSumExp);

    // integral
    auto biased = MlasMultiplyAdd(clamped_x, constants.Log2Reciprocal, constants.RoundingBias);
    auto m = MlasSubtract(biased, constants.RoundingBias);

    // residual
    auto r = MlasMultiplyAdd(m, constants.Log2High, clamped_x);
    r = MlasMultiplyAdd(m, constants.Log2Mid, r);
    r = MlasMultiplyAdd(m, constants.Log2Low, r);

    // 2^m
    auto normal = MlasShiftLeft<10>(MlasReinterpretAsInt16(biased));
    normal = MlasAdd(normal, MlasReinterpretAsInt16(constants.MaximumExponent));

    // polynomial approximation
    auto p = MlasMultiplyAdd(constants.poly_0, r, constants.poly_1);
    p = MlasMultiplyAdd(p, r, constants.poly_2);
    p = MlasMultiplyAdd(p, r, constants.poly_3);
    p = MlasMultiplyAdd(p, r, constants.poly_4);
    p = MlasMultiplyAdd(p, r, constants.poly_56);
    p = MlasMultiplyAdd(p, r, constants.poly_56);

    p = MlasMultiply(p, MlasReinterpretAsFloat16(normal));

    return p;
}

MLAS_FORCEINLINE
float16x8_t AddUp(float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3, float16x8_t v4) {
    v0 = MlasAdd(v0, v1);
    v2 = MlasAdd(v2, v3);
    return MlasAdd(MlasAdd(v0, v2), v4);
}

MLAS_FORCEINLINE
float16x8_t AddUp(float16x8_t v0, float16x8_t v1, float16x8_t v2) {
    return MlasAdd(MlasAdd(v0, v1), v2);
}

MLAS_FP16 SumExp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum) {
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    float16x8_t negative_maximum8 = MlasBroadcastFloat16x8(NegativeMaximum.val);
    float16x4_t negative_maximum4 = MlasBroadcastFloat16x4(NegativeMaximum.val);
    const bool store_output = Output != nullptr;
    float16x8_t accumulator8 = MlasZeroFloat16x8();
    float16x4_t accumulator4 = MlasZeroFloat16x4();
    _mlas_fp16_ buffer[4] = {0};

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
        accumulator8 = MlasAdd(r0, accumulator8);

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
        accumulator4 = MlasAdd(r0, accumulator4);

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

        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 3));
        accumulator4 = MlasAdd(r0, accumulator4);
    } else if (N == 2) {
        auto v0 = MlasLoadPartialFloat16x4(input, 2);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);

        if (store_output) {
            MlasStorePartialFloat16x4(output, r0, 2);
        }

        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 3));
        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 2));
        accumulator4 = MlasAdd(r0, accumulator4);
    } else if (N == 1) {
        auto v0 = MlasLoadPartialFloat16x4(input, 1);
        auto r0 = SumExp_Vector_Fp16(v0, negative_maximum4);

        if (store_output) {
            MlasStorePartialFloat16x4(output, r0, 1);
        }

        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 3));
        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 2));
        r0 = MlasReinterpretAsFloat16(vset_lane_s16(0, MlasReinterpretAsInt16(r0), 1));
        accumulator4 = MlasAdd(r0, accumulator4);
    }

    auto t = MlasAdd(vget_low_f16(accumulator8), vget_high_f16(accumulator8));
    t = MlasAdd(t, accumulator4);
    _mlas_fp16_ result = MlasReduceAdd(t);
    return MLAS_FP16::FromBits(result);
}

const struct {
    _mlas_fp16_ LowerRange;
    _mlas_fp16_ UpperRange;
    _mlas_fp16_ alpha_9;
    _mlas_fp16_ alpha_7;
    _mlas_fp16_ alpha_5;
    _mlas_fp16_ alpha_3;
    _mlas_fp16_ alpha_1;
    _mlas_fp16_ beta_10;
    _mlas_fp16_ beta_8;
    _mlas_fp16_ beta_6;
    _mlas_fp16_ beta_4;
    _mlas_fp16_ beta_2;
    _mlas_fp16_ beta_0;
} MlasTanh16Constants = {
    0xc500, // -5.0f16
    0x4500, // 5.0f16
    0x002e, // 1/9!
    0x0a80, // 1/7!
    0x2044, // 1/5!
    0x3155, // 1/3!
    0x3c00, // 1
    0x0005, // 1/10!
    0x01a0, // 1/8!
    0x15b0, // 1/6!
    0x2955, // 1/4!
    0x3800, // 1/2!
    0x3c00, // 1
};

//  _Float16 my_tanh(_Float16 Value) {
//     _Float16 v_tmp;
//     v_tmp = (Value < MlasTanh16Constants.LowerRange) ? MlasTanh16Constants.LowerRange : Value;
//     Value = (v_tmp > MlasTanh16Constants.UpperRange) ? MlasTanh16Constants.UpperRange : v_tmp;

//     _Float16 ValueSquared = Value * Value;

//     _Float16 p = MlasTanh16Constants.alpha_9;
//     p = p * ValueSquared + MlasTanh16Constants.alpha_7;
//     p = p * ValueSquared + MlasTanh16Constants.alpha_5;
//     p = p * ValueSquared + MlasTanh16Constants.alpha_3;
//     p = p * ValueSquared + MlasTanh16Constants.alpha_1;
//     p = p * Value;

//     _Float16 q = MlasTanh16Constants.beta_10;
//     q = q * ValueSquared + MlasTanh16Constants.beta_8;
//     q = q * ValueSquared + MlasTanh16Constants.beta_6;
//     q = q * ValueSquared + MlasTanh16Constants.beta_4;
//     q = q * ValueSquared + MlasTanh16Constants.beta_2;
//     q = q * ValueSquared + MlasTanh16Constants.beta_0;

//     return (p / q);
//   }


//   _Float16 my_tanh_no_overflow(_Float16 Value) {
//     if (Value > 0.5f16) {
//       _Float16 exp = my_exp(Value);
//       return (exp - 1.0f16/exp) / (exp + 1.0f16/exp);
//     } else {
//       return my_tanh(Value);
//     }
//   }

void Tanh_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N) {

}

void Softcap_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Softcap) {

}

MLAS_FP16 ReduceMax_Kernel_Fp16(const MLAS_FP16* Input, size_t N) {
    return MLAS_FP16::FromBits(0);
}

void Softmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 scale) {

}

void LogSoftmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum) {
}

}  // namespace rope_neon
