/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

    elementwise_sve.cpp

Abstract:

    This module contains the implementation of SVE-based elementwise operations
--*/

#include "mlasi_sve.h"
#include <limits>

//
// Bundles the constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const struct {
    float ErfUpperAbsRange;
    float ErfSplitBoundary;
    float ErfSMALL_P0;
    float ErfSMALL_P1;
    float ErfSMALL_P2;
    float ErfSMALL_P3;
    float ErfSMALL_P4;
    float ErfSMALL_P5_Minus_One;
    float ErfReserved0;
    float ErfBIG_P0;
    float ErfBIG_P1;
    float ErfBIG_P2;
    float ErfBIG_P3;
    float ErfBIG_P4;
    float ErfBIG_P5;
    float ErfBIG_P6_Minus_One;
    float ErfNegZero;
    float ErfOne;

    float Exp_UpperRange;
    float Exp_LowerRange;
    float Exp_Log2Reciprocal;
    float Exp_log2_hi;
    float Exp_log2_lo;
    float Exp_P0;
    float Exp_P1;
    float Exp_P2;
    float Exp_P3;
    float Exp_P4;
    float Exp_P5;
    float Exp_P6;
    float Exp_C;
    int32_t Exp_X7F;
} MlasSveErfConstants = {
    3.925f,
    0.921875f,
    -5.99104969e-4f,
    4.99339588e-3f,
    -2.67667342e-2f,
    1.12818025e-1f,
    -3.76124859e-1f,
    1.28379151e-1f,
    0.0f,
    1.72948930e-5f,
    -3.83208680e-4f,
    3.88393435e-3f,
    -2.42545605e-2f,
    1.06777847e-1f,
    6.34846687e-1f,
    1.28717512e-1f,
    -0.0f,
    1.0f,

    // Independent parameters to calculate Exp for Erff()
    88.3762626647950f,
    -88.3762626647949f,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    1.38319808e-3f,
    8.37550033e-3f,
    4.16689515e-2f,
    1.66664466e-1f,
    4.99999851e-1f,
    1.00000000e+0f,
    1.00000000e+0f,
    1.25829120e+7f,
    127,
};

MLAS_INTERNAL_DATA const struct {
    float LowerRange;
    float UpperRange;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
} MlasSveLogisticConstants = {
    -18.0f,
    18.0f,
    4.37031012579801e-11f,
    1.15627324459942e-07f,
    6.08574864600143e-05f,
    8.51377133304701e-03f,
    2.48287947061529e-01f,
    6.10247389755681e-13f,
    5.76102136993427e-09f,
    6.29106785017040e-06f,
    1.70198817374094e-03f,
    1.16817656904453e-01f,
    9.93151921023180e-01f,
    0.5f,
};

MLAS_INTERNAL_DATA const struct {
    float LowerRange;
    float UpperRange;
    float LowerRangeSumExp;
    float UpperRangeSumExp;
    float RoundingBias;
    float Log2Reciprocal;
    float Log2High;
    float Log2Low;
    float poly_0;
    float poly_1;
    float poly_2;
    float poly_3;
    float poly_4;
    float poly_56;
    int32_t MinimumExponent;
    int32_t MaximumExponent;
} MlasSveExpConstants = {
    -103.9720840454f,
    88.7762626647950f,
    -88.3762626647949f,
    88.3762626647949f,
    MLAS_ROUNDING_BIAS_MAGIC,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    0x1.694000p-10,
    0x1.125edcp-7,
    0x1.555b5ap-5,
    0x1.555450p-3,
    0x1.fffff6p-2,
    0x1.000000p+0,
    int32_t(0xC1000000),
    int32_t(0x3F800000),
};

MLAS_INTERNAL_DATA const float MlasSveMinimumF32Value = std::numeric_limits<float>::lowest();

void
MLASCALL
MlasSveErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t sve_veclen = svcntw();
    size_t stride = sve_veclen;

    while (N > 0) {
        // If fewer that SVE vector length elements are remaining, adjust the predicate
        if (N < sve_veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Value = MlasSveLoadFloat32(Pred, Input);
        MLAS_SVFLOAT32 NegZero = MlasSveBroadcastFloat32(MlasSveErfConstants.ErfNegZero);
        MLAS_SVFLOAT32 SignMask = MlasSveAndFloat32(Pred, Value, NegZero);
        MLAS_SVFLOAT32 AbsValue = MlasSveAndNotFloat32(Pred, NegZero, Value);
        AbsValue = MlasSveMinimumFloat32(Pred, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfUpperAbsRange), AbsValue);
        MLAS_SVFLOAT32 SquareValue = MlasSveMultiplyFloat32(Pred, AbsValue, AbsValue);

        MLAS_SVFLOAT32 r_small = MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P0);
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, SquareValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P1));
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, SquareValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P2));
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, SquareValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P3));
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, SquareValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P4));
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, SquareValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSMALL_P5_Minus_One));
        r_small = MlasSveMultiplyAddFloat32(Pred, r_small, AbsValue, AbsValue);
        MLAS_SVFLOAT32 split_mask = MlasSveGreaterThanFloat32(Pred, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfSplitBoundary));
        r_small = MlasSveAndNotFloat32(Pred, split_mask, r_small);

        AbsValue = MlasSveAndFloat32(Pred, split_mask, AbsValue);
        MLAS_SVFLOAT32 r_big = MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P0);
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P1));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P2));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P3));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P4));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P5));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfBIG_P6_Minus_One));
        r_big = MlasSveMultiplyAddFloat32(Pred, r_big, AbsValue, AbsValue);

        r_big = MlasSveXorFloat32(Pred, r_big, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfNegZero));
        r_big = MlasSveMaximumFloat32(Pred, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_LowerRange), r_big);
        MLAS_SVFLOAT32 exp_c = MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_C);
        MLAS_SVFLOAT32 r = MlasSveMultiplyAddFloat32(Pred, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_Log2Reciprocal), r_big, exp_c);
        r = MlasSveSubtractFloat32(Pred, r, exp_c);

        MLAS_SVFLOAT32 fx = MlasSveMultiplyAddFloat32(Pred, r, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_log2_hi), r_big);
        fx = MlasSveMultiplyAddFloat32(Pred, r, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_log2_lo), fx);

        MLAS_SVFLOAT32 y = MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P0);
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P1));
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P2));
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P3));
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P4));
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P5));
        y = MlasSveMultiplyAddFloat32(Pred, y, fx, MlasSveBroadcastFloat32(MlasSveErfConstants.Exp_P6));

        y = MlasSveMultiplyFloat32(Pred, y, MlasSvePowerOf2Float32(Pred, r));
        y = MlasSveSubtractFloat32(Pred, MlasSveBroadcastFloat32(MlasSveErfConstants.ErfOne), y);

        y = MlasSveOrFloat32(Pred, r_small, y);
        y = MlasSveOrFloat32(Pred, y, SignMask);
        MlasSveStoreFloat32(Pred, Output, y);

        Input += stride;
        Output += stride;
        N -= stride;
    }
}

void 
MLASCALL
MlasSveLogisticKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the logistic function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t sve_veclen = svcntw();
    size_t stride = sve_veclen;

    while (N > 0) {
        // If fewer that SVE vector length elements are remaining, adjust the predicate
        if (N < sve_veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Value = MlasSveLoadFloat32(Pred, Input);

        Value = MlasSveMaximumFloat32(Pred, MlasSveBroadcastFloat32(MlasSveLogisticConstants.LowerRange), Value);
        Value = MlasSveMinimumFloat32(Pred, MlasSveBroadcastFloat32(MlasSveLogisticConstants.UpperRange), Value);

        MLAS_SVFLOAT32 ValueSquared = MlasSveMultiplyFloat32(Pred, Value, Value);
        
        MLAS_SVFLOAT32 p;
        p = MlasSveMultiplyAddFloat32(
            Pred, 
            ValueSquared, 
            MlasSveBroadcastFloat32(MlasSveLogisticConstants.alpha_9),
            MlasSveBroadcastFloat32(MlasSveLogisticConstants.alpha_7)
        );
        p = MlasSveMultiplyAddFloat32(Pred, p, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.alpha_5));
        p = MlasSveMultiplyAddFloat32(Pred, p, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.alpha_3));
        p = MlasSveMultiplyAddFloat32(Pred, p, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.alpha_1));
        p = MlasSveMultiplyFloat32(Pred, p, Value);

        MLAS_SVFLOAT32 q;
        q = MlasSveMultiplyAddFloat32(
            Pred,
            ValueSquared,
            MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_10),
            MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_8)
        );
        q = MlasSveMultiplyAddFloat32(Pred, q, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_6));
        q = MlasSveMultiplyAddFloat32(Pred, q, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_4));
        q = MlasSveMultiplyAddFloat32(Pred, q, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_2));
        q = MlasSveMultiplyAddFloat32(Pred, q, ValueSquared, MlasSveBroadcastFloat32(MlasSveLogisticConstants.beta_0));

        MlasSveStoreFloat32(
            Pred, 
            Output,
            MlasSveClampFloat32(Pred,MlasSveAddFloat32(
                Pred, 
                MlasSveDivideFloat32(Pred, p, q),
                MlasSveBroadcastFloat32(0.5f)
            ), 0.0f, 1.0f)
        ); 

        Input += stride;
        Output += stride;
        N -= stride;
    }
}

/*
SVE implementation of expf() using a polynomial approximation is taken from ARM Compute Library Repository.
https://github.com/ARM-software/ComputeLibrary/blob/9f7a1fb06bc0435d989a9a6a3c0fd2cebfedbf5f/src/core/NEON/SVEMath.inl#L105
*/
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector
)
{
    const uint32_t svexp_f32_coeff[] = {
        0x3f7ffff6, // x^1: 0x1.ffffecp-1f
        0x3efffedb, // x^2: 0x1.fffdb6p-2f
        0x3e2aaf33, // x^3: 0x1.555e66p-3f
        0x3d2b9f17, // x^4: 0x1.573e2ep-5f
        0x3c072010, // x^5: 0x1.0e4020p-7f
    };

    const auto c1 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(svexp_f32_coeff[0]));
    const auto c2 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(svexp_f32_coeff[1]));
    const auto c3 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(svexp_f32_coeff[2]));
    const auto c4 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(svexp_f32_coeff[3]));
    const auto c5 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(svexp_f32_coeff[4]));

    const auto shift   = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi =
        MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo =
        MlasSveReinterpretAsFLOAT32(MlasSveBroadcastUINT32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = MlasSveBroadcastFloat32(std::numeric_limits<float>::infinity());
    const auto max_input = MlasSveBroadcastFloat32(88.37f); // Approximately ln(2^127.5)
    const auto zero      = MlasSveZeroFloat32();
    const auto min_input = MlasSveBroadcastFloat32(-86.64f); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127 forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n) + 127 will occupy
    //     the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part of x / ln(2)
    //     (i.e. n) because the decimal part has been pushed out and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    const auto z     = MlasSveMultiplyAddFloat32(Pred, Vector, inv_ln2, shift);
    const auto n     = MlasSveSubtractFloat32(Pred, z, shift);
    const auto scale = MlasSveReinterpretAsFLOAT32(MlasSveShiftLeftUInt32<23>(Pred, MlasSveReinterpretAsUInt32(z))); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in term of accuracy and performance.
    const auto r_hi = MlasSveMultiplyAddFloat32(Pred, n, neg_ln2_hi, Vector);
    const auto r    = MlasSveMultiplyAddFloat32(Pred, n, neg_ln2_lo, r_hi);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = MlasSveMultiplyFloat32(Pred, r, r);

    const auto p1     = MlasSveMultiplyFloat32(Pred, c1, r);
    const auto p23    = MlasSveMultiplyAddFloat32(Pred, c3, r, c2);
    const auto p45    = MlasSveMultiplyAddFloat32(Pred, c5, r, c4);
    const auto p2345  = MlasSveMultiplyAddFloat32(Pred, p45, r2, p23);
    const auto p12345 = MlasSveMultiplyAddFloat32(Pred, p2345, r2, p1);

    auto poly = MlasSveMultiplyAddFloat32(Pred, p12345, scale, scale);

    // Handle underflow and overflow.
    poly = MlasSveSelect(MlasSveCompareLessThan(Pred, Vector, min_input), zero, poly);
    poly = MlasSveSelect(MlasSveCompareGreaterThan(Pred, Vector, max_input), inf, poly);

    return poly;
}

void
MLASCALL
MlasSveComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
)
{
    const size_t veclen = svcntw();

    // Fast path: Use scalar loop when N is 1
    if (N == 1) {
        Output[0] = expf(Input[0]);
        return;
    }
    
    // Vectorized path
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t stride = veclen;

    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }

        MLAS_SVFLOAT32 Vector = MlasSveLoadFloat32(Pred, Input);
        Vector = MlasSveComputeExpVector(Pred, Vector);
        MlasSveStoreFloat32(Pred, Output, Vector);

        Input += stride;
        Output += stride;
        N -= stride;
    }
}

MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeSumExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector,
    MLAS_SVFLOAT32 NegativeMaximumVector
)
{
    Vector = MlasSveAddFloat32(Pred, Vector, NegativeMaximumVector);
    Vector = MlasSveMaximumFloat32(Pred, MlasSveBroadcastFloat32(MlasSveExpConstants.LowerRangeSumExp), Vector);
    
    const auto RoundingBias = MlasSveBroadcastFloat32(MlasSveExpConstants.RoundingBias);
    auto biased = MlasSveMultiplyAddFloat32(Pred, Vector, MlasSveExpConstants.Log2Reciprocal, RoundingBias);
    auto m = MlasSveSubtractFloat32(Pred, biased, RoundingBias);

    Vector = MlasSveMultiplyAddFloat32(Pred, m, MlasSveExpConstants.Log2High, Vector);
    Vector = MlasSveMultiplyAddFloat32(Pred, m, MlasSveExpConstants.Log2Low, Vector);

    auto normal = MlasSveShiftLeftInt32<23>(Pred, MlasSveReinterpretAsInt32(biased));
    normal = MlasSveAddInt32(Pred, normal, MlasSveBroadcastInt32(MlasSveExpConstants.MaximumExponent));

    auto p = MlasSveBroadcastFloat32(MlasSveExpConstants.poly_0);
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_1);
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_2);
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_3);
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_4);
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_56);  // <--|
    p = MlasSveMultiplyAddFloat32(Pred, p, Vector, MlasSveExpConstants.poly_56);  // Twice?

    p = MlasSveMultiplyFloat32(Pred, p, MlasSveReinterpretAsFloat32(normal));
    return p;
}

float
MLASCALL
MlasSveComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
)
/**
 * Potential optimization: Consider applying loop unrolling to improve instruction-level
 * parallelism (ILP) in this kernel. Evaluate the performance impact using benchmarks
 * before and after implementing the optimization.
 */
{
    if (N == 1) {
        float result = expf(Input[0] + *NegativeMaximum);
        if (Output != nullptr) {
            Output[0] = result;
        }
        return result;
    }
    
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t veclen = svcntw();
    size_t stride = veclen;
    float sum = 0.0f;

    MLAS_SVFLOAT32 NegativeMaximumVector = MlasSveBroadcastFloat32(*NegativeMaximum);
   
    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
       
        MLAS_SVFLOAT32 Vector = MlasSveLoadFloat32(Pred, Input);
        Vector = MlasSveComputeSumExpVector(Pred, Vector, NegativeMaximumVector);

        if (Output != nullptr) {
            MlasSveStoreFloat32(Pred, Output, Vector);
            Output += stride;
        }

        sum += MlasSveReduceAddFloat32(Pred, Vector);

        Input += stride;
        N -= stride;
    }
    return sum;
}

float MLASCALL
MlasSveReduceMaximumF32Kernel(
    const float* Input,
    size_t N
)
{
    size_t veclen = svcntw();
    MLAS_SVBOOL Pred = svptrue_b32();

    float Maximum;
    MLAS_SVFLOAT32 MaximumVector0 = MlasSveBroadcastFloat32(MlasSveMinimumF32Value);
    
    if (N >= veclen * 4) {
        MLAS_SVFLOAT32 MaximumVector1 = MaximumVector0;
        MLAS_SVFLOAT32 MaximumVector2 = MaximumVector0;
        MLAS_SVFLOAT32 MaximumVector3 = MaximumVector0;
        
        while (N >= veclen * 4) {
            MaximumVector0 = MlasSveMaximumFloat32(Pred, MaximumVector0, MlasSveLoadFloat32(Pred, Input));
            MaximumVector1 = MlasSveMaximumFloat32(Pred, MaximumVector1, MlasSveLoadFloat32(Pred, Input + veclen));
            MaximumVector2 = MlasSveMaximumFloat32(Pred, MaximumVector2, MlasSveLoadFloat32(Pred, Input + 2 * veclen));
            MaximumVector3 = MlasSveMaximumFloat32(Pred, MaximumVector3, MlasSveLoadFloat32(Pred, Input + 3 * veclen));

            Input += veclen * 4;
            N -= veclen * 4;
        }

        MaximumVector0 = MlasSveMaximumFloat32(Pred, MaximumVector0, MaximumVector1);
        MaximumVector2 = MlasSveMaximumFloat32(Pred, MaximumVector2, MaximumVector3);
        MaximumVector0 = MlasSveMaximumFloat32(Pred, MaximumVector0, MaximumVector2);
    }
    size_t stride = veclen;
    
    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Vector = MlasSveLoadFloat32(Pred, Input);
        MaximumVector0 = MlasSveMaximumFloat32(Pred, MaximumVector0, Vector);

        Input += stride;
        N -= stride;
    }

    Maximum = MlasSveReduceMaximumFloat32(svptrue_b32(), MaximumVector0);
    return Maximum;
}

void
MLASCALL
MlasSveReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
)
{
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t veclen = svcntw();
    size_t stride = veclen;

    float tmp_min = std::numeric_limits<float>::max();
    float tmp_max = std::numeric_limits<float>::lowest();

    MLAS_SVFLOAT32 MaximumVector = MlasSveBroadcastFloat32(tmp_max);
    MLAS_SVFLOAT32 MinimumVector = MlasSveBroadcastFloat32(tmp_min);

    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Vector = MlasSveLoadFloat32(Pred, Input);
        MaximumVector = MlasSveMaximumFloat32(Pred, MaximumVector, Vector);
        MinimumVector = MlasSveMinimumFloat32(Pred, MinimumVector, Vector);

        Input += stride;
        N -= stride;
    }
    *Min = MlasSveReduceMinimumFloat32(svptrue_b32(), MinimumVector);
    *Max = MlasSveReduceMaximumFloat32(svptrue_b32(), MaximumVector);
}

void
MLASCALL
MlasSveComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
)
{
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t veclen = svcntw();
    size_t stride = veclen;

    const float Scale = Parameters[0];
    const MLAS_SVFLOAT32 ScaleVector = MlasSveBroadcastFloat32(Scale);
    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Vector = MlasSveMultiplyFloat32(Pred, ScaleVector, MlasSveLoadFloat32(Pred, Output));
        MlasSveStoreFloat32(Pred, Output, Vector);

        Output += stride;
        N -= stride;
    }
}

void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
)
{
    MLAS_SVBOOL Pred = svptrue_b32();
    size_t veclen = svcntw();
    size_t stride = veclen;
    
    const float NegativeMaximum = Parameters[0];
    const float Logarithm = Parameters[1];
    MLAS_SVFLOAT32 NegativeMaximumVector = MlasSveBroadcastFloat32(NegativeMaximum);
    MLAS_SVFLOAT32 LogarithmVector = MlasSveBroadcastFloat32(Logarithm);

    while (N > 0) {
        if (N < veclen) {
            Pred = svwhilelt_b32(0, (int32_t)N);
            stride = N;
        }
        MLAS_SVFLOAT32 Vector = MlasSveLoadFloat32(Pred, Input);
        Vector = MlasSveAddFloat32(Pred, Vector, NegativeMaximumVector);
        Vector = MlasSveSubtractFloat32(Pred, Vector, LogarithmVector);
        MlasSveStoreFloat32(Pred, Output, Vector);

        Input += stride;
        Output += stride;
        N -= stride;
    }
    
}
