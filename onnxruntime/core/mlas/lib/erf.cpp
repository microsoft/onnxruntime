/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    erf.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in: https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff
    Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"

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
} MlasErfConstants = {
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

void
MLASCALL
MlasErfKernel(
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
    while (N >= 4) {
        MLAS_FLOAT32X4 Value = MlasLoadFloat32x4(Input);
        MLAS_FLOAT32X4 NegZero = MlasBroadcastFloat32x4(MlasErfConstants.ErfNegZero);
        MLAS_FLOAT32X4 SignMask = MlasAndFloat32x4(Value, NegZero);
        MLAS_FLOAT32X4 AbsValue = MlasAndNotFloat32x4(NegZero, Value);
        AbsValue = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.ErfUpperAbsRange), AbsValue);
        MLAS_FLOAT32X4 SquareValue = MlasMultiplyFloat32x4(AbsValue, AbsValue);

        MLAS_FLOAT32X4 r_small = MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P0);
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P1));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P2));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P3));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P4));
        r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P5_Minus_One));
        r_small = MlasMultiplyAddFloat32x4(r_small, AbsValue, AbsValue);
        MLAS_FLOAT32X4 split_mask = MlasGreaterThanFloat32x4(AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfSplitBoundary));
        r_small = MlasAndNotFloat32x4(split_mask, r_small);

        AbsValue = MlasAndFloat32x4(split_mask, AbsValue); // clear smaller value into zero for bigger number calculation
        MLAS_FLOAT32X4 r_big = MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P0);
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P1));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P2));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P3));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P4));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P5));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P6_Minus_One));
        r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, AbsValue);

        // 1.0 - exp(-r_big), no need to do min()
        r_big = MlasXorFloat32x4(r_big, MlasBroadcastFloat32x4(MlasErfConstants.ErfNegZero)); // -r_big
        r_big = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.Exp_LowerRange), r_big);
        MLAS_FLOAT32X4 exp_c = MlasBroadcastFloat32x4(MlasErfConstants.Exp_C);
        MLAS_FLOAT32X4 r = MlasMultiplyAddFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.Exp_Log2Reciprocal), r_big, exp_c);
        r = MlasSubtractFloat32x4(r, exp_c);

        MLAS_FLOAT32X4 fx = MlasMultiplyAddFloat32x4(r, MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_hi), r_big);
        fx = MlasMultiplyAddFloat32x4(r, MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_lo), fx);
        // y = exp(fx)
        MLAS_FLOAT32X4 y = MlasBroadcastFloat32x4(MlasErfConstants.Exp_P0);
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P1));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P2));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P3));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P4));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P5));
        y = MlasMultiplyAddFloat32x4(y, fx, MlasBroadcastFloat32x4(MlasErfConstants.Exp_P6));
        // 1.0 - exp(fx) * 2^INT(r)
        y = MlasMultiplyFloat32x4(y, MlasPowerOf2Float32x4(r));
        y = MlasSubtractFloat32x4(MlasBroadcastFloat32x4(MlasErfConstants.ErfOne), y);

        // merge two splits results
        y = MlasOrFloat32x4(r_small, y);
        y = MlasOrFloat32x4(y, SignMask);

        MlasStoreFloat32x4(Output, y);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float Value = *Input++;
        float AbsValue = fabsf(Value);

        float r;
        if (AbsValue > MlasErfConstants.ErfSplitBoundary) {
            AbsValue = std::min(MlasErfConstants.ErfUpperAbsRange, AbsValue);
            float r_big = MlasErfConstants.ErfBIG_P0;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P1;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P2;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P3;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P4;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P5;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P6_Minus_One;
            r_big = r_big * AbsValue + AbsValue;

            r_big = std::max(-r_big, MlasErfConstants.Exp_LowerRange);
            r = MlasErfConstants.Exp_Log2Reciprocal * r_big + MlasErfConstants.Exp_C;
            r -= MlasErfConstants.Exp_C;
            float fx = r * MlasErfConstants.Exp_log2_hi + r_big;
            fx = r * MlasErfConstants.Exp_log2_lo + fx;

            float y = MlasErfConstants.Exp_P0;
            y = y * fx + MlasErfConstants.Exp_P1;
            y = y * fx + MlasErfConstants.Exp_P2;
            y = y * fx + MlasErfConstants.Exp_P3;
            y = y * fx + MlasErfConstants.Exp_P4;
            y = y * fx + MlasErfConstants.Exp_P5;
            y = y * fx + MlasErfConstants.Exp_P6;

            r = 1.0f - ldexpf(y, (int)r);
            r = (Value <= -0.0f) ? -r : r;
        }
        else {
            float SquareValue = AbsValue * AbsValue;
            r = MlasErfConstants.ErfSMALL_P0;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P1;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P2;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P3;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P4;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P5_Minus_One;
            r = r * Value + Value;
        }

        *Output++ = r;
        N -= 1;
    }
}

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.ErfKernelRoutine(Input, Output, N);
#else
    MlasErfKernel(Input, Output, N);
#endif
}
