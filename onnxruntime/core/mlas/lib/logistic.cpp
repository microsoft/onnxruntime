/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    logistic.cpp

Abstract:

    This module implements routines to compute the logistic function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in Eigen. Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"

//
// Bundles the floating point constants for use by kernels written in assembly.
//

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
} MlasLogisticConstants = {
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

void
MLASCALL
MlasLogisticKernel(
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
    while (N >= 4) {

        MLAS_FLOAT32X4 Value = MlasLoadFloat32x4(Input);

        Value = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasLogisticConstants.LowerRange), Value);
        Value = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasLogisticConstants.UpperRange), Value);

        MLAS_FLOAT32X4 ValueSquared = MlasMultiplyFloat32x4(Value, Value);

        MLAS_FLOAT32X4 p;
        p = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.alpha_9),
            MlasBroadcastFloat32x4(MlasLogisticConstants.alpha_7));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.alpha_5));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.alpha_3));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.alpha_1));
        p = MlasMultiplyFloat32x4(p, Value);

        MLAS_FLOAT32X4 q;
        q = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.beta_10),
            MlasBroadcastFloat32x4(MlasLogisticConstants.beta_8));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.beta_6));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.beta_4));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.beta_2));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasLogisticConstants.beta_0));

        MlasStoreFloat32x4(Output, MlasAddFloat32x4(MlasDivideFloat32x4(p, q), MlasBroadcastFloat32x4(0.5f)));

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {

        float Value = *Input++;

        Value = std::min(MlasLogisticConstants.UpperRange, std::max(MlasLogisticConstants.LowerRange, Value));

        float ValueSquared = Value * Value;

        float p;
        p = ValueSquared * MlasLogisticConstants.alpha_9 + MlasLogisticConstants.alpha_7;
        p = p * ValueSquared + MlasLogisticConstants.alpha_5;
        p = p * ValueSquared + MlasLogisticConstants.alpha_3;
        p = p * ValueSquared + MlasLogisticConstants.alpha_1;
        p = p * Value;

        float q;
        q = ValueSquared * MlasLogisticConstants.beta_10 + MlasLogisticConstants.beta_8;
        q = q * ValueSquared + MlasLogisticConstants.beta_6;
        q = q * ValueSquared + MlasLogisticConstants.beta_4;
        q = q * ValueSquared + MlasLogisticConstants.beta_2;
        q = q * ValueSquared + MlasLogisticConstants.beta_0;

        *Output++ = (p / q) + 0.5f;

        N -= 1;
    }
}

void
MLASCALL
MlasComputeLogistic(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the logistic function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.LogisticKernelRoutine(Input, Output, N);
#else
    MlasLogisticKernel(Input, Output, N);
#endif
}
