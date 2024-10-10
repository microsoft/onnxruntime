/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    tanh.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

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
    uint32_t _nc2;
    uint32_t _nc1;
    uint32_t _nc0;
    uint32_t _dc2;
    uint32_t _dc1;
    uint32_t _dc0;
    uint32_t _absmask;
    uint32_t _ubound;
} MlasTanhConstants = {
    0x3c520a84, /* _nc2  */
    0x3edef102, /* _nc1  */
    0x3f800000, /* _nc0  */
    0x3a2fc8e6, /* _dc2  */
    0x3dd1c060, /* _dc1  */
    0xb859e195, /* _dc0  */
    0x7fffffff, /* _absmask  */
    0x40a00000, /* _ubound = +5.0f */
};

void
MLASCALL
MlasTanhKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const MLAS_FLOAT32X4 nc0 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._nc0));
    const MLAS_FLOAT32X4 nc1 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._nc1));
    const MLAS_FLOAT32X4 nc2 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._nc2));
    const MLAS_FLOAT32X4 dc0 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._dc0));
    const MLAS_FLOAT32X4 dc1 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._dc1));
    const MLAS_FLOAT32X4 dc2 = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._dc2));
    const MLAS_FLOAT32X4 ub = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._ubound));
    const MLAS_FLOAT32X4 absmask = MlasBroadcastFloat32x4(reinterpret_cast<const float*>(&MlasTanhConstants._absmask));
    MLAS_FLOAT32X4 Val;

    size_t count = 0;
    while (count < N) {
        if (N - count >= 4) {
            Val = MlasLoadFloat32x4(Input);
        } else {
            Val = MlasPartialLoadFloat32x4(Input, static_cast<int>(N - count));
        }
        MLAS_FLOAT32X4 ValAbs = MlasAndFloat32x4(Val, absmask);
        MLAS_FLOAT32X4 boundmask = MlasGreaterThanEqualFloat32x4(ValAbs, ub);
        MLAS_FLOAT32X4 signVal = MlasXorFloat32x4(ValAbs, Val);
        MLAS_FLOAT32X4 ValSq = MlasMultiplyFloat32x4(ValAbs, ValAbs);

        MLAS_FLOAT32X4 npoly = MlasMultiplyAddFloat32x4(nc2, ValSq, nc1);
        npoly = MlasMultiplyAddFloat32x4(npoly, ValSq, nc0);

        MLAS_FLOAT32X4 dpoly = MlasMultiplyAddFloat32x4(dc2, ValSq, dc1);
        dpoly = MlasMultiplyAddFloat32x4(dpoly, ValSq, dc0);
        dpoly = MlasMultiplyAddFloat32x4(dpoly, ValAbs, ValAbs);

        MLAS_FLOAT32X4 out = MlasDivideFloat32x4(dpoly, npoly);
        out = MlasBlendFloat32x4(out, nc0, boundmask);
        out = MlasXorFloat32x4(out, signVal);

        if (N - count >= 4) {
            MlasStoreFloat32x4(Output, out);
        } else {
            MlasPartialStoreFloat32x4(Output, out, static_cast<int>(N - count));
        }

        Input += 4;
        Output += 4;
        count += 4;
    }
}

void
MLASCALL
MlasComputeTanh(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().TanhKernelRoutine(Input, Output, N);
#else
    MlasTanhKernel(Input, Output, N);
#endif
}
