/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchw_depthwise_multiplier_greater_than_1.cpp

Abstract:

    This module implements the single precision NCHW grouped convolution entry
    point for the currently supported depth multiplier > 1 case.

    At present, this entry point is only valid for the exact MobileClip
    grouped projection shape family:

      - 2D convolution
      - input channels per group = 1
      - output channels per group = 2
      - kernel = 7x7
      - stride = 2x2
      - dilation = 1x1
      - padding = 3,3,3,3
      - AVX512F NCHW float kernel selected on AMD64

--*/

#include "mlasi.h"

#include <cassert>

void
MlasConvDepthwiseWithMultiplierFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    Dispatches the currently supported depth-multiplier-greater-than-1
    implementation.

    This routine is intentionally narrow for now and assumes that the caller
    has already matched the exact MobileClip grouped projection constraints in
    MlasConvPrepare().

Arguments:

    Parameters - Supplies the prepared convolution parameters. The following
        constraints are required:
          * Dimensions == 2
          * GroupCount > 1
          * InputChannels == 1
          * FilterCount == 2
          * KernelShape == {7, 7}
          * StrideShape == {2, 2}
          * DilationShape == {1, 1}
          * Padding == {3, 3, 3, 3}

    Input - Supplies one batch/group input slice in CHW layout.

    Filter - Supplies one group filter block in OIHW layout.

    Output - Supplies one batch/group output slice in CHW layout.

    Zeros - Accepted for signature consistency with MlasConvDepthwiseFloat_CHW()

Return Value:

    None.

--*/
{
    MLAS_UNREFERENCED_PARAMETER(Zeros);

    assert(Parameters->Dimensions == 2);
    assert(Parameters->GroupCount > 1);
    assert(Parameters->InputChannels == 1);
    assert(Parameters->FilterCount == 2);
    assert(Parameters->KernelShape[0] == 7);
    assert(Parameters->KernelShape[1] == 7);
    assert(Parameters->StrideShape[0] == 2);
    assert(Parameters->StrideShape[1] == 2);
    assert(Parameters->DilationShape[0] == 1);
    assert(Parameters->DilationShape[1] == 1);
    assert(Parameters->Padding[0] == 3);
    assert(Parameters->Padding[1] == 3);
    assert(Parameters->Padding[2] == 3);
    assert(Parameters->Padding[3] == 3);

#if defined(MLAS_TARGET_AMD64)
    if (GetMlasPlatform().ConvNchwFloatKernel != MlasConvNchwFloatKernelAvx512F) {
        MLAS_THROW_EX(std::runtime_error,
            "MlasConvDepthwiseWithMultiplierFloat_CHW: invalid AVX512F kernel dispatch");
    }

    MlasConvDepthwiseMultiplier2CHWKernel7x7S2Avx512F(
        Input,
        Parameters->InputShape[0],
        Parameters->InputShape[1],
        Filter,
        Output,
        Parameters->OutputShape[0],
        Parameters->OutputShape[1],
        Parameters->Beta);
#else
    MLAS_UNREFERENCED_PARAMETER(Parameters);
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_THROW_EX(std::runtime_error,
        "MlasConvDepthwiseWithMultiplierFloat_CHW: not implemented for this platform");
#endif
}
