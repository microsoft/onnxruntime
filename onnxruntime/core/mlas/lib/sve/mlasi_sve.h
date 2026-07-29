/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlasi_sve.h

Abstract:

    This module contains the procedure prototypes for the SVE elementwise
    kernels.

    This header is deliberately free of SVE types and intrinsics so that it
    can be included from translation units that are not compiled with SVE
    support (platform.cpp today, and eventually MSVC, which has no SVE
    intrinsics at all). The kernel implementations get their SVE code
    generation from their per-file -march compile flags.

--*/

#pragma once

#include "../mlasi.h"

//
// Layouts of the shared constant tables defined in erf.cpp, logistic.cpp and
// compute.cpp ("bundles the constants for use by kernels written in
// assembly"). The tables have C linkage precisely so other implementations of
// the same kernels can reference them; the SVE kernels bind to the same
// symbols instead of maintaining duplicate copies.
//

struct MLAS_ERF_CONSTANTS {
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
};

struct MLAS_LOGISTIC_CONSTANTS {
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
};

struct MLAS_EXP_CONSTANTS {
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
};

extern "C" const MLAS_ERF_CONSTANTS MlasErfConstants;
extern "C" const MLAS_LOGISTIC_CONSTANTS MlasLogisticConstants;
extern "C" const MLAS_EXP_CONSTANTS MlasExpConstants;
extern "C" const float MlasMinimumF32Value;

//
// Constants of the ARM Compute Library exp() approximation used by the SVE
// Exp kernel, as float bit patterns. Defined in elementwise_sve_dispatch.cpp
// and passed to the kernel by pointer so its code stays free of data
// relocations.
//

struct MLAS_SVE_EXP_CONSTANTS {
    uint32_t C1;
    uint32_t C2;
    uint32_t C3;
    uint32_t C4;
    uint32_t C5;
    uint32_t Shift;
    uint32_t InvLn2;
    uint32_t NegLn2Hi;
    uint32_t NegLn2Lo;
    uint32_t Inf;
    uint32_t MaxInput;
    uint32_t MinInput;
    uint32_t FexpaShift;
    uint32_t OneHalf;
    uint32_t FexpaLowerRange;
};

extern "C" const MLAS_SVE_EXP_CONSTANTS MlasSveExpConstants;

//
// Float32 elementwise kernels (elementwise_sve.cpp).
//

void
MLASCALL
MlasSveErfKernel(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasSveLogisticKernel(
    const float* Input,
    float* Output,
    size_t N
    );

void
MLASCALL
MlasSveComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
    );

float
MLASCALL
MlasSveComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
    );

float
MLASCALL
MlasSveReduceMaximumF32Kernel(
    const float* Input,
    size_t N
    );

void
MLASCALL
MlasSveReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    );

void
MLASCALL
MlasSveComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
    );

void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
    );

//
// Float16 elementwise kernels (elementwise_sve_fp16.cpp).
//

void
MLASCALL
MlasSveErfFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    );

void
MLASCALL
MlasSveTanhFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    );

void
MLASCALL
MlasSveGeluFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    MLAS_FP16* Temp,
    size_t N,
    MLAS_GELU_ALGORITHM Algo
    );
