/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    elementwise_constants.h

Abstract:

    This module contains the layouts of the floating point constant tables
    shared by the elementwise kernels (erf.cpp, logistic.cpp, tanh.cpp and
    compute.cpp define them).

    The tables have C linkage and hidden visibility (MLAS_INTERNAL_DATA)
    because kernels written in assembly reference them by symbol. Naming the
    layouts here lets every translation unit -- the definitions themselves and
    any other implementation of the same kernels -- agree on one type for each
    symbol.

--*/

#pragma once

#include <cstdint>

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

struct MLAS_TANH_CONSTANTS {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
};
