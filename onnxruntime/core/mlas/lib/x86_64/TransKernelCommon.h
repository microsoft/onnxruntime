/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    TransKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the
    transcendental functions.

--*/

//
// Structure layout for the exponential function constants block.
//

        .equ    .LExpConstants_LowerRange, 0
        .equ    .LExpConstants_UpperRange, 4
        .equ    .LExpConstants_LowerRangeSumExp, 8
        .equ    .LExpConstants_UpperRangeSumExp, 12
        .equ    .LExpConstants_RoundingBias, 16
        .equ    .LExpConstants_Log2Reciprocal, 20
        .equ    .LExpConstants_Log2High, 24
        .equ    .LExpConstants_Log2Low, 28
        .equ    .LExpConstants_poly_0, 32
        .equ    .LExpConstants_poly_1, 36
        .equ    .LExpConstants_poly_2, 40
        .equ    .LExpConstants_poly_3, 44
        .equ    .LExpConstants_poly_4, 48
        .equ    .LExpConstants_poly_56, 52
        .equ    .LExpConstants_MinimumExponent, 56
        .equ    .LExpConstants_MaximumExponent, 60
