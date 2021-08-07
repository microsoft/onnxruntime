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

//
// Structure layout for the logistic constants block.
//

        .equ    .LLogisticConstants_LowerRange, 0
        .equ    .LLogisticConstants_UpperRange, 4
        .equ    .LLogisticConstants_alpha_9, 8
        .equ    .LLogisticConstants_alpha_7, 12
        .equ    .LLogisticConstants_alpha_5, 16
        .equ    .LLogisticConstants_alpha_3, 20
        .equ    .LLogisticConstants_alpha_1, 24
        .equ    .LLogisticConstants_beta_10, 28
        .equ    .LLogisticConstants_beta_8, 32
        .equ    .LLogisticConstants_beta_6, 36
        .equ    .LLogisticConstants_beta_4, 40
        .equ    .LLogisticConstants_beta_2, 44
        .equ    .LLogisticConstants_beta_0, 48
        .equ    .LLogisticConstants_one_half, 52

//
// Structure layout for the tanh constants block.
//

        .equ    .LTanhConstants_LowerRange, 0
        .equ    .LTanhConstants_UpperRange, 4
        .equ    .LTanhConstants_alpha_13, 8
        .equ    .LTanhConstants_alpha_11, 12
        .equ    .LTanhConstants_alpha_9, 16
        .equ    .LTanhConstants_alpha_7, 20
        .equ    .LTanhConstants_alpha_5, 24
        .equ    .LTanhConstants_alpha_3, 28
        .equ    .LTanhConstants_alpha_1, 32
        .equ    .LTanhConstants_beta_6, 36
        .equ    .LTanhConstants_beta_4, 40
        .equ    .LTanhConstants_beta_2, 44
        .equ    .LTanhConstants_beta_0, 48
