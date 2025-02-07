/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    softmax.

--*/

#pragma once

#include "mlasi.h"

struct MLAS_SOFTMAX_DISPATCH {
    /**
     * @brief Compute the hyperbolic tangent function for each element of the input array
     * @param Input         Address of the input array
     * @param Output        Address of the output array. Could be the same as the input array.
     * @param N             Number of elements in the input array
     */
    typedef void(Tanh_Fp16_Fn)(
        const MLAS_FP16* Input,
        MLAS_FP16* Output,
        size_t N
    );

    Tanh_Fp16_Fn* Tanh_Fp16 = nullptr;

    /**
     * @brief Compute the exponential function for each element of the input array
     * @param Input         Address of the input array
     * @param Output        Address of the output array. Could be the same as the input array.
     * @param N             Number of elements in the input array
     */
    typedef void(Exp_Fp16_Fn)(
        const MLAS_FP16* Input,
        MLAS_FP16* Output,
        size_t N
    );

    Exp_Fp16_Fn* Exp_Fp16 = nullptr;

    /**
     * @brief Find the max value among the input array
     * @param Input         Address of the input array
     * @param N             Number of elements in the input array
     */
    typedef MLAS_FP16(ReduceMax_Fp16_Fn)(
        const MLAS_FP16* Input,
        size_t N
    );

    ReduceMax_Fp16_Fn* ReduceMax_Fp16 = nullptr;

    /**
     * @brief Compute the expotential function for each element of the input array and returnt he sum. It has smaller
     *        dynamic range for the input than Exp_Fp16_Fn.
     * @param Input         Address of the input array
     * @param Output        Address of the output array. Could be the same as the input array or nullptr.
     * @param N             Number of elements in the input array
     * @param NegativeMaximum   The negative of the maximum value in the input array
     */
    typedef MLAS_FP16(SumExp_Fp16_Fn)(
        const MLAS_FP16* Input,
        MLAS_FP16* Output,
        size_t N,
        const MLAS_FP16 NegativeMaximum
    );

    SumExp_Fp16_Fn* SumExp_Fp16 = nullptr;

    /**
     * @brief Compute the softmax output for each element of the input array
     * @param Input         Address of the input array
     * @param Output        Address of the output array. Could be the same as the input array.
     * @param N             Number of elements in the input array
     * @param scale         The scale factor to apply to the output
     */
    typedef void(Softmax_Fp16_Fn)(
        const MLAS_FP16* Input,
        MLAS_FP16* Output,
        size_t N,
        const MLAS_FP16 scale
    );

    Softmax_Fp16_Fn* Softmax_Fp16 = nullptr;

    /**
     * @brief Compute the log softmax output for each element of the input array
     * @param Input         Address of the input array
     * @param Output        Address of the output array. Could be the same as the input array.
     * @param N             Number of elements in the input array
     * @param NagativeMaximum   The negative of the maximum value in the input array
     * @param LogSum        The logarithm of the sum of the exponential function of the input array
     */
    typedef void(LogSoftmax_Fp16_Fn)(
        const MLAS_FP16* Input,
        MLAS_FP16* Output,
        size_t N,
        const MLAS_FP16 NagativeMaximum,
        const MLAS_FP16 LogSum
    );

    LogSoftmax_Fp16_Fn* LogSoftmax_Fp16 = nullptr;
};
