//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

/*++

Module Name:

    halfconv.cpp

Abstract:

    This module implements public dispatch wrappers for optional half precision
    convolution backends.

--*/

#include "mlasi.h"

bool
MLASCALL
MlasHalfConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    float Beta,
    bool InputOutputChannelsLast,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    if (GetMlasPlatform().MlasHalfConvPrepareOverride == nullptr) {
        return false;
    }

    return GetMlasPlatform().MlasHalfConvPrepareOverride(
        Parameters,
        Dimensions,
        BatchCount,
        GroupCount,
        InputChannels,
        InputShape,
        KernelShape,
        DilationShape,
        Padding,
        StrideShape,
        OutputShape,
        FilterCount,
        Activation,
        WorkingBufferSize,
        Beta,
        InputOutputChannelsLast,
        ThreadPool,
        BackendKernelSelectorConfig);
}

bool
MLASCALL
MlasHalfConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const MLAS_FP16* Input,
    const MLAS_FP16* Filter,
    bool FilterAndBiasArePacked,
    const MLAS_FP16* Bias,
    MLAS_FP16* WorkingBuffer,
    MLAS_FP16* Output,
    MLAS_THREADPOOL* ThreadPool
    )
{
    if (GetMlasPlatform().MlasHalfConvOverride == nullptr) {
        return false;
    }

    if (FilterAndBiasArePacked && Bias != nullptr) {
        return false;
    }

    return GetMlasPlatform().MlasHalfConvOverride(
        Parameters,
        Input,
        Filter,
        FilterAndBiasArePacked,
        Bias,
        WorkingBuffer,
        Output,
        ThreadPool);
}

size_t
MLASCALL
MlasHalfConvPackWeightsAndBiasSize(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    if (BackendKernelSelectorConfig != nullptr && !BackendKernelSelectorConfig->use_kleidiai) {
        return 0;
    }

    if (GetMlasPlatform().MlasHalfConvPackWeightsAndBiasSizeOverride == nullptr) {
        return 0;
    }

    return GetMlasPlatform().MlasHalfConvPackWeightsAndBiasSizeOverride(
        FilterCount,
        InputChannels,
        KernelShape,
        DilationShape);
}

bool
MLASCALL
MlasHalfConvPackWeightsAndBias(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const MLAS_FP16* Filter,
    const MLAS_FP16* Bias,
    void* PackedWeightsAndBias,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    if (BackendKernelSelectorConfig != nullptr && !BackendKernelSelectorConfig->use_kleidiai) {
        return false;
    }

    if (GetMlasPlatform().MlasHalfConvPackWeightsAndBiasOverride == nullptr) {
        return false;
    }

    return GetMlasPlatform().MlasHalfConvPackWeightsAndBiasOverride(
        FilterCount,
        InputChannels,
        KernelShape,
        DilationShape,
        Filter,
        Bias,
        PackedWeightsAndBias,
        ThreadPool);
}
