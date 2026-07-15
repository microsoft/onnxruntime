/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_kernel_neon.h

Abstract:

    This module includes function declarations and common helper functions for
    softmax on ARM cpu.

--*/

#pragma once

#include <arm_neon.h>

#include "mlasi.h"

namespace softmax_neon {

void Tanh_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N);

void Softcap_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Softcap);

void Exp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N);

MLAS_FP16 ReduceMax_Kernel_Fp16(const MLAS_FP16* Input, size_t N);

MLAS_FP16 SumExp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum);

void Softmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Sum);

void LogSoftmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum);

}  // namespace rope_neon
