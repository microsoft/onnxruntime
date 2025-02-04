/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_kernel_neon_fp16.cpp

Abstract:

    This module implements the fp16 softmax kernels for ARM NEON.

--*/
#include <arm_neon.h>
#include <cassert>

#include "fp16_common.h"
#include "softmax.h"
#include "softmax_kernel_neon.h"

namespace softmax_neon {

// tanh kernel for fp16. Output and input can be the same buffer.
void Tanh_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N) {

}

void Softcap_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Softcap) {

}

// exp kernel for fp16. Output and input can be the same buffer.
void Exp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N) {

}

// reduce max kernel for fp16
MLAS_FP16 ReduceMax_Kernel_Fp16(const MLAS_FP16* Input, size_t N) {
    return MLAS_FP16::FromBits(0);
}

MLAS_FP16 SumExp_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum) {
    return MLAS_FP16::FromBits(0);
}

void Softmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 scale) {

}

void LogSoftmax_Kernel_Fp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum) {
}

}  // namespace rope_neon
