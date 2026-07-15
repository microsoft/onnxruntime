/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    softmax_kernel_neon.cpp

Abstract:

    This module implements the softmax kernels for ARM NEON.

--*/

#include "softmax.h"
#include "softmax_kernel_neon.h"

//
// Kernel dispatch structure definition.
//
const MLAS_SOFTMAX_DISPATCH MlasSoftmaxDispatchNeon = []() {
    MLAS_SOFTMAX_DISPATCH d;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    if (MlasFp16AccelerationSupported()) {
        d.Tanh_Fp16 = softmax_neon::Tanh_Kernel_Fp16;
        d.Softcap_Fp16 = softmax_neon::Softcap_Kernel_Fp16;
        d.Exp_Fp16 = softmax_neon::Exp_Kernel_Fp16;
        d.ReduceMax_Fp16 = softmax_neon::ReduceMax_Kernel_Fp16;
        d.SumExp_Fp16 = softmax_neon::SumExp_Kernel_Fp16;
        d.Softmax_Fp16 = softmax_neon::Softmax_Kernel_Fp16;
        d.LogSoftmax_Fp16 = softmax_neon::LogSoftmax_Kernel_Fp16;
    }
#endif
    return d;
}();
