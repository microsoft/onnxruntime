/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_lut_kernel_neon.h

Abstract:

    This module contains the dispatch table declaration for ARM64 NEON
    LUT-based n-bit quantized integer matrix multiplication kernels.

--*/

#pragma once

#include "qlutgemm.h"

//
// External dispatch table for ARM NEON LUT GEMM kernels.
// Kernel functions are internal to the .cpp file and accessed via this dispatch.
//
extern const MLAS_QNBIT_LUT_GEMM_DISPATCH MlasLutGenKernelNeon;

