/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelFma3.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"
#include "DgemmKernelCommon.h"
#include "FgemmKernelFma3Common.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelFma3Function MlasGemmDoubleKernelFma3

        .end
