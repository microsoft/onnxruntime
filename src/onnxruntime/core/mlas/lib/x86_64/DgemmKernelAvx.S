/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelAvx.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"
#include "DgemmKernelCommon.h"
#include "FgemmKernelAvxCommon.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelAvxFunction MlasGemmDoubleKernelAvx

        .end
