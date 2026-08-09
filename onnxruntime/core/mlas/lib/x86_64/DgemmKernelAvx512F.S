/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelAvx512F.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"
#include "DgemmKernelCommon.h"
#include "FgemmKernelAvx512FCommon.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelAvx512FFunction MlasGemmDoubleKernelAvx512F

        .end
