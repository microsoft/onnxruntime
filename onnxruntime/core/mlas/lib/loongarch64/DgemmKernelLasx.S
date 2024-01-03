/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelLasx.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses Lasx instructions.

--*/

#include "asmmacro.h"
#include "DgemmKernelCommon.h"
#include "FgemmKernelLasxCommon.h"

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelLasxFunction MlasGemmDoubleKernelLasx

        .end
