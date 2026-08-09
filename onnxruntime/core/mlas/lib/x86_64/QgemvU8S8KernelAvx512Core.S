/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemvU8S8KernelAvx512Core.s

Abstract:

    This module implements the kernels for the quantized integer matrix/vector
    multiply operation (QGEMV).

    This implementation uses AVX512 core instructions (BW/DQ/VL).

--*/

#include "asmmacro.h"
#include "QgemvU8S8KernelAvx512Common.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMV kernel.
//

GemvU8S8KernelAvx512Function Avx512Core

        .end
