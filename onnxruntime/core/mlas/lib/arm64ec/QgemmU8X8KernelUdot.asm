/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelUdot.asm

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses ARM v8.4 dot product instructions.

--*/

#include "..\arm64\QgemmU8X8KernelUdot.asm"
