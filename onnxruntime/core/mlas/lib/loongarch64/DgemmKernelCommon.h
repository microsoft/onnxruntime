/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the double
    precision matrix/matrix multiply operation (DGEMM).

--*/

#define     LFgemmElementShift      3
#define     LFgemmElementSize       (1 << LFgemmElementShift)
#define     LFgemmYmmElementCount   (32/LFgemmElementSize)

#include "FgemmKernelCommon.h"

FGEMM_TYPED_INSTRUCTION(xvfadd, xvfadd.d)
FGEMM_TYPED_INSTRUCTION(xvfmadd, xvfmadd.d)
FGEMM_TYPED_INSTRUCTION(xvldrepl, xvldrepl.d)
FGEMM_TYPED_INSTRUCTION(xvfmul, xvfmul.d)
