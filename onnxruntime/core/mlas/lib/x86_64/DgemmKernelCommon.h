/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the double
    precision matrix/matrix multiply operation (DGEMM).

--*/

//
// Define the double precision parameters.
//

        .equ    .LFgemmElementShift, 3
        .equ    .LFgemmElementSize, 1 << .LFgemmElementShift

#include "FgemmKernelCommon.h"

//
// Define the typed instructions for double precision.
//

FGEMM_TYPED_INSTRUCTION(addpf, addpd)
FGEMM_TYPED_INSTRUCTION(movsf, movsd)
FGEMM_TYPED_INSTRUCTION(movupf, movupd)

FGEMM_TYPED_INSTRUCTION(vaddpf, vaddpd)
FGEMM_TYPED_INSTRUCTION(vbroadcastsf, vbroadcastsd)
FGEMM_TYPED_INSTRUCTION(vfmadd213pf, vfmadd213pd)
FGEMM_TYPED_INSTRUCTION(vfmadd231pf, vfmadd231pd)
FGEMM_TYPED_INSTRUCTION(vmaskmovpf, vmaskmovpd)
FGEMM_TYPED_INSTRUCTION(vmovapf, vmovapd)
FGEMM_TYPED_INSTRUCTION(vmovsf, vmovsd)
FGEMM_TYPED_INSTRUCTION(vmovupf, vmovupd)
FGEMM_TYPED_INSTRUCTION(vmulpf, vmulpd)
FGEMM_TYPED_INSTRUCTION(vxorpf, vxorpd)

        .macro vfmadd231pf_bcst DestReg, SrcReg, Address

        vfmadd231pd \DestReg\(), \SrcReg\(), \Address\(){1to8}

        .endm
