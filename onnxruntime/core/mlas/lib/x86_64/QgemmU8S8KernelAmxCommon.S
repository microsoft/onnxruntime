/*++

Copyright (c) 2020 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemvU8S8KernelAvxVnni.s

Abstract:

    This module implements the kernels for the quantized integer matrix/vector
    multiply operation (QGEMV).

    This implementation uses AVXVNNI instructions.

--*/

#include "asmmacro.h"
#include "AssembleAvxVnni.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8S8 kernel.
//

        .equ    .LGemvU8S8KernelFrame_mask, -8
        .equ    .LGemvU8S8KernelFrame_SavedRbx, 0
        .equ    .LGemvU8S8KernelFrame_SavedRbp, 8
        .equ    .LGemvU8S8KernelFrame_ReturnAddress, 16

        .text

/*++

Routine Description:

    This routine is an inner kernel to compute matrix/vector multiplication.

Arguments:

    A (rdi) - Supplies the address of vector A.

    B (rsi) - Supplies the address of matrix B.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from vector A and the number
        of rows from matrix B to iterate over.

    CountN (r8) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb (r9) - Supplies the first dimension of matrix B.

Return Value:

    None.

--*/
	
	.globl  C_UNDERSCORE(MlasTdpAmxT4T2T0)
C_UNDERSCORE(MlasTdpAmxT4T2T0):

	push    rbp
        push    rbx

        #mov     r10,rdx
        #mov     r11,rsp                     # set ZeroMode to any non-zero value
	
	TdpbusdTmmTmmTmm tmm4, tmm2, tmm0	

	pop rbx
	pop rbp

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT5T3T0)
C_UNDERSCORE(MlasTdpAmxT5T3T0):

	push    rbp
        push    rbx

        #mov     r10,rdx
        #mov     r11,rsp                     # set ZeroMode to any non-zero value
	
	TdpbusdTmmTmmTmm tmm5, tmm3, tmm0	

	pop rbx
	pop rbp

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT6T2T1)
C_UNDERSCORE(MlasTdpAmxT6T2T1):

	push    rbp
        push    rbx

        #mov     r10,rdx
        #mov     r11,rsp                     # set ZeroMode to any non-zero value
	
	TdpbusdTmmTmmTmm tmm6, tmm2, tmm1	

	pop rbx
	pop rbp

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT7T3T1)
C_UNDERSCORE(MlasTdpAmxT7T3T1):

	push    rbp
        push    rbx

        #mov     r10,rdx
        #mov     r11,rsp                     # set ZeroMode to any non-zero value
	
	TdpbusdTmmTmmTmm tmm7, tmm3, tmm1	

	pop rbx
	pop rbp

	ret

	.end

