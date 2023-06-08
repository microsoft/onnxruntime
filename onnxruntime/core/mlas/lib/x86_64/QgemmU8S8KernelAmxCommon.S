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

	TdpbusdTmmTmmTmm tmm4, tmm2, tmm0	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT5T3T0)
C_UNDERSCORE(MlasTdpAmxT5T3T0):

	TdpbusdTmmTmmTmm tmm5, tmm3, tmm0	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT6T2T1)
C_UNDERSCORE(MlasTdpAmxT6T2T1):

	TdpbusdTmmTmmTmm tmm6, tmm2, tmm1	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT7T3T1)
C_UNDERSCORE(MlasTdpAmxT7T3T1):

	TdpbusdTmmTmmTmm tmm7, tmm3, tmm1	

	ret


.globl  C_UNDERSCORE(MlasLoadAmxT0)
C_UNDERSCORE(MlasLoadAmxT0):

	TileloaddTmmMem tmm0, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT1)
C_UNDERSCORE(MlasLoadAmxT1):

	TileloaddTmmMem tmm1, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT2)
C_UNDERSCORE(MlasLoadAmxT2):

	TileloaddTmmMem tmm2, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT3)
C_UNDERSCORE(MlasLoadAmxT3):

	TileloaddTmmMem tmm3, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT4)
C_UNDERSCORE(MlasLoadAmxT4):

	TileloaddTmmMem tmm4, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT5)
C_UNDERSCORE(MlasLoadAmxT5):

	TileloaddTmmMem tmm5, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT6)
C_UNDERSCORE(MlasLoadAmxT6):

	TileloaddTmmMem tmm6, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT7)
C_UNDERSCORE(MlasLoadAmxT7):

	TileloaddTmmMem tmm7, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT0)
C_UNDERSCORE(MlasStoreAmxT0):

	TileStoredMemTmm tmm0, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT1)
C_UNDERSCORE(MlasStoreAmxT1):

	TileStoredMemTmm tmm1, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT2)
C_UNDERSCORE(MlasStoreAmxT2):

	TileStoredMemTmm tmm2, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT3)
C_UNDERSCORE(MlasStoreAmxT3):

	TileStoredMemTmm tmm3, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT4)
C_UNDERSCORE(MlasStoreAmxT4):

	TileStoredMemTmm tmm4, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT5)
C_UNDERSCORE(MlasStoreAmxT5):

	TileStoredMemTmm tmm5, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT6)
C_UNDERSCORE(MlasStoreAmxT6):

	TileStoredMemTmm tmm6, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT7)
C_UNDERSCORE(MlasStoreAmxT7):

	TileStoredMemTmm tmm7, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadConfigAmx)
C_UNDERSCORE(MlasLoadConfigAmx):

	ldtilecfgMacro rdi

	ret

.globl  C_UNDERSCORE(MlasStoreConfigAmx)
C_UNDERSCORE(MlasStoreConfigAmx):

	sttilecfgMacro rdi

	ret

	.end

