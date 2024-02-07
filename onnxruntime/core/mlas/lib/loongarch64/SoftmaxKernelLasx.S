/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SoftmaxKernelLasx.s

Abstract:

    This module implements the kernels for the single precision softmax
    operation.

    This implementation uses Lasx instructions.

--*/

#include "asmmacro.h"

        .text

/*++

Routine Description:

    This routine implements a vectorized kernel to find the maximum value of
    the supplied buffer.

Arguments:

    Input (a0) - Supplies the input buffer.

    N (a1) - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/

        FUNCTION_ENTRY MlasReduceMaximumF32KernelLasx
	addi.d	$sp, $sp, -32

	la.global	$t0, MlasMinimumF32Value
	ld.w	$t0, $t0, 0
	xvreplgr2vr.w	$xr0, $t0
	beqz	$a1, .LReduceMaximum.ExitKernel
	ori	$t0, $zero, 8
	bltu	$a1, $t0, .LReduceMaximum.ProcessRemainingCountBy1
	ori	$t1, $zero, 32
	bltu	$a1, $t1, .LReduceMaximum.ProcessRemainingCountBy8
	xvreplgr2vr.w	$xr16, $zero
	xvor.v	$xr1, $xr0, $xr16
	xvor.v	$xr2, $xr0, $xr16
	xvor.v	$xr3, $xr0, $xr16

.LReduceMaximum.ProcessRemainingCountBy32:
	xvld	$xr16, $a0, 0
	xvfmax.s	$xr0, $xr0, $xr16
	xvld	$xr16, $a0, 8*4
	xvfmax.s	$xr1, $xr1, $xr16
	addi.d	$a1, $a1, -0x20
	xvld	$xr16, $a0, 16*4
	xvfmax.s	$xr2, $xr2, $xr16
	xvld	$xr16, $a0, 24*4
	xvfmax.s	$xr3, $xr3, $xr16
	addi.d	$a0, $a0, 32*4                # advance input by 32 elements
	ori	$t1, $zero, 32
	bgeu	$a1, $t1, .LReduceMaximum.ProcessRemainingCountBy32
	xvfmax.s	$xr0, $xr0, $xr1
	xvfmax.s	$xr2, $xr2, $xr3
	xvfmax.s	$xr0, $xr0, $xr2

.LReduceMaximum.ProcessRemainingCountBy8:
	ori	$t1, $zero, 8
	bltu	$a1, $t1, .LReduceMaximum.ProcessRemainingCountLessThan8
	xvld	$xr16, $a0, 0
	xvfmax.s	$xr0, $xr0, $xr16
	addi.d	$a1, $a1, -8
	addi.d	$a0, $a0, 8*4
    b	.LReduceMaximum.ProcessRemainingCountBy8

.LReduceMaximum.ProcessRemainingCountLessThan8:
	xvst	$xr0, $sp, 0
	vld	$vr1, $sp, 0x10
	vld	$vr0, $sp, 0
	vfmax.s	$vr0, $vr0, $vr1
	vshuf4i.w	$vr1, $vr0, 0xee
	vfmax.s	$vr0, $vr0, $vr1
	vshuf4i.w	$vr1, $vr0, 0x55
	vfmax.s	$vr0, $vr0, $vr1
	beqz	$a1, .LReduceMaximum.ExitKernel

.LReduceMaximum.ProcessRemainingCountBy1:
	vld	$vr16, $a0, 0
	vfmax.s	$vr0, $vr0, $vr16
	addi.d	$a0, $a0, 4                     # advance input by 1 element
	addi.d	$a1, $a1, -1
        bnez	$a1, .LReduceMaximum.ProcessRemainingCountBy1

.LReduceMaximum.ExitKernel:
	xvinsgr2vr.d	$xr0, $zero, 2
	xvinsgr2vr.d	$xr0, $zero, 3
	xvinsgr2vr.d	$xr1, $zero, 2
	xvinsgr2vr.d	$xr1, $zero, 3
	xvinsgr2vr.d	$xr2, $zero, 2
	xvinsgr2vr.d	$xr2, $zero, 3
	xvinsgr2vr.d	$xr3, $zero, 2
	xvinsgr2vr.d	$xr3, $zero, 3
	xvinsgr2vr.d	$xr4, $zero, 2
	xvinsgr2vr.d	$xr4, $zero, 3
	xvinsgr2vr.d	$xr5, $zero, 2
	xvinsgr2vr.d	$xr5, $zero, 3
	xvinsgr2vr.d	$xr6, $zero, 2
	xvinsgr2vr.d	$xr6, $zero, 3
	xvinsgr2vr.d	$xr7, $zero, 2
	xvinsgr2vr.d	$xr7, $zero, 3
	xvinsgr2vr.d	$xr8, $zero, 2
	xvinsgr2vr.d	$xr8, $zero, 3
	xvinsgr2vr.d	$xr9, $zero, 2
	xvinsgr2vr.d	$xr9, $zero, 3
	xvinsgr2vr.d	$xr10, $zero, 2
	xvinsgr2vr.d	$xr10, $zero, 3
	xvinsgr2vr.d	$xr11, $zero, 2
	xvinsgr2vr.d	$xr11, $zero, 3
	xvinsgr2vr.d	$xr12, $zero, 2
	xvinsgr2vr.d	$xr12, $zero, 3
	xvinsgr2vr.d	$xr13, $zero, 2
	xvinsgr2vr.d	$xr13, $zero, 3
	xvinsgr2vr.d	$xr14, $zero, 2
	xvinsgr2vr.d	$xr14, $zero, 3
	xvinsgr2vr.d	$xr15, $zero, 2
	xvinsgr2vr.d	$xr15, $zero, 3
	addi.d	$sp, $sp, 32
	jr	$ra

/*++

Routine Description:

    This routine implements a vectorized kernel to produce the final output for
    the softmax operation.

Arguments:

    Output (a0) - Supplies the output buffer.

    N (a1) - Supplies the number of elements to process.

    Parameters (a2) - Supplies an array containing the scale value.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeSoftmaxOutputF32KernelLasx

	ld.w	$t0, $a2, 0
	xvreplgr2vr.w	$xr4, $t0
	ori	$t1, $zero, 0x20
	bltu	$a1, $t1, .LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountBy32:
	xvld	$xr16, $a0, 0
	xvfmul.s	$xr0, $xr4, $xr16
	xvld	$xr16, $a0, 8*4
	xvfmul.s	$xr1, $xr4, $xr16
	addi.d	$a1, $a1, -0x20
	xvld	$xr16, $a0, 16*4
	xvfmul.s	$xr2, $xr4, $xr16
	xvld	$xr16, $a0, 24*4
	xvfmul.s	$xr3, $xr4, $xr16
	xvst	$xr0, $a0, 0
	xvst	$xr1, $a0, 8*4
	xvst	$xr2, $a0, 16*4
	xvst	$xr3, $a0, 24*4
	addi.d	$a0, $a0, 0x80                   # advance output by 32 elements
	bgeu	$a1, $t1, .LComputeSoftmaxOutput.ProcessRemainingCountBy32

.LComputeSoftmaxOutput.ProcessRemainingCountBy8:
	ori	$t2, $zero, 8
	bltu	$a1, $t2, .LComputeSoftmaxOutput.ProcessRemainingCountLessThan8
	xvld	$xr16, $a0, 0
	xvfmul.s	$xr0, $xr4, $xr16
	addi.d	$a1, $a1, -8
	xvst	$xr0, $a0, 0
	addi.d	$a0, $a0, 8*4                   # advance output by 8 elements
        b	.LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountLessThan8:
	beqz	$a1, .LComputeSoftmaxOutput.ExitKernel

.LComputeSoftmaxOutput.ProcessRemainingCountBy1:
    fld.s   $f16, $a0, 0
    fmul.s  $f0, $f4, $f16
    fst.s   $f0, $a0, 0
	addi.d	$a0, $a0, 4                      # advance output by 1 element
	addi.d	$a1, $a1, -1
        bnez	$a1, .LComputeSoftmaxOutput.ProcessRemainingCountBy1

.LComputeSoftmaxOutput.ExitKernel:
	xvinsgr2vr.d	$xr0, $zero, 2
	xvinsgr2vr.d	$xr0, $zero, 3
	xvinsgr2vr.d	$xr1, $zero, 2
	xvinsgr2vr.d	$xr1, $zero, 3
	xvinsgr2vr.d	$xr2, $zero, 2
	xvinsgr2vr.d	$xr2, $zero, 3
	xvinsgr2vr.d	$xr3, $zero, 2
	xvinsgr2vr.d	$xr3, $zero, 3
	xvinsgr2vr.d	$xr4, $zero, 2
	xvinsgr2vr.d	$xr4, $zero, 3
	xvinsgr2vr.d	$xr5, $zero, 2
	xvinsgr2vr.d	$xr5, $zero, 3
	xvinsgr2vr.d	$xr6, $zero, 2
	xvinsgr2vr.d	$xr6, $zero, 3
	xvinsgr2vr.d	$xr7, $zero, 2
	xvinsgr2vr.d	$xr7, $zero, 3
	xvinsgr2vr.d	$xr8, $zero, 2
	xvinsgr2vr.d	$xr8, $zero, 3
	xvinsgr2vr.d	$xr9, $zero, 2
	xvinsgr2vr.d	$xr9, $zero, 3
	xvinsgr2vr.d	$xr10, $zero, 2
	xvinsgr2vr.d	$xr10, $zero, 3
	xvinsgr2vr.d	$xr11, $zero, 2
	xvinsgr2vr.d	$xr11, $zero, 3
	xvinsgr2vr.d	$xr12, $zero, 2
	xvinsgr2vr.d	$xr12, $zero, 3
	xvinsgr2vr.d	$xr13, $zero, 2
	xvinsgr2vr.d	$xr13, $zero, 3
	xvinsgr2vr.d	$xr14, $zero, 2
	xvinsgr2vr.d	$xr14, $zero, 3
	xvinsgr2vr.d	$xr15, $zero, 2
	xvinsgr2vr.d	$xr15, $zero, 3
	jr	$ra

/*++

Routine Description:

    This routine implements a vectorized kernel to produce the final output for
    the log softmax operation.

Arguments:

    Input (a0) - Supplies the output buffer.

    Output (a1) - Supplies the output buffer.

    N (a2) - Supplies the number of elements to process.

    Parameters (a3) - Supplies an array containing the negative maximum and
        logarithm values.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeLogSoftmaxOutputF32KernelLasx

	ld.w	$t0, $a3, 0
	ld.w	$t1, $a3, 4
	ori	$t2, $zero, 0x20
	xvreplgr2vr.w	$xr4, $t0       # broadcast negative minimum value
	xvreplgr2vr.w	$xr5, $t1     # broadcast log(SumExp)
        bltu	$a2, $t2, .LComputeLogSoftmaxOutput.ProcessRemainingCountBy8

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy32:
	xvld	$xr16, $a0, 0
	xvfadd.s	$xr0, $xr4, $xr16
	xvld	$xr16, $a0, 0x20
	xvfadd.s	$xr1, $xr4, $xr16
	addi.d	$a2, $a2, -0x20
	xvld	$xr16, $a0, 0x40
	xvfadd.s	$xr2, $xr4, $xr16
	xvld	$xr16, $a0, 0x60
	xvfadd.s	$xr3, $xr4, $xr16
	addi.d	$a0, $a0, 0x80                   # advance input by 32 elements
	xvfsub.s	$xr0, $xr0, $xr5         # do as two steps for numeric stability
	xvfsub.s	$xr1, $xr1, $xr5         # do as two steps for numeric stability
	xvfsub.s	$xr2, $xr2, $xr5         # do as two steps for numeric stability
	xvfsub.s	$xr3, $xr3, $xr5         # do as two steps for numeric stability
	xvst	$xr0, $a1, 0
	xvst	$xr1, $a1, 0x20
	xvst	$xr2, $a1, 0x40
	xvst	$xr3, $a1, 0x60
	addi.d	$a1, $a1, 0x80                   # advance output by 32 elements
	bgeu	$a2, $t2, .LComputeLogSoftmaxOutput.ProcessRemainingCountBy32

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy8:
	ori	$t3, $zero, 8
	bltu	$a2, $t3, .LComputeLogSoftmaxOutput.ProcessRemainingCountLessThan8
	xvld	$xr16, $a0, 0
	xvfadd.s	$xr0, $xr4, $xr16
	addi.d	$a0, $a0, 0x20
	xvfsub.s	$xr0, $xr0, $xr5
	addi.d	$a2, $a2, -8
	xvst	$xr0, $a1, 0
	addi.d	$a1, $a1, 0x20                   # advance output by 8 elements
        b	.LComputeLogSoftmaxOutput.ProcessRemainingCountBy8

.LComputeLogSoftmaxOutput.ProcessRemainingCountLessThan8:
        beqz	$a2, .LComputeLogSoftmaxOutput.ExitKernel

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy1:
    fld.s   $f16, $a0, 0
    fadd.s  $f0, $f4, $f16

	addi.d	$a0, $a0, 4
    fsub.s  $f0, $f0, $f5
    fst.s   $f0, $a1, 0

	addi.d	$a1, $a1, 4
	addi.d	$a2, $a2, -1
        bnez	$a2, .LComputeLogSoftmaxOutput.ProcessRemainingCountBy1

.LComputeLogSoftmaxOutput.ExitKernel:
	xvinsgr2vr.d	$xr0, $zero, 2
	xvinsgr2vr.d	$xr0, $zero, 3
	xvinsgr2vr.d	$xr1, $zero, 2
	xvinsgr2vr.d	$xr1, $zero, 3
	xvinsgr2vr.d	$xr2, $zero, 2
	xvinsgr2vr.d	$xr2, $zero, 3
	xvinsgr2vr.d	$xr3, $zero, 2
	xvinsgr2vr.d	$xr3, $zero, 3
	xvinsgr2vr.d	$xr4, $zero, 2
	xvinsgr2vr.d	$xr4, $zero, 3
	xvinsgr2vr.d	$xr5, $zero, 2
	xvinsgr2vr.d	$xr5, $zero, 3
	xvinsgr2vr.d	$xr6, $zero, 2
	xvinsgr2vr.d	$xr6, $zero, 3
	xvinsgr2vr.d	$xr7, $zero, 2
	xvinsgr2vr.d	$xr7, $zero, 3
	xvinsgr2vr.d	$xr8, $zero, 2
	xvinsgr2vr.d	$xr8, $zero, 3
	xvinsgr2vr.d	$xr9, $zero, 2
	xvinsgr2vr.d	$xr9, $zero, 3
	xvinsgr2vr.d	$xr10, $zero, 2
	xvinsgr2vr.d	$xr10, $zero, 3
	xvinsgr2vr.d	$xr11, $zero, 2
	xvinsgr2vr.d	$xr11, $zero, 3
	xvinsgr2vr.d	$xr12, $zero, 2
	xvinsgr2vr.d	$xr12, $zero, 3
	xvinsgr2vr.d	$xr13, $zero, 2
	xvinsgr2vr.d	$xr13, $zero, 3
	xvinsgr2vr.d	$xr14, $zero, 2
	xvinsgr2vr.d	$xr14, $zero, 3
	xvinsgr2vr.d	$xr15, $zero, 2
	xvinsgr2vr.d	$xr15, $zero, 3
	jr	$ra

        .end
