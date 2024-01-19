/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelLasx.s

Abstract:

    This module implements the kernels for the single precision pooling
    operation.

    This implementation uses Lasx instructions.

--*/

#include "asmmacro.h"
#include "SpoolKernelLasxCommon.h"

        .text

/*++

Macro Description:

    This macro generates code to initialize registers used across the kernel.

Arguments:

    PoolingType - Supplies the pooling type string.

Implicit Arguments:

    a5 - Supplies the ActualKernelSize parameter (see function description).

--*/

        .macro InitializeKernel PoolingType

.ifeqs "\PoolingType\()","Maximum"
	li.w	$s0, 0xFF7FFFFF
	xvreplgr2vr.w	$xr5, $s0
.else
	xvxor.v	$xr5, $xr5, $xr5
.ifeqs "\PoolingType\()","AverageExcludePad"
	move	$t6, $a6
	mul.d	$t6, $t6, $a7
    xvreplgr2vr.w   $xr5, $t6
.else
    xvreplgr2vr.w   $xr5, $a5
.endif
    xvffint.s.w  $xr5, $xr5
.endif

        .endm

/*++

Macro Description:

    This macro generates code to clear the pooling intermediates.

    For PoolingType==Maximum, the pooling intermediates are set to the minimum
    float value. Otherwise, the pooling intermediates are cleared to zero.

Arguments:

    PoolingType - Supplies the pooling type string.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a1 - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    xr0-xr2 - Supplies the pooling intermediates.

    xr5 - Supplies a vector containing the minimum float value broadcasted,
        if PoolingType==Maximum.

--*/

        .macro ClearBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
        EmitIfCountGE \OutputCount\(), 1, "xvor.v $xr0, $xr5, $xr5"
        EmitIfCountGE \OutputCount\(), 2, "xvor.v $xr1, $xr5, $xr5"
        EmitIfCountGE \OutputCount\(), 3, "xvor.v $xr2, $xr5, $xr5"
.else
        EmitIfCountGE \OutputCount\(), 1, "xvxor.v $xr0, $xr0, $xr0"
        EmitIfCountGE \OutputCount\(), 2, "xvxor.v $xr1, $xr1, $xr1"
        EmitIfCountGE \OutputCount\(), 3, "xvxor.v $xr2, $xr2, $xr2"
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
.if \OutputCount\() == 1
	xor	$a1, $a1, $a1                # reset valid block counter
.endif
.endif

        .endm

/*++

Macro Description:

    This macro generates code to sample the input buffer and update the pooling
    intermediates as appropriate.

Arguments:

    PoolingType - Supplies the pooling type string.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a3 - Supplies the address of the input buffer.

    a1 - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    a4 - Supplies the StrideWidth parameter (see function description).

    xr0-xr2 - Supplies the pooling intermediates.

--*/

        .macro ComputeBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
        EmitIfCountGE \OutputCount\(), 1, "xvld	$xr16, $a3, 0"
        EmitIfCountGE \OutputCount\(), 1, "xvfmax.s	$xr0, $xr0, $xr16"
        EmitIfCountGE \OutputCount\(), 2, "xvldx	$xr16, $a3, $a4"
        EmitIfCountGE \OutputCount\(), 2, "xvfmax.s	$xr1, $xr1, $xr16"
        EmitIfCountGE \OutputCount\(), 3, "slli.d	$s0, $a4, 1"
        EmitIfCountGE \OutputCount\(), 3, "xvldx	$xr16, $a3, $s0"
        EmitIfCountGE \OutputCount\(), 3, "xvfmax.s	$xr2, $xr2, $xr16"
.else
        EmitIfCountGE \OutputCount\(), 1, "xvld	$xr16, $a3, 0"
        EmitIfCountGE \OutputCount\(), 1, "xvfadd.s	$xr0, $xr0, $xr16"
        EmitIfCountGE \OutputCount\(), 2, "xvldx	$xr16, $a3, $a4"
        EmitIfCountGE \OutputCount\(), 2, "xvfadd.s	$xr1, $xr1, $xr16"
        EmitIfCountGE \OutputCount\(), 3, "slli.d	$s0, $a4, 1"
        EmitIfCountGE \OutputCount\(), 3, "xvldx	$xr16, $a3, $s0"
        EmitIfCountGE \OutputCount\(), 3, "xvfadd.s	$xr2, $xr2, $xr16"
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
.if \OutputCount\() == 1
	addi.d	$a1, $a1, 1                  # increment valid block counter
.endif
.endif

        .endm

/*++

Macro Description:

    This macro generates code to process and store the pooling intermediates.

Arguments:

    PoolingType - Supplies the pooling type string.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a2 - Supplies the address of the output buffer.

    a1 - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    xr0-xr2 - Supplies the pooling intermediates.

    xr5 - Supplies the kernel size computed by InitializeKernel, if
        PoolingType=AverageExcludePad, else the actual kernel size, if
        PoolingType=AverageIncludePad.

--*/

        .macro PostProcessBlock PoolingType, OutputCount

//
// If PoolingType=AverageExcludePad, divide the sum by the number of non-padding
// blocks. OutputCount=1 generates code to count the number of blocks accessed by
// ComputeBlock. Other cases use the kernel size computed by InitializeKernel.
//

.ifeqs "\PoolingType\()","AverageExcludePad"
.if \OutputCount\() == 1
	xvxor.v	$xr4, $xr4, $xr4
	xvreplgr2vr.w	$xr4, $a1
    xvffint.s.w  $xr4, $xr4
	xvfdiv.s	$xr0, $xr0, $xr4
.else
        EmitIfCountGE \OutputCount\(), 1, "xvfdiv.s $xr0, $xr0, $xr5"
        EmitIfCountGE \OutputCount\(), 2, "xvfdiv.s $xr1, $xr1, $xr5"
        EmitIfCountGE \OutputCount\(), 3, "xvfdiv.s $xr2, $xr2, $xr5"
.endif
.endif

//
// If PoolingType=AverageIncludePad, divide the sum by the actual kernel size.
//

.ifeqs "\PoolingType\()","AverageIncludePad"
        EmitIfCountGE \OutputCount\(), 1, "xvfdiv.s $xr0, $xr0, $xr5"
        EmitIfCountGE \OutputCount\(), 2, "xvfdiv.s $xr1, $xr1, $xr5"
        EmitIfCountGE \OutputCount\(), 3, "xvfdiv.s $xr2, $xr2, $xr5"
.endif

//
// Store the output block in the output buffer.
//

        EmitIfCountGE \OutputCount\(), 1, "xvst $xr0, $a2, 0"
        EmitIfCountGE \OutputCount\(), 2, "xvst $xr1, $a2, 0x20"
        EmitIfCountGE \OutputCount\(), 3, "xvst $xr2, $a2, 0x40"
        add_immed $a2,\OutputCount\()*8*4   # advance output by N nchw8c blocks

        .endm

//
// Generate the pooling kernels.
//

        SpoolKernelFunction Maximum, Lasx
        SpoolKernelFunction AverageExcludePad, Lasx
        SpoolKernelFunction AverageIncludePad, Lasx

        .end
