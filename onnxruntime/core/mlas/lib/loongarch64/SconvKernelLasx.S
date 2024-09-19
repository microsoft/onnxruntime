/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelLasx.S

Abstract:

    This module implements the kernels for the single precision convolution
    operation.

    This implementation uses Lasx instructions.

--*/

#include "asmmacro.h"
#include "SconvKernelLasxCommon.h"

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for FilterCount by OutputCount block
    of the output buffer.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

    VectorOffset - Supplies the byte offset from the filter buffer to fetch
        elements.

    BroadcastOffset - Supplies the byte offset from the input buffer to fetch
        elements.

Implicit Arguments:

    a3 - Supplies the address of the input buffer.

    a2 - Supplies the address of the filter buffer.

    a1 - Supplies the FilterStride parameter (see function description).

    t7 - Supplies the address of the filter buffer plus 2 * FilterStride.

    a5 - Supplies the StrideWidth parameter (see function description).

    xr0-xr7 - Supplies the block accumulators.

--*/

        .macro ComputeBlock KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

.ifeqs "\KernelType\()","Depthwise"
	xvld	$xr12, $a2, 0
        EmitIfCountGE \OutputCount\(), 1, "xvld $xr8, $a3, 0"
        EmitIfCountGE \OutputCount\(), 1, "xvfmadd.s $xr0, $xr8, $xr12, $xr0"
        EmitIfCountGE \OutputCount\(), 2, "xvldx $xr9, $a3, $a5"
        EmitIfCountGE \OutputCount\(), 2, "xvfmadd.s $xr4, $xr9, $xr12, $xr4"

.else
        EmitIfCountGE \OutputCount\(), 1, "xvldrepl.w $xr13, $a3, \BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 2, "add.d $s0, $a3, $a5"
        EmitIfCountGE \OutputCount\(), 2, "xvldrepl.w $xr14, $s0, \BroadcastOffset\()"
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "xvld $xr8, $a2, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 1, "xvfmadd.s $xr0, $xr8, $xr13, $xr0"
        EmitIfCountGE \FilterCount\(), 2, "add.d $s0, $a2, $a1"
        EmitIfCountGE \FilterCount\(), 2, "xvld $xr9, $s0, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 2, "xvfmadd.s $xr1, $xr9, $xr13, $xr1"
        EmitIfCountGE \FilterCount\(), 3, "xvld $xr10, $t7, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 3, "xvfmadd.s $xr2, $xr10, $xr13, $xr2"
        EmitIfCountGE \FilterCount\(), 4, "add.d $s0, $t7, $a1"
        EmitIfCountGE \FilterCount\(), 4, "xvld $xr11, $s0, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 4, "xvfmadd.s $xr3, $xr11, $xr13, $xr3"
.else
        EmitIfCountGE \FilterCount\(), 1, "xvld $xr12, $a2, \VectorOffset\()"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvfmadd.s $xr0, $xr12, $xr13, $xr0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvfmadd.s $xr4, $xr12, $xr14, $xr4"
        EmitIfCountGE \FilterCount\(), 2, "add.d $s0, $a2, $a1"
        EmitIfCountGE \FilterCount\(), 2, "xvld $xr12, $s0, \VectorOffset\()"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvfmadd.s $xr1, $xr13, $xr12, $xr1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvfmadd.s $xr5, $xr14, $xr12, $xr5"
        EmitIfCountGE \FilterCount\(), 3, "xvld $xr12, $t7, \VectorOffset\()"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvfmadd.s $xr2, $xr13, $xr12, $xr2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvfmadd.s $xr6, $xr14, $xr12, $xr6"
        EmitIfCountGE \FilterCount\(), 4, "add.d $s0, $t7, $a1"
        EmitIfCountGE \FilterCount\(), 4, "xvld $xr12, $s0, \VectorOffset\()"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvfmadd.s $xr3, $xr13, $xr12, $xr3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvfmadd.s $xr7, $xr14, $xr12, $xr7"
.endif
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    t7 - Supplies the DilationWidth parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    t5 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessFilterCountN KernelFrame, KernelType, FilterCount

//
// Process the output blocks that include left padding.
//

	ld.d	$t0, $sp, OutputCountLeftPad_arg
	beqz	$t0, .L\KernelType\().\FilterCount\().ProcessOutputCount
    bl    MlasConv\KernelType\()FloatSingleLasxFilter\FilterCount\()

//
// Process the output blocks that do not include any padding.
//

.L\KernelType\().\FilterCount\().ProcessOutputCount:
	ld.d	$t0, $sp, OutputCount_arg
    li.d    $s0, 2
    bltu	$t0, $s0, .L\KernelType\().\FilterCount\().ProcessRemainingOutputCount

.L\KernelType\().\FilterCount\().ProcessNextOutputCountBy2:
        ProcessOutputCountN Lasx, \KernelFrame\(), \KernelType\(), 8, \FilterCount\(), 2
	slli.d	$s0, $a5, 1              # advance input by 2 elements
	add.d	$a0, $a0, $s0
	addi.d	$t0, $t0, -2
    li.d    $s0, 2
	bgeu	$t0, $s0, .L\KernelType\().\FilterCount\().ProcessNextOutputCountBy2

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCount:

//
// Process the output blocks that include right padding plus any remaining output
// blocks from above.
//

.L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining:
	ld.d	$s0, $sp, OutputCountRightPad_arg
	add.d	$t0, $t0, $s0
	beqz	$t0, .L\KernelType\().ExitKernel
        bl	MlasConv\KernelType\()FloatSingleLasxFilter\FilterCount\()

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows for a pointwise convolution.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description).

    t8 - Supplies the InputStride parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    t0 - Supplies the OutputCount parameter (see function description).

    t2 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseFilterCountN FilterCount
        li.d    $s0, 2
        bltu	$t0, $s0, .LPointwise.\FilterCount\().ProcessRemainingOutputCount

.LPointwise.\FilterCount\().ProcessNextOutputCountBy2:
        ProcessPointwiseOutputCountN Lasx, 8, \FilterCount\(), 2
	slli.d	$s0, $a5, 1              # advance input by 2 elements
	add.d	$a0, $a0, $s0
	addi.d	$t0, $t0, -2
    li.d    $s0, 2
    bgeu	$t0, $s0, .LPointwise.\FilterCount\().ProcessNextOutputCountBy2

.LPointwise.\FilterCount\().ProcessRemainingOutputCount:
        beqz	$t0, .LPointwise.ExitKernel
        ProcessPointwiseOutputCountN Lasx, 8, \FilterCount\(), 1

        .endm

//
// Generate the convolution kernels.
//

        SconvKernelFunction Nchw, 8, Lasx
        SconvKernelFunction Nchwc, 8, Lasx, BiasFilter
        SconvKernelDepthwiseFunction 8, Lasx
        SconvKernelPointwiseFunction Lasx, BiasFilter

/*++

Macro Description:

    This macro generates code to process an output block after the inner
    convolution kernel has executed and then stores the output block to the
    output buffer.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

--*/

        .macro PostProcessBlock FilterCount, OutputCount

        .globl  MlasConvPostProcessFloatLasxFilter\FilterCount\()Output\OutputCount\()
        .hidden MlasConvPostProcessFloatLasxFilter\FilterCount\()Output\OutputCount\()
MlasConvPostProcessFloatLasxFilter\FilterCount\()Output\OutputCount\():

        .globl  MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\()
        .hidden MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\()
MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\():

.if \FilterCount\() > 2
	slli.d	$s0, $t6, 1              # compute output plus 2 rows
	add.d	$t7, $a4, $s0
.endif

//
// Test if the existing contents of the output buffer should be accumulated
// with the output block.
//

	andi	$s0, $a2, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        beqz	$s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvld $xr16, $a4, 0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvfadd.s $xr0, $xr0, $xr16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvld $xr16, $a4, 32"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvfadd.s $xr4, $xr4, $xr16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvld $xr16, $a4, 0x40"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvfadd.s $xr8, $xr8, $xr16"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvldx $xr16, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvfadd.s $xr1, $xr1, $xr16"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "add.d $s0, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvld $xr16, $s0, 0x20"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvfadd.s $xr5, $xr5, $xr16"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "add.d $s0, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvld $xr16, $s0, 0x40"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvfadd.s $xr9, $xr9, $xr16"

        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvld $xr16,$t7, 0"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvfadd.s $xr2, $xr2, $xr16"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvld $xr16,$t7, 0x20"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvfadd.s $xr6, $xr6, $xr16"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvld $xr16,$t7, 0x40"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvfadd.s $xr10, $xr10, $xr16"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvldx $xr16,$t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvfadd.s $xr3, $xr3, $xr16"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "add.d $s0, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvld $xr16,$s0, 0x20"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvfadd.s $xr7, $xr7, $xr16"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "add.d $s0, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvld $xr16,$s0, 0x40"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvfadd.s $xr11, $xr11, $xr16"


.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput:

//
// Test if the bias buffer should be accumulated with the output block.
//

	andi	$s0, $a2, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        beqz	$s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "xvld $xr16, $a3, 0"
        EmitIfCountGE \FilterCount\(), 1, "xvfadd.s $xr0, $xr0, $xr16"
        EmitIfCountGE \FilterCount\(), 2, "xvld $xr16, $a3, 0x20"
        EmitIfCountGE \FilterCount\(), 2, "xvfadd.s $xr1, $xr1, $xr16"
        EmitIfCountGE \FilterCount\(), 3, "xvld $xr16, $a3, 0x40"
        EmitIfCountGE \FilterCount\(), 3, "xvfadd.s $xr2, $xr2, $xr16"
        EmitIfCountGE \FilterCount\(), 4, "xvld $xr16, $a3, 0x60"
        EmitIfCountGE \FilterCount\(), 4, "xvfadd.s $xr3, $xr3, $xr16"
.else
        EmitIfCountGE \FilterCount\(), 1, "xvld $xr12, $a3, 0"
        EmitIfCountGE \FilterCount\(), 2, "xvld $xr13, $a3, 0x20"
        EmitIfCountGE \FilterCount\(), 3, "xvld $xr14, $a3, 0x40"
        EmitIfCountGE \FilterCount\(), 4, "xvld $xr15, $a3, 0x60"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvfadd.s $xr0, $xr0, $xr12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvfadd.s $xr4, $xr4, $xr12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvfadd.s $xr8, $xr8, $xr12"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvfadd.s $xr1, $xr1, $xr13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvfadd.s $xr5, $xr5, $xr13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvfadd.s $xr9, $xr9, $xr13"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvfadd.s $xr2, $xr2, $xr14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvfadd.s $xr6, $xr6, $xr14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvfadd.s $xr10, $xr10, $xr14"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvfadd.s $xr3, $xr3, $xr15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvfadd.s $xr7, $xr7, $xr15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvfadd.s $xr11, $xr11, $xr15"

.endif

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

	andi	$s0, $a2, MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        beqz	$s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation
	xvxor.v	$xr15, $xr15, $xr15
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvfmax.s $xr0, $xr15, $xr0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvfmax.s $xr4, $xr15, $xr4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvfmax.s $xr8, $xr15, $xr8"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvfmax.s $xr1, $xr15, $xr1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvfmax.s $xr5, $xr15, $xr5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvfmax.s $xr9, $xr15, $xr9"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvfmax.s $xr2, $xr15, $xr2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvfmax.s $xr6, $xr15, $xr6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvfmax.s $xr10, $xr15, $xr10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvfmax.s $xr3, $xr15, $xr3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvfmax.s $xr7, $xr15, $xr7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvfmax.s $xr11, $xr15, $xr11"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation:

//
// Store the output block in the output buffer.
//
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvst $xr0, $a4, 0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvst $xr4, $a4, 0x20"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvst $xr8, $a4, 0x40"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvstx $xr1, $a4, $t6"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "add.d $s0, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvst $xr5, $s0, 0x20"

        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "add.d $s0, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvst $xr9, $s0, 0x40"

        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvst $xr2, $t7, 0"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvst $xr6, $t7, 0x20"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvst $xr10, $t7, 0x40"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvstx $xr3, $t7, $t6"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "add.d $s0, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvst $xr7, $s0, 0x20"

        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "add.d $s0, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvst $xr11, $s0, 0x40"

        add_immed $a4,\OutputCount\()*8*4    # advance output by N nchw8c blocks
	jr	$ra

        .endm

        .irp    FilterCount, 1, 2, 3, 4
        .irp    OutputCount, 1, 2, 3
            PostProcessBlock \FilterCount\(), \OutputCount\()
        .endr
        .endr

        .end
