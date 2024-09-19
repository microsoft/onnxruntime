/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelasxCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision pooling operation for the Lasx kernels.

--*/

//
// Stack frame layout for the pooling kernels.
//

#define SP_SIZE 8*8
#define InputBase_arg                   SP_SIZE+0*8
#define InputWidth_arg                  SP_SIZE+1*8
#define DilatedInputWidth_arg           SP_SIZE+2*8
#define OutputCountLeftPad_arg          SP_SIZE+3*8
#define OutputCount_arg                 SP_SIZE+4*8
#define OutputCountRightPad_arg         SP_SIZE+5*8
/*++

Macro Description:

    This macro generates the common prologue code for the pooling kernels.

Arguments:

    PoolingType - Supplies the pooling type string.

--*/

        .macro SpoolKernelEntry PoolingType

	addi.d	$sp, $sp, -SP_SIZE
	st.d	$s0, $sp, 0
	st.d	$s1, $sp, 1*8
    fst.d   $f16, $sp, 2*8
	st.d	$ra, $sp, 5*8

        InitializeKernel \PoolingType\()
	move	$t8, $a4
	move	$a4, $a2
	move	$a5, $a3
	move	$a2, $a1

        .endm

/*++

Macro Description:

    This macro generates the common epilogue code for the pooling kernels.

Arguments:

    None.

--*/

        .macro SpoolKernelExit

	ld.d	$s0, $sp, 0
	ld.d	$s1, $sp,  1*8
    fld.d   $f16, $sp, 2*8
	ld.d	$ra, $sp, 5*8
	addi.d	$sp, $sp, SP_SIZE
	jr	$ra

        .endm

/*++

Macro Description:

    This macro generates code to compute pooling for a vector of input blocks
    to produce a matrix of output blocks.

    OutputCount=1 generates special case code to handle padding blocks. All
    other output counts assume no padding.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a2 - Supplies the address of the output buffer.

    a4 - Supplies the StrideWidth parameter (see function description).

    a5 - Supplies the DilationWidth parameter (see function description).

    t8 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessOutputCountN KernelFrame, PoolingType, OutputCount

	move	$a3, $a0
	move	$t1, $a6
	move	$t2, $a7
.if \OutputCount\() == 1
	ld.d	$t3, $sp, InputBase_arg
	ld.d	$t4, $sp, InputWidth_arg
	sub.d	$t3, $zero, $t3
.endif
        ClearBlock \PoolingType\(), \OutputCount\()
        beqz	$t1, .L\PoolingType\().\OutputCount\().HandlePostProcessing

.L\PoolingType\().\OutputCount\().ProcessNextRow:
	move	$t6, $t2

.L\PoolingType\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
	add.d	$t7, $a3, $t3               # compute (Input - InputBase)
        # (Input - InputBase) >= InputWidth?
        bgeu	$t7, $t4, .L\PoolingType\().\OutputCount\().SkipOverPadding
.endif
        ComputeBlock \PoolingType\(), \OutputCount\()

.L\PoolingType\().\OutputCount\().SkipOverPadding:
	add.d	$a3, $a3, $a5                # advance input by dilation width
	addi.d	$t6, $t6, -1                 # decrement columns remaining
        bnez	$t6, .L\PoolingType\().\OutputCount\().ProcessNextColumn
	add.d	$a3, $a3, $t8                # advance input to next row
.if \OutputCount\() == 1
	ld.d	$s0, $sp, DilatedInputWidth_arg
	sub.d	$t3, $t3, $s0
                                            # advance input base to next row
.endif
	addi.d	$t1, $t1, -1
        bnez	$t1, .L\PoolingType\().\OutputCount\().ProcessNextRow

.L\PoolingType\().\OutputCount\().HandlePostProcessing:
        PostProcessBlock \PoolingType\(), \OutputCount\()

        .endm
/*++

Macro Description:

    This macro generates code for the inner pooling kernel.

Arguments:

    PoolingType - Supplies the pooling type string.

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro SpoolKernelFunction PoolingType, Isa

/*++

Routine Description:

    This routine is the inner kernel to compute pooling for the elements of an
    output row for a set of filter rows.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Output (a1) - Supplies the address of the output buffer.

    StrideWidth (a2) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (a3) - Supplies the length in bytes of the blocked dilation
        width.

    InputStride (a4) - Supplies the length in bytes to advance the input buffer to
        the next input row.

    ActualKernelSize (a5) - Supplies the size of the kernel based on the original
        kernel dimensions, used for PoolingType=AverageIncludePad.

    KernelHeight (a6) - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth (a7)- Supplies the width of the kernel to apply.

    InputBase (sp + 0)- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (sp + 0x8)- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (sp + 0x10)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (sp + 0x18)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (sp + 0x20)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (sp + 0x28)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasPool\PoolingType\()FloatKernel\Isa\()

        SpoolKernelEntry \PoolingType\()

.L\PoolingType\().ProcessOutputCountLeftPad:
	ld.d	$t0, $sp, OutputCountLeftPad_arg

        beqz	$t0, .L\PoolingType\().ProcessOutputCount
        bl    MlasPool\PoolingType\()FloatSingle\Isa\()

.L\PoolingType\().ProcessOutputCount:
	ld.d	$t0, $sp, OutputCount_arg
    li.d    $s0, 3
    bltu	$t0, $s0, .L\PoolingType\().ProcessRemainingOutputCount

.L\PoolingType\().ProcessNextOutputCountBy3:
        ProcessOutputCountN .LSpoolKernelFrame, \PoolingType\(), 3
	slli.d	$s0, $a4, 1
	add.d	$t6, $s0, $a4
	add.d	$a0, $a0, $t6                # advance input by 3 elements
	addi.d	$t0, $t0, -3
    li.d    $s0, 3
    bgeu	$t0, $s0, .L\PoolingType\().ProcessNextOutputCountBy3

.L\PoolingType\().ProcessRemainingOutputCount:

.L\PoolingType\().ProcessOutputCountRightPad:
	ld.d	$s0, $sp, OutputCountRightPad_arg
	add.d	$t0, $t0, $s0
        beqz	$t0, .L\PoolingType\().ExitKernel
        bl    MlasPool\PoolingType\()FloatSingle\Isa\()

.L\PoolingType\().ExitKernel:
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
        SpoolKernelExit

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

MlasPool\PoolingType\()FloatSingle\Isa\():
	st.d	$ra, $sp, 6*8
loopMlasPool\PoolingType\()FloatSingle\Isa\():
        ProcessOutputCountN .LSpoolKernelSingleFrame, \PoolingType\(), 1
	add.d	$a0, $a0, $a4                # advance input by 1 element
	addi.d	$t0, $t0, -1                 # decrement output count remaining
        bnez	$t0, loopMlasPool\PoolingType\()FloatSingle\Isa\()
	ld.d	$ra, $sp, 6*8
	jr	$ra

        .endm
