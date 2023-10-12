/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelLasxCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision convolution operation for the Lasx kernels.

--*/


#define SP_SIZE 32*8

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

#define OutputStride_arg                6*8
#define KernelHeight_arg                7*8
#define KernelWidth_arg                 8*8
#define InputBase_arg                   9*8
#define InputWidth_arg                  10*8
#define DilatedInputWidth_arg           11*8
#define OutputCountLeftPad_arg          12*8
#define OutputCount_arg                 13*8
#define OutputCountRightPad_arg         14*8
#define Bias_arg                        15*8
#define Flags_arg                       16*8
#define InputChannels_arg               17*8
#define Filter_save_offset 18*8

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks.

    OutputCount=1 generates special case code to handle padding blocks. All
    other output counts assume no padding.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    s8 - Supplies the DilationWidth parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    t5 - Supplies the InputStride parameter (see function description).
--*/
        .macro ProcessOutputCountN Isa, KernelFrame, KernelType, BlockSize, FilterCount, OutputCount

	move	$a3, $a0
.ifeqs "\KernelType\()","Depthwise"
	move	$a2, $a1
.else
	ld.d	$a2, $sp, Filter_save_offset
.endif
	ld.d	$t1, $sp, KernelHeight_arg
	ld.d	$t2, $sp, KernelWidth_arg
.if \OutputCount\() == 1
	ld.d	$t3, $sp, InputBase_arg
	ld.d	$t4, $sp, InputWidth_arg
	sub.d	$t3, $zero, $t3
.endif
        ClearBlock \FilterCount\(), \OutputCount\()
        beqz	$t1, .L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow:
	move	$t6, $t2                    # reload kernel width remaining

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
	add.d	$t7, $a3, $t3               # compute (Input - InputBase)
        # (Input - InputBase) >= InputWidth?
        bgeu	$t7, $t4, .L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding
.endif
.if \OutputCount\() > 3
	slli.d	$s0, $a5, 1
	add.d	$s0, $s0, $a5
	add.d	$t4, $a3, $s0                # compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
	slli.d	$s0, $a1, 1             # compute filter plus 2 rows
	add.d	$t7, $a2, $s0
.endif
.ifeqs "\KernelType\()","Nchwc"
.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
.else
        ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), 0, 0
.endif

.L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding:
        # advance input by dilation width
	add.d	$a3, $a3, $t8
.ifeqs "\KernelType\()","Nchwc"
       # advance filter by 8i8o/16i16o block
	addi.d	$a2, $a2, \BlockSize\()*\BlockSize\()*4
.else
	addi.d	$a2, $a2, \BlockSize\()*4    # advance filter by 8o/16o block
.endif
	addi.d	$t6, $t6, -1
        bnez	$t6, .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn
	add.d	$a3, $a3, $t5                # advance input to next row
.if \OutputCount\() == 1
	ld.d	$s0, $sp, DilatedInputWidth_arg
        # advance input base to next row
	sub.d	$t3, $t3, $s0
.endif
	addi.d	$t1, $t1, -1                 # decrement rows remaining
        bnez	$t1, .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow

//
// Handle post processing of the output block.
//

.L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing:
	ld.w	$a2, $sp, Flags_arg
.if \FilterCount\() > 1
	ld.d	$t6, $sp, OutputStride_arg
.endif
	ld.d	$a3, $sp, Bias_arg
        bl    MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

    BiasFilter - Supplies a non-blank value if the address of the filter buffer
        should be biased to point to the middle of a OIhw8i8o block in order to
        reduce the code size from relative byte offsets.

--*/

        .macro SconvKernelFunction KernelType, BlockSize, Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (a1) - Supplies the address of the filter buffer.

    Output (a2) - Supplies the address of the output buffer.

    StrideWidth (a3) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (a4) - Supplies the length in bytes of the blocked dilation
        width.

    FilterCount (a5) - Supplies the number of filters to process in this
        iteration.

    InputStride (a6)- Supplies the length in bytes to advance the input buffer to
        the next input row.

    FilterStride (a7) - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride (sp + 0)- Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    KernelHeight (sp + 8)- Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth (sp + 0x10)- Supplies the width of the kernel to apply.

    InputBase (sp + 0x18)- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (sp + 0x20)- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (sp + 0x28)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (sp + 0x30)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (sp + 0x38)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (sp + 0x40)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias (sp + 0x48)- Supplies the address of the bias buffer.

    Flags (sp + 0x50)- Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConv\KernelType\()FloatKernel\Isa\()

	addi.d	$sp, $sp, -SP_SIZE
	st.d	$s0, $sp, 0
	st.d	$s1, $sp, 8
	st.d	$s2, $sp, 2*8
	st.d	$ra, $sp, 5*8

    ld.d    $t0, $sp, SP_SIZE+0*8
    ld.d    $t1, $sp, SP_SIZE+1*8
    ld.d    $t2, $sp, SP_SIZE+2*8
    ld.d    $t3, $sp, SP_SIZE+3*8
    st.d    $t0, $sp, OutputStride_arg
    st.d    $t1, $sp, KernelHeight_arg
    st.d    $t2, $sp, KernelWidth_arg
    st.d    $t3, $sp, InputBase_arg
    ld.d    $t0, $sp, SP_SIZE+4*8
    ld.d    $t1, $sp, SP_SIZE+5*8
    ld.d    $t2, $sp, SP_SIZE+6*8
    ld.d    $t3, $sp, SP_SIZE+7*8
    st.d    $t0, $sp, InputWidth_arg
    st.d    $t1, $sp, DilatedInputWidth_arg
    st.d    $t2, $sp, OutputCountLeftPad_arg
    st.d    $t3, $sp, OutputCount_arg
    ld.d    $t0, $sp, SP_SIZE+8*8
    ld.d    $t1, $sp, SP_SIZE+9*8
    ld.d    $t2, $sp, SP_SIZE+10*8
    st.d    $t0, $sp, OutputCountRightPad_arg
    st.d    $t1, $sp, Bias_arg
    st.d    $t2, $sp, Flags_arg

.ifeqs "\BiasFilter\()","BiasFilter"
	addi.d	$a1, $a1, 4*8*4
.endif
	st.d	$a1, $sp, Filter_save_offset
	move	$a1, $a7
	move	$t5, $a6
	move	$t8, $a4
	move	$t1, $a5
	move	$a4, $a2
	move	$a5, $a3

//
// Process the specified number of filter rows.
//

	ori	$s0, $zero, 3
	beq	$t1, $s0, .L\KernelType\().ProcessFilterCount3
	bltu	$t1, $s0, .L\KernelType\().ProcessFilterCountLessThan3
        ProcessFilterCountN LSconvKernelFrame, \KernelType\(), 4
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount3:
        ProcessFilterCountN LSconvKernelFrame, \KernelType\(), 3
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCountLessThan3:
	ori	$s0, $zero, 2
	bltu	$t1, $s0, .L\KernelType\().ProcessFilterCount1
        ProcessFilterCountN LSconvKernelFrame, \KernelType\(), 2
        b     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount1:
        ProcessFilterCountN LSconvKernelFrame, \KernelType\(), 1

//
// Restore non-volatile registers and return.
//

.L\KernelType\().ExitKernel:
.ifnes "\Isa\()","LSX"
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
.endif
	ld.d	$s0, $sp, 0
	ld.d	$s1, $sp, 8
	ld.d	$s2, $sp, 2*8
	ld.d	$ra, $sp, 5*8
	addi.d	$sp, $sp, SP_SIZE
	jirl	$zero, $ra, 0

.ifnes "\Isa\()","LSX"

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

        .irp FilterCount, 1, 2, 3, 4

MlasConv\KernelType\()FloatSingle\Isa\()Filter\FilterCount\():
    st.d	$ra, $sp, 19*8
loopMlasConv\KernelType\()FloatSingle\Isa\()Filter\FilterCount\():
        ProcessOutputCountN \Isa\(), LSconvKernelSingleFrame, \KernelType\(), \BlockSize\(), \FilterCount\(), 1
	add.d	$a0, $a0, $a5                # advance input by 1 element
	addi.d	$t0, $t0, -1                 # decrement output count remaining
    bnez	$t0, loopMlasConv\KernelType\()FloatSingle\Isa\()Filter\FilterCount\()
    ld.d	$ra, $sp, 19*8
	jr	$ra

        .endr

.endif

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case of a depthwise separable convolution.

Arguments:

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro SconvKernelDepthwiseFunction BlockSize, Isa

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Depthwise separable convolutions are a form of grouped convolution where
    the number of input and output channels per group are one.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (a1) - Supplies the address of the filter buffer.

    Output (a2) - Supplies the address of the output buffer.

    StrideWidth (a3) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (a4) - Supplies the length in bytes of the blocked dilation
        width.

    InputStride (a5) - Supplies the length in bytes to advance the input buffer
        to the next input row.

    KernelHeight (a6)- Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth (a7)- Supplies the width of the kernel to apply.

    InputBase (sp + 0 )- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (sp + 8 )- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (sp + 0x10)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (sp + 0x18)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (sp + 0x20)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (sp + 0x28)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias (sp + 0x30)- Supplies the address of the bias buffer.

    Flags (sp + 0x38)- Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvDepthwiseFloatKernel\Isa\()

	addi.d	$sp, $sp, -SP_SIZE
	st.d	$s0, $sp, 0
	st.d	$s1, $sp, 8
	st.d	$s2, $sp, 2*8
	st.d	$ra, $sp, 5*8

    st.d    $a6, $sp, KernelHeight_arg
    st.d    $a7, $sp, KernelWidth_arg

    ld.d    $t0, $sp, SP_SIZE+0*8
    ld.d    $t1, $sp, SP_SIZE+1*8
    ld.d    $t2, $sp, SP_SIZE+2*8
    ld.d    $t3, $sp, SP_SIZE+3*8
    st.d    $t0, $sp, InputBase_arg
    st.d    $t1, $sp, InputWidth_arg
    st.d    $t2, $sp, DilatedInputWidth_arg
    st.d    $t3, $sp, OutputCountLeftPad_arg
    ld.d    $t0, $sp, SP_SIZE+4*8
    ld.d    $t1, $sp, SP_SIZE+5*8
    ld.d    $t2, $sp, SP_SIZE+6*8
    ld.d    $t3, $sp, SP_SIZE+7*8
    st.d    $t0, $sp, OutputCount_arg
    st.d    $t1, $sp, OutputCountRightPad_arg
    st.d    $t2, $sp, Bias_arg
    st.d    $t3, $sp, Flags_arg

	move	$t8, $a4
	move	$t5, $a5
	move	$a4, $a2
	move	$a5, $a3

//
// Process the specified number of filter rows.
//

        ProcessFilterCountN LSconvKernelDepthwiseFrame, Depthwise, 1

//
// Restore non-volatile registers and return.
//

.LDepthwise.ExitKernel:
.ifnes "\Isa\()","LSX"
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
.endif
	ld.d	$s0, $sp, 0
	ld.d	$s1, $sp, 8
	ld.d	$s2, $sp, 2*8
	ld.d	$ra, $sp, 5*8
	addi.d	$sp, $sp, SP_SIZE
	jr	$ra

.ifnes "\Isa\()","LSX"

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

MlasConvDepthwiseFloatSingle\Isa\()Filter1:
    st.d	$ra, $sp, 20*8
MlasConvDepthwiseFloatSingle\Isa\()Filter1_loop:
        ProcessOutputCountN \Isa\(), LSconvKernelDepthwiseSingleFrame, Depthwise, \BlockSize\(), 1, 1
	add.d	$a0, $a0, $a5                # advance input by 1 element
	addi.d	$t0, $t0, -1                # decrement output count remaining

        bnez	$t0, MlasConvDepthwiseFloatSingle\Isa\()Filter1_loop
	ld.d	$ra, $sp, 20*8
	jr	$ra

.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks
    for a pointwise convolution.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description).

    t8 - Supplies the InputStride parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    t2 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseOutputCountN Isa, BlockSize, FilterCount, OutputCount

	move	$a3, $a0
	move	$a2, $t2
	ld.d	$t1, $sp, InputChannels_arg
        ClearBlock \FilterCount\(), \OutputCount\()

.LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock:
.if \OutputCount\() > 3
	slli.d	$s0, $a5, 1
	add.d	$s0, $s0, $a5
	add.d	$t4, $s0, $a3
.endif
.if \FilterCount\() > 2
	slli.d	$s0, $a1, 1
	add.d	$t7, $a2, $s0
.endif
.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
	add.d	$a3, $a3, $t8                # advance input to next channel block

	addi.d	$a2, $a2, \BlockSize\()*\BlockSize\()*4    # advance filter by 8i8o/16i16o block
	addi.d	$t1, $t1, -1                 # decrement input blocks remaining

        bnez	$t1, .LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock

//
// Handle post processing of the output block.
//

	ld.w	$a2, $sp, Flags_arg
.if \FilterCount\() > 1
	ld.d	$t6, $sp, OutputStride_arg
.endif
	ld.d	$a3, $sp, Bias_arg
        bl    MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case where the kernel dimensions are 1.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    BiasFilter - Supplies a non-blank value if the address of the filter buffer
        should be biased to point to the middle of a OIhw8i8o block in order to
        reduce the code size from relative byte offsets.

--*/

        .macro SconvKernelPointwiseFunction Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Pointwise convolutions have a kernel size of one. To simplify this
    implementation, no input padding is allowed, which matches typical usage in
    models.

Arguments:

    Input (a0) - Supplies the address of the input buffer.

    Filter (a1) - Supplies the address of the filter buffer.

    Output (a2) - Supplies the address of the output buffer.

    StrideWidth (a3) - Supplies the length in bytes of the blocked stride width.

    InputChannels (a4) - Supplies the number of input channels to process.

    FilterCount (a5) - Supplies the number of rows from the filter to process.

    InputStride (a6) - Supplies the length in bytes to advance the input buffer to
        the next input channel of the same input row.

    FilterStride (a7) - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride (sp + 0)- Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    OutputCount (sp + 8)- Supplies the number of output elements.

    Bias (sp + 0x10)- Supplies the address of the bias buffer.

    Flags (sp + 0x18)- Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvPointwiseFloatKernel\Isa\()

	addi.d	$sp, $sp, -SP_SIZE
	st.d	$s0, $sp, 0*8
	st.d	$s1, $sp, 1*8
	st.d	$s2, $sp, 2*8
	st.d	$ra, $sp, 5*8

    ld.d    $t0, $sp, SP_SIZE+0*8
    ld.d    $t1, $sp, SP_SIZE+1*8
    ld.d    $t2, $sp, SP_SIZE+2*8
    ld.d    $t3, $sp, SP_SIZE+3*8
    st.d    $t0, $sp, OutputStride_arg
    st.d    $t1, $sp, OutputCount_arg
    st.d    $t2, $sp, Bias_arg
    st.d    $t3, $sp, Flags_arg
    st.d    $a4, $sp, InputChannels_arg

.ifeqs "\BiasFilter\()","BiasFilter"
	addi.d	$t2, $a1, 4*8*4
.else
	move	$t2, $a1
.endif
	ld.d	$t0, $sp, OutputCount_arg
	move	$a1, $a7
	move	$t8, $a6
	move	$t1, $a5
	move	$a4, $a2
	move	$a5, $a3

//
// Process the specified number of filter rows.
//

	ori	$s0, $zero, 3
	beq	$t1, $s0, .LPointwise.ProcessFilterCount3
	bltu	$t1, $s0, .LPointwise.ProcessFilterCountLessThan3
        ProcessPointwiseFilterCountN 4
        b     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount3:
        ProcessPointwiseFilterCountN 3
        b     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCountLessThan3:
	ori	$s0, $zero, 2
	bltu	$t1, $s0, .LPointwise.ProcessFilterCount1
        ProcessPointwiseFilterCountN 2
        b     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount1:
        ProcessPointwiseFilterCountN 1

//
// Restore non-volatile registers and return.
//

.LPointwise.ExitKernel:
.ifnes "\Isa\()","LSX"
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
.endif
	ld.d	$s0, $sp, 0*8
	ld.d	$s1, $sp, 1*8
	ld.d	$s2, $sp, 2*8
	ld.d	$ra, $sp, 5*8
	addi.d	$sp, $sp, SP_SIZE
	jr	$ra

        .endm

/*++

Macro Description:

    This macro generates code to clear the block accumulators.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    xr0-xr11 - Supplies the block accumulators.

--*/

        .macro ClearBlock FilterCount, OutputCount
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "xvxor.v $xr0, $xr0, $xr0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "xvxor.v $xr4, $xr4, $xr4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "xvxor.v $xr8, $xr8, $xr8"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "xvxor.v $xr1, $xr1, $xr1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "xvxor.v $xr5, $xr5, $xr5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "xvxor.v $xr9, $xr9, $xr9"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "xvxor.v $xr2, $xr2, $xr2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "xvxor.v $xr6, $xr6, $xr6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "xvxor.v $xr10, $xr10, $xr10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "xvxor.v $xr3, $xr3, $xr3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "xvxor.v $xr7, $xr7, $xr7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "xvxor.v $xr11, $xr11, $xr11"

        .endm
