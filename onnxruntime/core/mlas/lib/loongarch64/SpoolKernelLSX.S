/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelLSX.s

Abstract:

    This module implements the kernels for the single precision pooling
    operation.

    This implementation uses LSX instructions.

--*/

#define SP_SIZE 32*8
#define InputBase_arg                   SP_SIZE+0*8
#define InputWidth_arg                  SP_SIZE+1*8
#define DilatedInputWidth_arg           SP_SIZE+2*8
#define OutputCountLeftPad_arg          SP_SIZE+3*8
#define OutputCount_arg                 SP_SIZE+4*8
#define OutputCountRightPad_arg         SP_SIZE+5*8

        .macro FUNCTION_ENTRY FunctionName

        .p2align 4
        .globl  \FunctionName\()
        .type   \FunctionName\(),@function
\FunctionName\():

        .endm


        .text

/*++

Macro Description:

    This macro generates code to initialize registers used across the kernel.

Arguments:

    PoolingType - Supplies the pooling type string.

--*/

        .macro InitializeKernel PoolingType

.ifeqs "\PoolingType\()","Maximum"
	li.w	$s0, 0xFF7FFFFF
	vreplgr2vr.w	$vr5, $s0
.endif

.ifeqs "\PoolingType\()","AverageIncludePad"
	vreplgr2vr.w	$vr5, $a5
    vffint.s.w      $vr5, $vr5
.endif

        .endm
/*++

Macro Description:

    This macro generates the common prologue code for the pooling kernels.

Arguments:

    PoolingType - Supplies the pooling type string.

--*/

        .macro SpoolKernelEntry PoolingType

        addi.d  $sp, $sp, -SP_SIZE
        st.d    $s0, $sp, 0*8
        st.d    $s1, $sp, 1*8
        st.d    $s2, $sp, 2*8
        st.d    $s3, $sp, 3*8
        st.d    $s4, $sp, 4*8
        st.d    $ra, $sp, 5*8
        fst.d   $f24,$sp, 6*8

        InitializeKernel \PoolingType\()
	# move InputStride to s8
	or	$t8, $a4, $r0
	# move StrideWidth to a4
	or	$a4, $a2, $r0
	# move DilationWidth to a5
	or	$a5, $a3, $r0
	# move Output to a2
	or	$a2, $a1, $r0

        .endm

/*++

Macro Description:

    This macro generates the common epilogue code for the pooling kernels.

Arguments:

    None.

--*/

        .macro SpoolKernelExit

        ld.d    $s0, $sp, 0*8
        ld.d    $s1, $sp, 1*8
        ld.d    $s2, $sp, 2*8
        ld.d    $s3, $sp, 3*8
        ld.d    $s4, $sp, 4*8
        ld.d    $ra, $sp, 5*8
        fld.d   $f24,$sp, 6*8

        addi.d  $sp, $sp, SP_SIZE
        jr $ra

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

    vr0-vr1 - Supplies the pooling intermediates.

    vr2 - Supplies a vector containing the minimum float value broadcasted,
        if PoolingType==Maximum.

--*/

        .macro ClearBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
	vor.v	$vr0, $vr5, $vr5
	vor.v	$vr1, $vr5, $vr5
.else
	vxor.v	$vr0, $vr0, $vr0
	vxor.v	$vr1, $vr1, $vr1
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
	xor	$a1, $a1, $a1		# reset valid block counter
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

    vr0-vr1 - Supplies the pooling intermediates.

--*/

        .macro ComputeBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
	vld	$vr24, $a3, 0
	vfmax.s	$vr0, $vr0, $vr24
	vld	$vr24, $a3, 16
	vfmax.s	$vr1, $vr1, $vr24
.else
	vld	$vr24, $a3, 0
	vfadd.s	$vr0, $vr0, $vr24
	vld	$vr24, $a3, 16
	vfadd.s	$vr1, $vr1, $vr24
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
        # increment valid block counter
	addi.d	$a1, $a1, 1
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

    vr0-vr1 - Supplies the pooling intermediates.

    vr5 - Supplies the kernel size computed by InitializeKernel, if
        PoolingType=AverageExcludePad, else the actual kernel size, if
        PoolingType=AverageIncludePad.

--*/

        .macro PostProcessBlock PoolingType, OutputCount

//
// If PoolingType=AverageExcludePad, divide the sum by the number of non-padding
// blocks.
//

.ifeqs "\PoolingType\()","AverageExcludePad"
	# convert valid block counter
	vreplgr2vr.w	$vr4, $a1
    vffint.s.w      $vr4, $vr4
	vfdiv.s	$vr0, $vr0, $vr4
	vfdiv.s	$vr1, $vr1, $vr4
.endif

//
// If PoolingType=AverageIncludePad, divide the sum by the actual kernel size.
//

.ifeqs "\PoolingType\()","AverageIncludePad"
	vfdiv.s	$vr0, $vr0, $vr5
	vfdiv.s	$vr1, $vr1, $vr5
.endif

//
// Store the output block in the output buffer.
//

	vst	$vr0, $a2, 0
	vst	$vr1, $a2, 16
        # advance output by 1 nchw8c block
	addi.d	$a2, $a2, 8*4

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

    s8 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessOutputCountN KernelFrame, PoolingType, OutputCount

	move	$a3, $a0
	move	$t1, $a6
	move	$t2, $a7
.if \OutputCount\() == 1
	ld.d	$t3, $sp, InputBase_arg
	ld.d	$t4, $sp, InputWidth_arg
	sub.d	$t3, $r0, $t3		# keep negative for lea usage below
.endif
        ClearBlock \PoolingType\(), \OutputCount\()
        beqz	$t1, .L\PoolingType\().\OutputCount\().HandlePostProcessing

.L\PoolingType\().\OutputCount\().ProcessNextRow:
	or	$t6, $t2, $t2

.L\PoolingType\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
        # (Input - InputBase) >= InputWidth?
	add.d	$t7, $a3, $t3
    bgeu	$t7, $t4, .L\PoolingType\().\OutputCount\().SkipOverPadding
.endif
        ComputeBlock \PoolingType\(), \OutputCount\()

.L\PoolingType\().\OutputCount\().SkipOverPadding:
        add.d	$a3, $a3, $a5       # advance input by dilation width
        # decrement columns remaining
	    addi.d	$t6, $t6, -1
        bnez	$t6, .L\PoolingType\().\OutputCount\().ProcessNextColumn
        add.d	$a3, $a3, $t8      # advance input to next row
.if \OutputCount\() == 1
	ld.d	$s0, $sp, DilatedInputWidth_arg
        # advance input base to next row
	sub.d	$t3, $t3, $s0
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

    KernelWidth (a7) - Supplies the width of the kernel to apply.

    InputBase (0)- Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth (1*8)- Supplies the length in bytes of the blocked input width.

    DilatedInputWidth (2*8)- Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad (3*8)- Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount (4*8)- Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad (5*8)- Supplies the number of output elements that include
        one or more padding elements from the right edge.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasPool\PoolingType\()FloatKernel\Isa\()
        SpoolKernelEntry \PoolingType\()

	ld.d	$s0, $sp, OutputCountLeftPad_arg
	ld.d	$s1, $sp, OutputCount_arg
	add.d	$t0, $s0, $s1
	ld.d	$s0, $sp, OutputCountRightPad_arg
	add.d	$t0, $t0, $s0
    beqz	$t0, .L\PoolingType\().ExitKernel

.L\PoolingType\().ProcessNextOutputCount:
    ProcessOutputCountN .LSpoolKernelFrame, \PoolingType\(), 1
	add.d	$a0, $a0, $a4
	addi.d	$t0, $t0, -1
    bnez	$t0, .L\PoolingType\().ProcessNextOutputCount

.L\PoolingType\().ExitKernel:
        SpoolKernelExit

        .endm

//
// Generate the pooling kernels.
//

        SpoolKernelFunction Maximum, LSX
        SpoolKernelFunction AverageExcludePad, LSX
        SpoolKernelFunction AverageIncludePad, LSX

        .end
