/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelSse2.s

Abstract:

    This module implements the kernels for the single precision pooling
    operation.

    This implementation uses SSE2 instructions.

--*/

#include "asmmacro.h"
#include "SpoolKernelCommon.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro generates code to initialize registers used across the kernel.

Arguments:

    PoolingType - Supplies the pooling type string.

--*/

        .macro InitializeKernel PoolingType

.ifeqs "\PoolingType\()","Maximum"
        mov     eax,0xFF7FFFFF
        movd    xmm5,eax
        shufps  xmm5,xmm5,0
.endif

.ifeqs "\PoolingType\()","AverageIncludePad"
        cvtsi2ss xmm5,r9
        shufps  xmm5,xmm5,0
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

    rsi - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    xmm0-xmm1 - Supplies the pooling intermediates.

    xmm5 - Supplies a vector containing the minimum float value broadcasted,
        if PoolingType==Maximum.

--*/

        .macro ClearBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
        movaps  xmm0,xmm5
        movaps  xmm1,xmm5
.else
        xorps   xmm0,xmm0
        xorps   xmm1,xmm1
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
        xor     rsi,rsi                     # reset valid block counter
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

    rcx - Supplies the address of the input buffer.

    rsi - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    r8 - Supplies the StrideWidth parameter (see function description).

    xmm0-xmm1 - Supplies the pooling intermediates.

--*/

        .macro ComputeBlock PoolingType, OutputCount

.ifeqs "\PoolingType\()","Maximum"
        maxps   xmm0,XMMWORD PTR [rcx]
        maxps   xmm1,XMMWORD PTR [rcx+16]
.else
        addps   xmm0,XMMWORD PTR [rcx]
        addps   xmm1,XMMWORD PTR [rcx+16]
.endif

.ifeqs "\PoolingType\()","AverageExcludePad"
        inc     rsi                         # increment valid block counter
.endif

        .endm

/*++

Macro Description:

    This macro generates code to process and store the pooling intermediates.

Arguments:

    PoolingType - Supplies the pooling type string.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    rdx - Supplies the address of the output buffer.

    rsi - Supplies the number of blocks accessed by ComputeBlock, if
        PoolingType=AverageExcludePad and OutputCount=1.

    xmm0-xmm1 - Supplies the pooling intermediates.

    xmm5 - Supplies the kernel size computed by InitializeKernel, if
        PoolingType=AverageExcludePad, else the actual kernel size, if
        PoolingType=AverageIncludePad.

--*/

        .macro PostProcessBlock PoolingType, OutputCount

//
// If PoolingType=AverageExcludePad, divide the sum by the number of non-padding
// blocks.
//

.ifeqs "\PoolingType\()","AverageExcludePad"
        xorps   xmm4,xmm4
        cvtsi2ss xmm4,rsi                   # convert valid block counter
        shufps xmm4,xmm4,0
        divps  xmm0,xmm4
        divps  xmm1,xmm4
.endif

//
// If PoolingType=AverageIncludePad, divide the sum by the actual kernel size.
//

.ifeqs "\PoolingType\()","AverageIncludePad"
        divps   xmm0,xmm5
        divps   xmm1,xmm5
.endif

//
// Store the output block in the output buffer.
//

        movups  XMMWORD PTR [rdx],xmm0
        movups  XMMWORD PTR [rdx+16],xmm1
        add     rdx,8*4                     # advance output by 1 nchw8c block

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

    Input (rdi) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Output (rsi) - Supplies the address of the output buffer.

    StrideWidth (rdx) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (rcx) - Supplies the length in bytes of the blocked dilation
        width.

    InputStride (r8) - Supplies the length in bytes to advance the input buffer to
        the next input row.

    ActualKernelSize (r9) - Supplies the size of the kernel based on the original
        kernel dimensions, used for PoolingType=AverageIncludePad.

    KernelHeight - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth - Supplies the width of the kernel to apply.

    InputBase - Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth - Supplies the length in bytes of the blocked input width.

    DilatedInputWidth - Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad - Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount - Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad - Supplies the number of output elements that include
        one or more padding elements from the right edge.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasPool\PoolingType\()FloatKernel\Isa\()

        SpoolKernelEntry \PoolingType\()

        mov     r10,.LSpoolKernelFrame_OutputCountLeftPad[rsp]
        add     r10,.LSpoolKernelFrame_OutputCount[rsp]
        add     r10,.LSpoolKernelFrame_OutputCountRightPad[rsp]
        jz      .L\PoolingType\().ExitKernel

.L\PoolingType\().ProcessNextOutputCount:
        ProcessOutputCountN .LSpoolKernelFrame, \PoolingType\(), 1
        add     rdi,r8                      # advance input by 1 element
        dec     r10
        jnz     .L\PoolingType\().ProcessNextOutputCount

.L\PoolingType\().ExitKernel:
        SpoolKernelExit

        .endm

//
// Generate the pooling kernels.
//

        SpoolKernelFunction Maximum, Sse
        SpoolKernelFunction AverageExcludePad, Sse
        SpoolKernelFunction AverageIncludePad, Sse

        .end
