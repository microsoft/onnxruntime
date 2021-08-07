/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelAvxCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision pooling operation for the AVX and AVX512F kernels.

--*/

#include "SpoolKernelCommon.h"

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

.L\PoolingType\().ProcessOutputCountLeftPad:
        mov     r10,.LSpoolKernelFrame_OutputCountLeftPad[rsp]
        test    r10,r10
        jz      .L\PoolingType\().ProcessOutputCount
        call    MlasPool\PoolingType\()FloatSingle\Isa\()

.L\PoolingType\().ProcessOutputCount:
        mov     r10,.LSpoolKernelFrame_OutputCount[rsp]
        sub     r10,3
        jb      .L\PoolingType\().ProcessRemainingOutputCount

.L\PoolingType\().ProcessNextOutputCountBy3:
        ProcessOutputCountN .LSpoolKernelFrame, \PoolingType\(), 3
        lea     rax,[r8*2+r8]
        add     rdi,rax                     # advance input by 3 elements
        sub     r10,3
        jae     .L\PoolingType\().ProcessNextOutputCountBy3

.L\PoolingType\().ProcessRemainingOutputCount:
        add     r10,3                       # correct for over-subtract above

.L\PoolingType\().ProcessOutputCountRightPad:
        add     r10,.LSpoolKernelFrame_OutputCountRightPad[rsp]
        jz      .L\PoolingType\().ExitKernel
        call    MlasPool\PoolingType\()FloatSingle\Isa\()

.L\PoolingType\().ExitKernel:
        vzeroupper
        SpoolKernelExit

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

MlasPool\PoolingType\()FloatSingle\Isa\():
        ProcessOutputCountN .LSpoolKernelSingleFrame, \PoolingType\(), 1
        add     rdi,r8                      # advance input by 1 element
        dec     r10                         # decrement output count remaining
        jnz     MlasPool\PoolingType\()FloatSingle\Isa\()
        ret

        .endm
