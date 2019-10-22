/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SpoolKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision pooling operation.

--*/

//
// Stack frame layout for the pooling kernels.
//

        .equ    .LSpoolKernelFrame_BroadcastValue, -8
        .equ    .LSpoolKernelFrame_SavedR12, 0
        .equ    .LSpoolKernelFrame_SavedR13, 8
        .equ    .LSpoolKernelFrame_SavedR14, 16
        .equ    .LSpoolKernelFrame_SavedRbx, 24
        .equ    .LSpoolKernelFrame_SavedRbp, 32
        .equ    .LSpoolKernelFrame_ReturnAddress, 40
        .equ    .LSpoolKernelFrame_KernelHeight, 48
        .equ    .LSpoolKernelFrame_KernelWidth, 56
        .equ    .LSpoolKernelFrame_InputBase, 64
        .equ    .LSpoolKernelFrame_InputWidth, 72
        .equ    .LSpoolKernelFrame_DilatedInputWidth, 80
        .equ    .LSpoolKernelFrame_OutputCountLeftPad, 88
        .equ    .LSpoolKernelFrame_OutputCount, 96
        .equ    .LSpoolKernelFrame_OutputCountRightPad, 104

        .equ    .LSpoolKernelSingleFrame_ReturnAddress, 0
        .equ    .LSpoolKernelSingleFrame_SavedR12, 8
        .equ    .LSpoolKernelSingleFrame_SavedR13, 16
        .equ    .LSpoolKernelSingleFrame_SavedR14, 24
        .equ    .LSpoolKernelSingleFrame_SavedRbx, 32
        .equ    .LSpoolKernelSingleFrame_SavedRbp, 40
        .equ    .LSpoolKernelSingleFrame_ParentReturnAddress, 48
        .equ    .LSpoolKernelSingleFrame_KernelHeight, 56
        .equ    .LSpoolKernelSingleFrame_KernelWidth, 64
        .equ    .LSpoolKernelSingleFrame_InputBase, 72
        .equ    .LSpoolKernelSingleFrame_InputWidth, 80
        .equ    .LSpoolKernelSingleFrame_DilatedInputWidth, 88
        .equ    .LSpoolKernelSingleFrame_OutputCountLeftPad, 96
        .equ    .LSpoolKernelSingleFrame_OutputCount, 104
        .equ    .LSpoolKernelSingleFrame_OutputCountRightPad, 112

/*++

Macro Description:

    This macro generates the common prologue code for the pooling kernels.

Arguments:

    PoolingType - Supplies the pooling type string.

--*/

        .macro SpoolKernelEntry PoolingType

        push    rbp
        push    rbx
        push    r14
        push    r13
        push    r12

        InitializeKernel \PoolingType\()
        mov     rbp,r8                      # shuffle to Win64 register usage
        mov     r8,rdx
        mov     r9,rcx
        mov     rdx,rsi

        .endm

/*++

Macro Description:

    This macro generates the common epilogue code for the pooling kernels.

Arguments:

    None.

--*/

        .macro SpoolKernelExit

        pop     r12
        pop     r13
        pop     r14
        pop     rbx
        pop     rbp
        ret

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

    rdi - Supplies the address of the input buffer.

    rdx - Supplies the address of the output buffer.

    r8 - Supplies the StrideWidth parameter (see function description).

    r9 - Supplies the DilationWidth parameter (see function description).

    rbp - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessOutputCountN KernelFrame, PoolingType, OutputCount

        mov     rcx,rdi
        mov     r11,\KernelFrame\()_KernelHeight[rsp]
        mov     r12,\KernelFrame\()_KernelWidth[rsp]
.if \OutputCount\() == 1
        mov     r13,\KernelFrame\()_InputBase[rsp]
        mov     r14,\KernelFrame\()_InputWidth[rsp]
        neg     r13                         # keep negative for lea usage below
.endif
        ClearBlock \PoolingType\(), \OutputCount\()
        test    r11,r11                     # zero sized kernel?
        jz      .L\PoolingType\().\OutputCount\().HandlePostProcessing

.L\PoolingType\().\OutputCount\().ProcessNextRow:
        mov     rax,r12

.L\PoolingType\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
        lea     rbx,[rcx+r13]               # compute (Input - InputBase)
        cmp     rbx,r14                     # (Input - InputBase) >= InputWidth?
        jae     .L\PoolingType\().\OutputCount\().SkipOverPadding
.endif
        ComputeBlock \PoolingType\(), \OutputCount\()

.L\PoolingType\().\OutputCount\().SkipOverPadding:
        add     rcx,r9                      # advance input by dilation width
        dec     rax                         # decrement columns remaining
        jnz     .L\PoolingType\().\OutputCount\().ProcessNextColumn
        add     rcx,rbp                     # advance input to next row
.if \OutputCount\() == 1
        sub     r13,\KernelFrame\()_DilatedInputWidth[rsp]
                                            # advance input base to next row
.endif
        dec     r11
        jnz     .L\PoolingType\().\OutputCount\().ProcessNextRow

.L\PoolingType\().\OutputCount\().HandlePostProcessing:
        PostProcessBlock \PoolingType\(), \OutputCount\()

        .endm
