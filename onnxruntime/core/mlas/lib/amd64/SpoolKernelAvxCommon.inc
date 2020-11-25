;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SpoolKernelAvxCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision pooling operation for the AVX and AVX512F kernels.
;
;--

INCLUDE SpoolKernelCommon.inc

;
; Macro Description:
;
;   This macro generates code for the inner pooling kernel.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;
;   Isa - Supplies the instruction set architecture string for function tags.
;

SpoolKernelFunction MACRO PoolingType, Isa

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute pooling for the elements of an
;   output row for a set of filter rows.
;
; Arguments:
;
;   Input (rcx) - Supplies the address of the input buffer.
;
;       The address is biased to include padding blocks for the left width
;       dimension. The address is not biased to include padding rows for the
;       left height dimension; these are accounted for in the outer kernel.
;
;   Output (rdx) - Supplies the address of the output buffer.
;
;   StrideWidth (r8) - Supplies the length in bytes of the blocked stride width.
;
;   DilationWidth (r9) - Supplies the length in bytes of the blocked dilation
;       width.
;
;   InputStride - Supplies the length in bytes to advance the input buffer to
;       the next input row.
;
;   ActualKernelSize - Supplies the size of the kernel based on the original
;       kernel dimensions, used for PoolingType=AverageIncludePad.
;
;   KernelHeight - Supplies the height of the kernel to apply. This height may
;       be less than the original kernel height after removing any padding
;       rows.
;
;   KernelWidth - Supplies the width of the kernel to apply.
;
;   InputBase - Supplies the address of the valid input buffer.
;
;       This parameter is similar to the Input parameter, but does not include
;       the padding blocks for the left width dimension. This parameter is used
;       with the following InputWidth parameter in order to validate that the
;       current input buffer address in bounds and not in the left or right
;       width padding region.
;
;   InputWidth - Supplies the length in bytes of the blocked input width.
;
;   DilatedInputWidth - Supplies the length in bytes to advance the input base
;       buffer to the next input row including dilation.
;
;   OutputCountLeftPad - Supplies the number of output elements that include
;       one or more padding elements from the left edge.
;
;   OutputCount - Supplies the number of output elements that do not include
;       any padding elements.
;
;   OutputCountRightPad - Supplies the number of output elements that include
;       one or more padding elements from the right edge.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasPool&PoolingType&FloatKernel&Isa&, _TEXT

        SpoolKernelEntry PoolingType

ProcessOutputCountLeftPad:
        mov     r10,SpoolKernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      ProcessOutputCount
        call    MlasPool&PoolingType&FloatSingle&Isa&

ProcessOutputCount:
        mov     r10,SpoolKernelFrame.OutputCount[rsp]
        sub     r10,3
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy3:
        ProcessOutputCountN SpoolKernelFrame, PoolingType, 3
        lea     rax,[r8*2+r8]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jae     ProcessNextOutputCountBy3

ProcessRemainingOutputCount:
        add     r10,3                       ; correct for over-subtract above

ProcessOutputCountRightPad:
        add     r10,SpoolKernelFrame.OutputCountRightPad[rsp]
        jz      ExitKernel
        call    MlasPool&PoolingType&FloatSingle&Isa&

ExitKernel:
        vzeroupper
        SpoolKernelExit

        NESTED_END MlasPool&PoolingType&FloatKernel&Isa&, _TEXT

;
; Generate out-of-band helpers for handling output blocks involving padding.
;

        LEAF_ENTRY MlasPool&PoolingType&FloatSingle&Isa&, _TEXT

ProcessNextOutputCount:
        ProcessOutputCountN SpoolKernelSingleFrame.KernelFrame, PoolingType, 1
        add     rdi,r8                      ; advance input by 1 element
        dec     r10                         ; decrement output count remaining
        jnz     ProcessNextOutputCount
        ret

        LEAF_END MlasPool&PoolingType&FloatSingle&Isa&, _TEXT

        ENDM
