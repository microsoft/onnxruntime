;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SpoolKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision pooling operation.
;
;--

;
; Stack frame layout for the pooling kernels.
;

SpoolKernelFrame STRUCT

        SavedR12 QWORD ?
        SavedR13 QWORD ?
        SavedR14 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?              ; Input
        PreviousP2Home QWORD ?              ; Output
        PreviousP3Home QWORD ?              ; StrideWidth
        PreviousP4Home QWORD ?              ; DilationWidth
        InputStride QWORD ?
        ActualKernelSize QWORD ?
        KernelHeight QWORD ?
        KernelWidth QWORD ?
        InputBase QWORD ?
        InputWidth QWORD ?
        DilatedInputWidth QWORD ?
        OutputCountLeftPad QWORD ?
        OutputCount QWORD ?
        OutputCountRightPad QWORD ?

SpoolKernelFrame ENDS

SpoolKernelSingleFrame STRUCT

        ReturnAddress QWORD ?
        KernelFrame SpoolKernelFrame <>

SpoolKernelSingleFrame ENDS

;
; Macro Description:
;
;   This macro generates the common prologue code for the pooling kernels.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;

SpoolKernelEntry MACRO PoolingType

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r14
        push_reg r13
        push_reg r12

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbp,SpoolKernelFrame.InputStride[rsp]
        InitializeKernel PoolingType

        ENDM

;
; Macro Description:
;
;   This macro generates the common epilogue code for the pooling kernels.
;
; Arguments:
;
;   None.
;

SpoolKernelExit MACRO

        BEGIN_EPILOGUE

        pop     r12
        pop     r13
        pop     r14
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute pooling for a vector of input blocks
;   to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
;
; Arguments:
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rdx - Supplies the address of the output buffer.
;
;   r8 - Supplies the StrideWidth parameter (see function description).
;
;   r9 - Supplies the DilationWidth parameter (see function description).
;
;   rbp - Supplies the InputStride parameter (see function description).
;

ProcessOutputCountN MACRO KernelFrame, PoolingType, OutputCount

        LOCAL   ProcessNextRow
        LOCAL   ProcessNextColumn
        LOCAL   SkipOverPadding
        LOCAL   HandlePostProcessing

        mov     rcx,rdi
        mov     r11,KernelFrame.KernelHeight[rsp]
        mov     r12,KernelFrame.KernelWidth[rsp]
IF OutputCount EQ 1
        mov     r13,KernelFrame.InputBase[rsp]
        mov     r14,KernelFrame.InputWidth[rsp]
        neg     r13                         ; keep negative for lea usage below
ENDIF
        ClearBlock PoolingType, OutputCount
        test    r11,r11                     ; zero sized kernel?
        jz      HandlePostProcessing

ProcessNextRow:
        mov     rax,r12

ProcessNextColumn:
IF OutputCount EQ 1
        lea     rbx,[rcx+r13]               ; compute (Input - InputBase)
        cmp     rbx,r14                     ; (Input - InputBase) >= InputWidth?
        jae     SkipOverPadding
ENDIF
        ComputeBlock PoolingType, OutputCount

SkipOverPadding:
        add     rcx,r9                      ; advance input by dilation width
        dec     rax                         ; decrement columns remaining
        jnz     ProcessNextColumn
        add     rcx,rbp                     ; advance input to next row
IF OutputCount EQ 1
        sub     r13,KernelFrame.DilatedInputWidth[rsp]
                                            ; advance input base to next row
ENDIF
        dec     r11
        jnz     ProcessNextRow

HandlePostProcessing:
        PostProcessBlock PoolingType, OutputCount

        ENDM
