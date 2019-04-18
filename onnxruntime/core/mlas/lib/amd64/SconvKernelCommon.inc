;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision convolution operation.
;
;--

;
; Stack frame layout for the convolution kernels.
;

SconvKernelFrame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR12 QWORD ?
        SavedR13 QWORD ?
        SavedR14 QWORD ?
        SavedR15 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?              ; Input
        PreviousP2Home QWORD ?              ; Filter
        PreviousP3Home QWORD ?              ; Output
        PreviousP4Home QWORD ?              ; StrideWidth
        DilationWidth QWORD ?
        FilterCount QWORD ?
        InputStride QWORD ?
        FilterStride QWORD ?
        OutputStride QWORD ?
        KernelHeight QWORD ?
        KernelWidth QWORD ?
        InputBase QWORD ?
        InputWidth QWORD ?
        DilatedInputWidth QWORD ?
        OutputCountLeftPad QWORD ?
        OutputCount QWORD ?
        OutputCountRightPad QWORD ?
        Bias QWORD ?
        Flags QWORD ?

SconvKernelFrame ENDS

SconvKernelSingleFrame STRUCT

        ReturnAddress QWORD ?
        KernelFrame SconvKernelFrame <>

SconvKernelSingleFrame ENDS

SconvKernel1x1Frame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR12 QWORD ?
        SavedR14 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        InputChannels QWORD ?
        FilterCount QWORD ?
        AdjustCount QWORD ?
        FilterStride QWORD ?
        OutputStride QWORD ?
        OutputCount QWORD ?
        Bias QWORD ?
        Flags QWORD ?

SconvKernel1x1Frame ENDS

;
; ProcessOutputCountN
;
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
;
; Arguments:
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   Format - Supplies the block format of the input buffer.
;
;   BlockSize - Supplies the number of elements per block.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rbp - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rdi - Supplies the DilationWidth parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r15 - Supplies the InputStride parameter (see function description).
;

ProcessOutputCountN MACRO KernelFrame, Format, BlockSize, FilterCount, OutputCount

        LOCAL   ProcessNextRow
        LOCAL   ProcessNextColumn
        LOCAL   HandlePostProcessing
        LOCAL   SkipOverPadding

        mov     rcx,rbp
        mov     rdx,KernelFrame.PreviousP2Home[rsp]
        mov     r11,KernelFrame.KernelHeight[rsp]
        mov     r12,KernelFrame.KernelWidth[rsp]
IF OutputCount EQ 1
        mov     r13,KernelFrame.InputBase[rsp]
        mov     r14,KernelFrame.InputWidth[rsp]
        neg     r13                         ; keep negative for lea usage below
ENDIF
        ClearBlock FilterCount, OutputCount
        test    r11,r11                     ; zero sized kernel?
        jz      HandlePostProcessing

ProcessNextRow:
        mov     rax,r12                     ; reload kernel width remaining

ProcessNextColumn:
IF OutputCount EQ 1
        lea     rbx,[rcx+r13]               ; compute (Input - InputBase)
        cmp     rbx,r14                     ; (Input - InputBase) >= InputWidth?
        jae     SkipOverPadding
ENDIF
IF OutputCount GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx                     ; compute input plus 3 blocks
ENDIF
IF FilterCount GT 2
        lea     rbx,[rdx+rsi*2]             ; compute filter plus 2 rows
ENDIF
IFIDNI <Format>, <Nchw>
        ComputeBlock FilterCount, OutputCount, 0, 0
ELSE
IF BlockSize EQ 16
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>
            ComputeBlock FilterCount, OutputCount, Index*16*4, Index*4
        ENDM
ELSE
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7>
            ComputeBlock FilterCount, OutputCount, (Index-4)*8*4, Index*4
        ENDM
ENDIF
ENDIF

SkipOverPadding:
        add     rcx,rdi                     ; advance input by dilation width
IFIDNI <Format>, <Nchw>
        add     rdx,BlockSize*4             ; advance filter by 8o/16o block
ELSE
        add     rdx,BlockSize*BlockSize*4   ; advance filter by 8i8o/16i16o block
ENDIF
        dec     rax                         ; decrement columns remaining
        jnz     ProcessNextColumn
        add     rcx,r15                     ; advance input to next row
IF OutputCount EQ 1
        sub     r13,KernelFrame.DilatedInputWidth[rsp]
                                            ; advance input base to next row
ENDIF
        dec     r11                         ; decrement rows remaining
        jnz     ProcessNextRow

;
; Handle post processing of the output block.
;

HandlePostProcessing:
        PostProcessBlock KernelFrame, FilterCount, OutputCount

        ENDM

;
; SconvKernelFunction
;
;   This macro generates code for the inner convolution kernel.
;
; Arguments:
;
;   Format - Supplies the block format of the input buffer.
;
;   BlockSize - Supplies the number of elements per block.
;
;   Isa - Supplies the instruction set architecture string for function tags.
;
;   BiasFilter - Supplies a non-blank value if the address of the filter buffer
;       should be biased to point to the middle of a OIhw8i8o block in order to
;       reduce the code size from relative byte offsets.
;

SconvKernelFunction MACRO Format, BlockSize, Isa, BiasFilter

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute a convolution for the elements
;   of an output row for a set of filter rows.
;
; Arguments:
;
;   Input (rcx) - Supplies the address of the input buffer.
;
;       The address is biased to include padding blocks for the left width
;       dimension. The address is not biased to include padding rows for the
;       left height dimension; these are accounted for in the outer kernel.
;
;   Filter (rdx) - Supplies the address of the filter buffer.
;
;   Output (r8) - Supplies the address of the output buffer.
;
;   StrideWidth (r9) - Supplies the length in bytes of the blocked stride width.
;
;   DilationWidth - Supplies the length in bytes of the blocked dilation width.
;
;   FilterCount - Supplies the number of filters to process in this iteration.
;
;   InputStride - Supplies the length in bytes to advance the input buffer to
;       the next input row.
;
;   FilterStride - Supplies the length in bytes to advance the filter buffer
;       to the next set of filters.
;
;   OutputStride - Supplies the length in bytes to advance the output buffer
;       to the next output address associated with the next set of filters.
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
;   Bias - Supplies the address of the bias buffer.
;
;   Flags - Supplies additional flags controlling the convolution operation,
;       especially post calcuation options.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConv&Format&FloatKernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r15
        push_reg r14
        push_reg r13
        push_reg r12
        alloc_stack (SconvKernelFrame.SavedR12)

        save_xmm128 xmm6,SconvKernelFrame.SavedXmm6
        save_xmm128 xmm7,SconvKernelFrame.SavedXmm7
        save_xmm128 xmm8,SconvKernelFrame.SavedXmm8
        save_xmm128 xmm9,SconvKernelFrame.SavedXmm9
        save_xmm128 xmm10,SconvKernelFrame.SavedXmm10
        save_xmm128 xmm11,SconvKernelFrame.SavedXmm11
        save_xmm128 xmm12,SconvKernelFrame.SavedXmm12
        save_xmm128 xmm13,SconvKernelFrame.SavedXmm13
        save_xmm128 xmm14,SconvKernelFrame.SavedXmm14
        save_xmm128 xmm15,SconvKernelFrame.SavedXmm15

        END_PROLOGUE

        mov     rbp,rcx
IFNB <BiasFilter>
        add     rdx,4*8*4
ENDIF
        mov     SconvKernelFrame.PreviousP2Home[rsp],rdx
        mov     rsi,SconvKernelFrame.FilterStride[rsp]
        mov     rdi,SconvKernelFrame.DilationWidth[rsp]
        mov     r11,SconvKernelFrame.FilterCount[rsp]
        mov     r15,SconvKernelFrame.InputStride[rsp]

;
; Process the specified number of filter rows.
;

        cmp     r11,3
        je      ProcessFilterCount3
        jb      ProcessFilterCountLessThan3
        ProcessFilterCountN Format, 4
        jmp     ExitKernel

ProcessFilterCount3:
        ProcessFilterCountN Format, 3
        jmp     ExitKernel

ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      ProcessFilterCount1
        ProcessFilterCountN Format, 2
        jmp     ExitKernel

ProcessFilterCount1:
        ProcessFilterCountN Format, 1

;
; Restore non-volatile registers and return.
;

ExitKernel:
IFDIFI <Isa>, <Sse>
        vzeroupper
ENDIF
        movaps  xmm6,SconvKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,SconvKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,SconvKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,SconvKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,SconvKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,SconvKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,SconvKernelFrame.SavedXmm12[rsp]
        movaps  xmm13,SconvKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,SconvKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,SconvKernelFrame.SavedXmm15[rsp]
        add     rsp,(SconvKernelFrame.SavedR12)

        BEGIN_EPILOGUE

        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConv&Format&FloatKernel&Isa&, _TEXT

IFDIFI <Isa>, <Sse>

;
; Generate out-of-band helpers for handling output blocks involving padding.
;

        IRP FilterCount, <1, 2, 3, 4>

        LEAF_ENTRY MlasConv&Format&FloatSingle&Isa&FilterCount&FilterCount, _TEXT

ProcessNextOutputCount:
        ProcessOutputCountN SconvKernelSingleFrame.KernelFrame, Format, BlockSize, FilterCount, 1
        add     rbp,r9                      ; advance input by 1 element
        dec     r10                         ; decrement output count remaining
        jnz     ProcessNextOutputCount
        ret

        LEAF_END MlasConv&Format&FloatSingle&Isa&FilterCount&FilterCount, _TEXT

        ENDM

ENDIF

        ENDM

;
;
;

Process1x1OutputCountN MACRO BlockSize, FilterCount, OutputCount

        LOCAL   ProcessNextInputBlock
        LOCAL   SkipAccumulateOutput
        LOCAL   SkipBiasAddition
        LOCAL   SkipReluActivation

        mov     rcx,rbp
        mov     rdx,r12
        mov     r11,SconvKernel1x1Frame.InputChannels[rsp]
        ClearBlock FilterCount, OutputCount

ProcessNextInputBlock:
IF OutputCount GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx                     ; compute input plus 3 blocks
ENDIF
IF FilterCount GT 2
        lea     rbx,[rdx+rsi*2]             ; compute filter plus 2 rows
ENDIF
IF BlockSize EQ 16
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>
            ComputeBlock FilterCount, OutputCount, Index*16*4, Index*4
        ENDM
ELSE
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7>
            ComputeBlock FilterCount, OutputCount, (Index-4)*8*4, Index*4
        ENDM
ENDIF
        add     rcx,rdi                     ; advance input to next channel block
        add     rdx,BlockSize*BlockSize*4   ; advance filter by 8i8o/16i16o block
        dec     r11                         ; decrement input blocks remaining
        jnz     ProcessNextInputBlock

;
; Handle post processing of the output block.
;

        PostProcessBlock SconvKernel1x1Frame, FilterCount, OutputCount

        ENDM

;++
;
; SconvKernel1x1Function
;
;   This macro generates code for the inner convolution kernel for the special
;   case where the kernel dimensions are 1.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string for function tags.
;
;   BiasFilter - Supplies a non-blank value if the address of the filter buffer
;       should be biased to point to the middle of a OIhw8i8o block in order to
;       reduce the code size from relative byte offsets.
;
;--

SconvKernel1x1Function MACRO Isa, BiasFilter

;++
;
; Routine Description:
;
; Arguments:
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConv1x1FloatKernel&Isa, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r14
        push_reg r12
        alloc_stack (SconvKernel1x1Frame.SavedR12)

        save_xmm128 xmm6,SconvKernel1x1Frame.SavedXmm6
        save_xmm128 xmm7,SconvKernel1x1Frame.SavedXmm7
        save_xmm128 xmm8,SconvKernel1x1Frame.SavedXmm8
        save_xmm128 xmm9,SconvKernel1x1Frame.SavedXmm9
        save_xmm128 xmm10,SconvKernel1x1Frame.SavedXmm10
        save_xmm128 xmm11,SconvKernel1x1Frame.SavedXmm11
        save_xmm128 xmm12,SconvKernel1x1Frame.SavedXmm12
        save_xmm128 xmm13,SconvKernel1x1Frame.SavedXmm13
        save_xmm128 xmm14,SconvKernel1x1Frame.SavedXmm14
        save_xmm128 xmm15,SconvKernel1x1Frame.SavedXmm15

        END_PROLOGUE

        mov     rbp,rcx
IFNB <BiasFilter>
        lea     r12,[rdx+4*8*4]
ELSE
        mov     r12,rdx
ENDIF
        mov     r10,SconvKernel1x1Frame.OutputCount[rsp]
        mov     r11,SconvKernel1x1Frame.FilterCount[rsp]
        mov     rsi,SconvKernel1x1Frame.FilterStride[rsp]
        mov     rdi,SconvKernel1x1Frame.AdjustCount[rsp]

;
; Process the specified number of filter rows.
;

        cmp     r11,3
        je      ProcessFilterCount3
        jb      ProcessFilterCountLessThan3
        Process1x1FilterCountN 4
        jmp     ExitKernel

ProcessFilterCount3:
        Process1x1FilterCountN 3
        jmp     ExitKernel

ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      ProcessFilterCount1
        Process1x1FilterCountN 2
        jmp     ExitKernel

ProcessFilterCount1:
        Process1x1FilterCountN 1

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,SconvKernel1x1Frame.SavedXmm6[rsp]
        movaps  xmm7,SconvKernel1x1Frame.SavedXmm7[rsp]
        movaps  xmm8,SconvKernel1x1Frame.SavedXmm8[rsp]
        movaps  xmm9,SconvKernel1x1Frame.SavedXmm9[rsp]
        movaps  xmm10,SconvKernel1x1Frame.SavedXmm10[rsp]
        movaps  xmm11,SconvKernel1x1Frame.SavedXmm11[rsp]
        movaps  xmm12,SconvKernel1x1Frame.SavedXmm12[rsp]
        movaps  xmm13,SconvKernel1x1Frame.SavedXmm13[rsp]
        movaps  xmm14,SconvKernel1x1Frame.SavedXmm14[rsp]
        movaps  xmm15,SconvKernel1x1Frame.SavedXmm15[rsp]
        add     rsp,(SconvKernel1x1Frame.SavedR12)

        BEGIN_EPILOGUE

        pop     r12
        pop     r14
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConv1x1FloatKernel&Isa&, _TEXT

        ENDM
