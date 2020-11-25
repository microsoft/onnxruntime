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
; Define the convolution kernel flags.
;

MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     EQU     00000001h
MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         EQU     00000002h
MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       EQU     00000004h
MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      EQU     00000008h

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

SconvKernelDepthwiseFrame STRUCT

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
        InputStride QWORD ?
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

SconvKernelDepthwiseFrame ENDS

SconvKernelDepthwiseSingleFrame STRUCT

        ReturnAddress QWORD ?
        KernelFrame SconvKernelDepthwiseFrame <>

SconvKernelDepthwiseSingleFrame ENDS

SconvKernelPointwiseFrame STRUCT

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
        PreviousP1Home QWORD ?              ; Input
        PreviousP2Home QWORD ?              ; Filter
        PreviousP3Home QWORD ?              ; Output
        PreviousP4Home QWORD ?              ; StrideWidth
        InputChannels QWORD ?
        FilterCount QWORD ?
        InputStride QWORD ?
        FilterStride QWORD ?
        OutputStride QWORD ?
        OutputCount QWORD ?
        Bias QWORD ?
        Flags QWORD ?

SconvKernelPointwiseFrame ENDS

;
; Macro Description:
;
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks.
;
;   OutputCount=1 generates special case code to handle padding blocks. All
;   other output counts assume no padding.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string for function tags.
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   BlockSize - Supplies the number of elements per block.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description) when
;       KernelType!=Depthwise. Supplies the address of the filter buffer when
;       KernelType=Depthwise.
;
;   rbp - Supplies the DilationWidth parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r15 - Supplies the InputStride parameter (see function description).
;

ProcessOutputCountN MACRO Isa, KernelFrame, KernelType, BlockSize, FilterCount, OutputCount

        LOCAL   ProcessNextRow
        LOCAL   ProcessNextColumn
        LOCAL   HandlePostProcessing
        LOCAL   SkipOverPadding

        mov     rcx,rdi
IFIDNI <KernelType>, <Depthwise>
        mov     rdx,rsi
ELSE
        mov     rdx,KernelFrame.PreviousP2Home[rsp]
ENDIF
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
IFIDNI <KernelType>, <Nchwc>
IF BlockSize EQ 16
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>
            ComputeBlock KernelType, FilterCount, OutputCount, Index*16*4, Index*4
        ENDM
ELSE
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7>
            ComputeBlock KernelType, FilterCount, OutputCount, (Index-4)*8*4, Index*4
        ENDM
ENDIF
ELSE
        ComputeBlock KernelType, FilterCount, OutputCount, 0, 0
ENDIF

SkipOverPadding:
        add     rcx,rbp                     ; advance input by dilation width
IFIDNI <KernelType>, <Nchwc>
        add     rdx,BlockSize*BlockSize*4   ; advance filter by 8i8o/16i16o block
ELSE
        add     rdx,BlockSize*4             ; advance filter by 8o/16o block
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
        mov     edx,DWORD PTR KernelFrame.Flags[rsp]
IF FilterCount GT 1
        mov     rax,KernelFrame.OutputStride[rsp]
ENDIF
        mov     rcx,KernelFrame.Bias[rsp]
        call    MlasConvPostProcessFloat&Isa&Filter&FilterCount&Output&OutputCount

        ENDM

;
; Macro Description:
;
;   This macro generates code for the inner convolution kernel.
;
; Arguments:
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   BlockSize - Supplies the number of elements per block.
;
;   Isa - Supplies the instruction set architecture string for function tags.
;
;   BiasFilter - Supplies a non-blank value if the address of the filter buffer
;       should be biased to point to the middle of a OIhw8i8o block in order to
;       reduce the code size from relative byte offsets.
;

SconvKernelFunction MACRO KernelType, BlockSize, Isa, BiasFilter

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
;       especially post calculation options.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConv&KernelType&FloatKernel&Isa&, _TEXT

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

        mov     rdi,rcx
IFNB <BiasFilter>
        add_immed rdx,4*8*4
ENDIF
        mov     SconvKernelFrame.PreviousP2Home[rsp],rdx
        mov     rsi,SconvKernelFrame.FilterStride[rsp]
        mov     rbp,SconvKernelFrame.DilationWidth[rsp]
        mov     r11,SconvKernelFrame.FilterCount[rsp]
        mov     r15,SconvKernelFrame.InputStride[rsp]

;
; Process the specified number of filter rows.
;

        cmp     r11,3
        je      ProcessFilterCount3
        jb      ProcessFilterCountLessThan3
        ProcessFilterCountN SconvKernelFrame, KernelType, 4
        jmp     ExitKernel

ProcessFilterCount3:
        ProcessFilterCountN SconvKernelFrame, KernelType, 3
        jmp     ExitKernel

ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      ProcessFilterCount1
        ProcessFilterCountN SconvKernelFrame, KernelType, 2
        jmp     ExitKernel

ProcessFilterCount1:
        ProcessFilterCountN SconvKernelFrame, KernelType, 1

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

        NESTED_END MlasConv&KernelType&FloatKernel&Isa&, _TEXT

IFDIFI <Isa>, <Sse>

;
; Generate out-of-band helpers for handling output blocks involving padding.
;

        IRP     FilterCount, <1, 2, 3, 4>

        LEAF_ENTRY MlasConv&KernelType&FloatSingle&Isa&Filter&FilterCount, _TEXT

ProcessNextOutputCount:
        ProcessOutputCountN Isa, SconvKernelSingleFrame.KernelFrame, KernelType, BlockSize, FilterCount, 1
        add     rdi,r9                      ; advance input by 1 element
        dec     r10                         ; decrement output count remaining
        jnz     ProcessNextOutputCount
        ret

        LEAF_END MlasConv&KernelType&FloatSingle&Isa&Filter&FilterCount, _TEXT

        ENDM

ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code for the inner convolution kernel for the special
;   case of a depthwise separable convolution.
;
; Arguments:
;
;   BlockSize - Supplies the number of elements per block.
;
;   Isa - Supplies the instruction set architecture string for function tags.
;

SconvKernelDepthwiseFunction MACRO BlockSize, Isa

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute a convolution for the elements
;   of an output row for a set of filter rows.
;
;   Depthwise seperable convolutions are a form of grouped convolution where
;   the number of input and output channels per group are one.
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
;   InputStride - Supplies the length in bytes to advance the input buffer to
;       the next input row.
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
;       especially post calculation options.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConvDepthwiseFloatKernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r15
        push_reg r14
        push_reg r13
        push_reg r12
        alloc_stack (SconvKernelDepthwiseFrame.SavedR12)

        save_xmm128 xmm6,SconvKernelDepthwiseFrame.SavedXmm6
        save_xmm128 xmm7,SconvKernelDepthwiseFrame.SavedXmm7
        save_xmm128 xmm8,SconvKernelDepthwiseFrame.SavedXmm8
        save_xmm128 xmm9,SconvKernelDepthwiseFrame.SavedXmm9
        save_xmm128 xmm10,SconvKernelDepthwiseFrame.SavedXmm10
        save_xmm128 xmm11,SconvKernelDepthwiseFrame.SavedXmm11
        save_xmm128 xmm12,SconvKernelDepthwiseFrame.SavedXmm12
        save_xmm128 xmm13,SconvKernelDepthwiseFrame.SavedXmm13
        save_xmm128 xmm14,SconvKernelDepthwiseFrame.SavedXmm14
        save_xmm128 xmm15,SconvKernelDepthwiseFrame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx
        mov     rsi,rdx
        mov     rbp,SconvKernelDepthwiseFrame.DilationWidth[rsp]
        mov     r15,SconvKernelDepthwiseFrame.InputStride[rsp]

;
; Process the specified number of filter rows.
;

        ProcessFilterCountN SconvKernelDepthwiseFrame, Depthwise, 1

;
; Restore non-volatile registers and return.
;

ExitKernel:
IFDIFI <Isa>, <Sse>
        vzeroupper
ENDIF
        movaps  xmm6,SconvKernelDepthwiseFrame.SavedXmm6[rsp]
        movaps  xmm7,SconvKernelDepthwiseFrame.SavedXmm7[rsp]
        movaps  xmm8,SconvKernelDepthwiseFrame.SavedXmm8[rsp]
        movaps  xmm9,SconvKernelDepthwiseFrame.SavedXmm9[rsp]
        movaps  xmm10,SconvKernelDepthwiseFrame.SavedXmm10[rsp]
        movaps  xmm11,SconvKernelDepthwiseFrame.SavedXmm11[rsp]
        movaps  xmm12,SconvKernelDepthwiseFrame.SavedXmm12[rsp]
        movaps  xmm13,SconvKernelDepthwiseFrame.SavedXmm13[rsp]
        movaps  xmm14,SconvKernelDepthwiseFrame.SavedXmm14[rsp]
        movaps  xmm15,SconvKernelDepthwiseFrame.SavedXmm15[rsp]
        add     rsp,(SconvKernelDepthwiseFrame.SavedR12)

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

        NESTED_END MlasConvDepthwiseFloatKernel&Isa&, _TEXT

IFDIFI <Isa>, <Sse>

;
; Generate out-of-band helpers for handling output blocks involving padding.
;

        LEAF_ENTRY MlasConvDepthwiseFloatSingle&Isa&Filter1, _TEXT

ProcessNextOutputCount:
        ProcessOutputCountN Isa, SconvKernelDepthwiseSingleFrame.KernelFrame, Depthwise, BlockSize, 1, 1
        add     rdi,r9                      ; advance input by 1 element
        dec     r10                         ; decrement output count remaining
        jnz     ProcessNextOutputCount
        ret

        LEAF_END MlasConvDepthwiseFloatSingle&Isa&Filter1, _TEXT

ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute the convolution for a vector of input
;   blocks and a vector of filter blocks to produce a matrix of output blocks
;   for a pointwise convolution.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string for function tags.
;
;   BlockSize - Supplies the number of elements per block.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rbp - Supplies the InputStride parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r12 - Supplies the address of the filter buffer.
;

ProcessPointwiseOutputCountN MACRO Isa, BlockSize, FilterCount, OutputCount

        LOCAL   ProcessNextInputBlock
        LOCAL   SkipAccumulateOutput
        LOCAL   SkipBiasAddition
        LOCAL   SkipReluActivation

        mov     rcx,rdi
        mov     rdx,r12
        mov     r11,SconvKernelPointwiseFrame.InputChannels[rsp]
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
            ComputeBlock Pointwise, FilterCount, OutputCount, Index*16*4, Index*4
        ENDM
ELSE
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7>
            ComputeBlock Pointwise, FilterCount, OutputCount, (Index-4)*8*4, Index*4
        ENDM
ENDIF
        add     rcx,rbp                     ; advance input to next channel block
        add     rdx,BlockSize*BlockSize*4   ; advance filter by 8i8o/16i16o block
        dec     r11                         ; decrement input blocks remaining
        jnz     ProcessNextInputBlock

;
; Handle post processing of the output block.
;

        mov     edx,DWORD PTR SconvKernelPointwiseFrame.Flags[rsp]
IF FilterCount GT 1
        mov     rax,SconvKernelPointwiseFrame.OutputStride[rsp]
ENDIF
        mov     rcx,SconvKernelPointwiseFrame.Bias[rsp]
        call    MlasConvPostProcessFloat&Isa&Filter&FilterCount&Output&OutputCount

        ENDM

;++
;
; Macro Description:
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

SconvKernelPointwiseFunction MACRO Isa, BiasFilter

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute a convolution for the elements
;   of an output row for a set of filter rows.
;
;   Pointwise convolutions have a kernel size of one. To simplify this
;   implementation, no input padding is allowed, which matches typical usage in
;   models.
;
; Arguments:
;
;   Input (rcx) - Supplies the address of the input buffer.
;
;   Filter (rdx) - Supplies the address of the filter buffer.
;
;   Output (r8) - Supplies the address of the output buffer.
;
;   StrideWidth (r9) - Supplies the length in bytes of the blocked stride width.
;
;   InputChannels - Supplies the number of input channels to process.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   InputStride - Supplies the length in bytes to advance the input buffer to
;       the next input channel of the same input row.
;
;   FilterStride - Supplies the length in bytes to advance the filter buffer
;       to the next set of filters.
;
;   OutputStride - Supplies the length in bytes to advance the output buffer
;       to the next output address associated with the next set of filters.
;
;   OutputCount - Supplies the number of output elements.
;
;   Bias - Supplies the address of the bias buffer.
;
;   Flags - Supplies additional flags controlling the convolution operation,
;       especially post calculation options.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConvPointwiseFloatKernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r14
        push_reg r12
        alloc_stack (SconvKernelPointwiseFrame.SavedR12)

        save_xmm128 xmm6,SconvKernelPointwiseFrame.SavedXmm6
        save_xmm128 xmm7,SconvKernelPointwiseFrame.SavedXmm7
        save_xmm128 xmm8,SconvKernelPointwiseFrame.SavedXmm8
        save_xmm128 xmm9,SconvKernelPointwiseFrame.SavedXmm9
        save_xmm128 xmm10,SconvKernelPointwiseFrame.SavedXmm10
        save_xmm128 xmm11,SconvKernelPointwiseFrame.SavedXmm11
        save_xmm128 xmm12,SconvKernelPointwiseFrame.SavedXmm12
        save_xmm128 xmm13,SconvKernelPointwiseFrame.SavedXmm13
        save_xmm128 xmm14,SconvKernelPointwiseFrame.SavedXmm14
        save_xmm128 xmm15,SconvKernelPointwiseFrame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx
IFNB <BiasFilter>
        lea     r12,[rdx+4*8*4]
ELSE
        mov     r12,rdx
ENDIF
        mov     r10,SconvKernelPointwiseFrame.OutputCount[rsp]
        mov     r11,SconvKernelPointwiseFrame.FilterCount[rsp]
        mov     rsi,SconvKernelPointwiseFrame.FilterStride[rsp]
        mov     rbp,SconvKernelPointwiseFrame.InputStride[rsp]

;
; Process the specified number of filter rows.
;

        cmp     r11,3
        je      ProcessFilterCount3
        jb      ProcessFilterCountLessThan3
        ProcessPointwiseFilterCountN 4
        jmp     ExitKernel

ProcessFilterCount3:
        ProcessPointwiseFilterCountN 3
        jmp     ExitKernel

ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      ProcessFilterCount1
        ProcessPointwiseFilterCountN 2
        jmp     ExitKernel

ProcessFilterCount1:
        ProcessPointwiseFilterCountN 1

;
; Restore non-volatile registers and return.
;

ExitKernel:
IFDIFI <Isa>, <Sse>
        vzeroupper
ENDIF
        movaps  xmm6,SconvKernelPointwiseFrame.SavedXmm6[rsp]
        movaps  xmm7,SconvKernelPointwiseFrame.SavedXmm7[rsp]
        movaps  xmm8,SconvKernelPointwiseFrame.SavedXmm8[rsp]
        movaps  xmm9,SconvKernelPointwiseFrame.SavedXmm9[rsp]
        movaps  xmm10,SconvKernelPointwiseFrame.SavedXmm10[rsp]
        movaps  xmm11,SconvKernelPointwiseFrame.SavedXmm11[rsp]
        movaps  xmm12,SconvKernelPointwiseFrame.SavedXmm12[rsp]
        movaps  xmm13,SconvKernelPointwiseFrame.SavedXmm13[rsp]
        movaps  xmm14,SconvKernelPointwiseFrame.SavedXmm14[rsp]
        movaps  xmm15,SconvKernelPointwiseFrame.SavedXmm15[rsp]
        add     rsp,(SconvKernelPointwiseFrame.SavedR12)

        BEGIN_EPILOGUE

        pop     r12
        pop     r14
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConvPointwiseFloatKernel&Isa&, _TEXT

        ENDM
