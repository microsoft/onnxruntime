;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelFma3.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision convolution
;   operation.
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SconvKernelAvxCommon.inc
        .list

;
; Share the post process functions with the AVX implementation.
;

        IRP     FilterCount, <1, 2, 3, 4>
        IRP     OutputCount, <1, 2, 3>

        EXTERN  MlasConvPostProcessFloatFma3Filter&FilterCount&Output&OutputCount:NEAR

        ENDM
        ENDM

;
; Macro Description:
;
;   This macro multiplies and accumulates for FilterCount by OutputCount block
;   of the output buffer.
;
; Arguments:
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
;   VectorOffset - Supplies the byte offset from the filter buffer to fetch
;       elements.
;
;   BroadcastOffset - Supplies the byte offset from the input buffer to fetch
;       elements.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the input buffer.
;
;   rdx - Supplies the address of the filter buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rbx - Supplies the address of the filter buffer plus 2 * FilterStride.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   ymm0-ymm11 - Supplies the block accumulators.
;

ComputeBlock MACRO KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

IFIDNI <KernelType>, <Depthwise>
        vmovups ymm12,YMMWORD PTR [rdx]
        EmitIfCountGE OutputCount, 1, <vfmadd231ps ymm0,ymm12,DWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 2, <vfmadd231ps ymm4,ymm12,DWORD PTR [rcx+r9]>
        EmitIfCountGE OutputCount, 3, <vfmadd231ps ymm8,ymm12,DWORD PTR [rcx+r9*2]>
ELSE
        EmitIfCountGE OutputCount, 1, <vbroadcastss ymm13,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vbroadcastss ymm14,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vbroadcastss ymm15,DWORD PTR [rcx+r9*2+BroadcastOffset]>
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vfmadd231ps ymm0,ymm13,YMMWORD PTR [rdx+VectorOffset]>
        EmitIfCountGE FilterCount, 2, <vfmadd231ps ymm1,ymm13,YMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 3, <vfmadd231ps ymm2,ymm13,YMMWORD PTR [rbx+VectorOffset]>
        EmitIfCountGE FilterCount, 4, <vfmadd231ps ymm3,ymm13,YMMWORD PTR [rbx+rsi+VectorOffset]>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups ymm12,YMMWORD PTR [rdx+VectorOffset]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vfmadd231ps ymm0,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vfmadd231ps ymm4,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vfmadd231ps ymm8,ymm15,ymm12>
        EmitIfCountGE FilterCount, 2, <vmovups ymm12,YMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vfmadd231ps ymm1,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vfmadd231ps ymm5,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vfmadd231ps ymm9,ymm15,ymm12>
        EmitIfCountGE FilterCount, 3, <vmovups ymm12,YMMWORD PTR [rbx+VectorOffset]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vfmadd231ps ymm2,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vfmadd231ps ymm6,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vfmadd231ps ymm10,ymm15,ymm12>
        EmitIfCountGE FilterCount, 4, <vmovups ymm12,YMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vfmadd231ps ymm3,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vfmadd231ps ymm7,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vfmadd231ps ymm11,ymm15,ymm12>
ENDIF
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute the convolution for a specified number
;   of filter rows.
;
; Arguments:
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
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

ProcessFilterCountN MACRO KernelFrame, KernelType, FilterCount

        LOCAL   ProcessOutputCountLeftPad
        LOCAL   ProcessOutputCount
        LOCAL   ProcessNextOutputCountBy3
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCount1
        LOCAL   ProcessOutputCountRightPadAndRemaining

;
; Process the output blocks that include left padding.
;

        mov     r10,KernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      ProcessOutputCount
        call    MlasConv&KernelType&FloatSingleFma3Filter&FilterCount

;
; Process the output blocks that do not include any padding.
;

ProcessOutputCount:
        mov     r10,KernelFrame.OutputCount[rsp]
        sub     r10,3
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy3:
        ProcessOutputCountN Fma3, KernelFrame, KernelType, 8, FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jae     ProcessNextOutputCountBy3

ProcessRemainingOutputCount:
        add     r10,3                       ; correct for over-subtract above
        jz      ProcessOutputCountRightPadAndRemaining
        cmp     r10,2
        jb      ProcessOutputCountRightPadAndRemaining
        ProcessOutputCountN Fma3, KernelFrame, KernelType, 8, FilterCount, 2
        lea     rdi,[rdi+r9*2]              ; advance input by 2 elements
        sub     r10,2

;
; Process the output blocks that include right padding plus any remaining output
; blocks from above.
;

ProcessOutputCountRightPadAndRemaining:
        add     r10,KernelFrame.OutputCountRightPad[rsp]
        jz      ExitKernel
        call    MlasConv&KernelType&FloatSingleFma3Filter&FilterCount

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute the convolution for a specified number
;   of filter rows for a pointwise convolution.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
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
;   r10 - Supplies the OutputCount parameter (see function description).
;
;   r12 - Supplies the address of the filter buffer.
;

ProcessPointwiseFilterCountN MACRO FilterCount

        LOCAL   ProcessNextOutputCountBy3
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCount1

        sub     r10,3
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy3:
        ProcessPointwiseOutputCountN Fma3, 8, FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jae     ProcessNextOutputCountBy3

ProcessRemainingOutputCount:
        add     r10,3                       ; correct for over-subtract above
        jz      ExitKernel
        cmp     r10,2
        jb      ProcessRemainingOutputCount1
        ProcessPointwiseOutputCountN Fma3, 8, FilterCount, 2
        jmp     ExitKernel

ProcessRemainingOutputCount1:
        ProcessPointwiseOutputCountN Fma3, 8, FilterCount, 1

        ENDM

;
; Generate the convolution kernels.
;

SconvKernelFunction Nchw, 8, Fma3
SconvKernelFunction Nchwc, 8, Fma3, BiasFilter
SconvKernelDepthwiseFunction 8, Fma3
SconvKernelPointwiseFunction Fma3, BiasFilter

        END
