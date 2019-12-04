;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelSse2.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision convolution
;   operation.
;
;   This implementation uses SSE2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SconvKernelCommon.inc
        .list

;
; Macro Description:
;
;   This macro generates code to clear the block accumulators.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   xmm0-xmm7 - Supplies the block accumulators.
;

ClearBlock MACRO FilterCount, OutputCount

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <xorps xmm0,xmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <xorps xmm1,xmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <xorps xmm2,xmm2>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <xorps xmm3,xmm3>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <xorps xmm4,xmm4>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <xorps xmm5,xmm5>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <xorps xmm6,xmm6>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <xorps xmm7,xmm7>

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
;   xmm0-xmm7 - Supplies the block accumulators.
;

ComputeBlock MACRO KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

IFIDNI <KernelType>, <Depthwise>
        movups  xmm8,XMMWORD PTR [rdx]
        movups  xmm9,XMMWORD PTR [rdx+16]
        movups  xmm10,XMMWORD PTR [rcx]
        movups  xmm11,XMMWORD PTR [rcx+16]
        mulps   xmm8,xmm10
        addps   xmm0,xmm8
        mulps   xmm9,xmm11
        addps   xmm1,xmm9
ELSE
        EmitIfCountGE OutputCount, 1, <movss xmm12,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 1, <shufps xmm12,xmm12,0>
        EmitIfCountGE FilterCount, 1, <movups xmm8,XMMWORD PTR [rdx+VectorOffset]>
        EmitIfCountGE FilterCount, 1, <movups xmm9,XMMWORD PTR [rdx+VectorOffset+16]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <mulps xmm8,xmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm0,xmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <mulps xmm9,xmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm1,xmm9>
        EmitIfCountGE FilterCount, 2, <movups xmm8,XMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 2, <movups xmm9,XMMWORD PTR [rdx+rsi+VectorOffset+16]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <mulps xmm8,xmm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm2,xmm8>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <mulps xmm9,xmm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm3,xmm9>
        EmitIfCountGE FilterCount, 3, <movups xmm8,XMMWORD PTR [rbx+VectorOffset]>
        EmitIfCountGE FilterCount, 3, <movups xmm9,XMMWORD PTR [rbx+VectorOffset+16]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <mulps xmm8,xmm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm4,xmm8>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <mulps xmm9,xmm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm5,xmm9>
        EmitIfCountGE FilterCount, 4, <movups xmm8,XMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 4, <movups xmm9,XMMWORD PTR [rbx+rsi+VectorOffset+16]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <mulps xmm8,xmm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm6,xmm8>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <mulps xmm9,xmm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm7,xmm9>
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
;   rsi - Supplies the FilterStride parameter (see function description).
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

        LOCAL   ProcessNextOutputCount

        mov     r10,KernelFrame.OutputCountLeftPad[rsp]
        add     r10,KernelFrame.OutputCount[rsp]
        add     r10,KernelFrame.OutputCountRightPad[rsp]

ProcessNextOutputCount:
        ProcessOutputCountN Sse, KernelFrame, KernelType, 8, FilterCount, 1
        add     rdi,r9                      ; advance input by 1 element
        dec     r10
        jnz     ProcessNextOutputCount

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

        LOCAL   ProcessNextOutputCount

ProcessNextOutputCount:
        ProcessPointwiseOutputCountN Sse, 8, FilterCount, 1
        add     rdi,r9                      ; advance input by 1 element
        dec     r10
        jnz     ProcessNextOutputCount

        ENDM

;
; Generate the convolution kernels.
;

SconvKernelFunction Nchw, 8, Sse
SconvKernelFunction Nchwc, 8, Sse, BiasFilter
SconvKernelDepthwiseFunction 8, Sse
SconvKernelPointwiseFunction Sse, BiasFilter

;
; Macro Description:
;
;   This macro generates code to process an output block after the inner
;   convolution kernel has executed and then stores the output block to the
;   output buffer.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;

        IRP     FilterCount, <1, 2, 3, 4>
        IRP     OutputCount, <1>

        LEAF_ENTRY MlasConvPostProcessFloatSseFilter&FilterCount&Output&OutputCount, _TEXT

IF FilterCount GT 2
        lea     rbx,[r8+rax*2]              ; compute output plus 2 rows
ENDIF

;
; Test if the existing contents of the output buffer should be accumulated
; with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        jz      SkipAccumulateOutput
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups xmm8,XMMWORD PTR [r8]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups xmm9,XMMWORD PTR [r8+16]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups xmm10,XMMWORD PTR [r8+rax]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups xmm11,XMMWORD PTR [r8+rax+16]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups xmm12,XMMWORD PTR [rbx]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups xmm13,XMMWORD PTR [rbx+16]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups xmm14,XMMWORD PTR [rbx+rax]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups xmm15,XMMWORD PTR [rbx+rax+16]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm0,xmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm1,xmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm2,xmm10>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm3,xmm11>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm4,xmm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm5,xmm13>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm6,xmm14>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm7,xmm15>

SkipAccumulateOutput:

;
; Test if the bias buffer should be accumulated with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        jz      SkipBiasAddition
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups xmm8,XMMWORD PTR [rcx]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups xmm9,XMMWORD PTR [rcx+16]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups xmm10,XMMWORD PTR [rcx+32]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups xmm11,XMMWORD PTR [rcx+48]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups xmm12,XMMWORD PTR [rcx+64]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups xmm13,XMMWORD PTR [rcx+80]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups xmm14,XMMWORD PTR [rcx+96]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups xmm15,XMMWORD PTR [rcx+112]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm0,xmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <addps xmm1,xmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm2,xmm10>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <addps xmm3,xmm11>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm4,xmm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <addps xmm5,xmm13>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm6,xmm14>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <addps xmm7,xmm15>

SkipBiasAddition:

;
; Test for fused ReLU activation.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        jz      SkipReluActivation
        xorps   xmm15,xmm15
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <maxps xmm0,xmm15>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <maxps xmm1,xmm15>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <maxps xmm2,xmm15>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <maxps xmm3,xmm15>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <maxps xmm4,xmm15>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <maxps xmm5,xmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <maxps xmm6,xmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <maxps xmm7,xmm15>

SkipReluActivation:

;
; Store the output block in the output buffer.
;

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups XMMWORD PTR [r8],xmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <movups XMMWORD PTR [r8+16],xmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups XMMWORD PTR [r8+rax],xmm2>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <movups XMMWORD PTR [r8+rax+16],xmm3>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups XMMWORD PTR [rbx],xmm4>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <movups XMMWORD PTR [rbx+16],xmm5>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups XMMWORD PTR [rbx+rax],xmm6>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <movups XMMWORD PTR [rbx+rax+16],xmm7>
        add_immed r8,OutputCount*8*4        ; advance output by N nchw8c blocks
        ret

        LEAF_END MlasConvPostProcessFloatSseFilter&FilterCount&Output&OutputCount, _TEXT

        ENDM
        ENDM

        END
