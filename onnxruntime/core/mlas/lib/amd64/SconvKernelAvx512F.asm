;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision convolution
;   operation.
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SconvKernelCommon.inc
        .list

        EXTERN  MlasLogisticConstants:NEAR

MLAS_LOGISTIC_CONSTANTS_LOWER_RANGE EQU 0
MLAS_LOGISTIC_CONSTANTS_UPPER_RANGE EQU 4
MLAS_LOGISTIC_CONSTANTS_ALPHA_9     EQU 8
MLAS_LOGISTIC_CONSTANTS_ALPHA_7     EQU 12
MLAS_LOGISTIC_CONSTANTS_ALPHA_5     EQU 16
MLAS_LOGISTIC_CONSTANTS_ALPHA_3     EQU 20
MLAS_LOGISTIC_CONSTANTS_ALPHA_1     EQU 24
MLAS_LOGISTIC_CONSTANTS_BETA_10     EQU 28
MLAS_LOGISTIC_CONSTANTS_BETA_8      EQU 32
MLAS_LOGISTIC_CONSTANTS_BETA_6      EQU 36
MLAS_LOGISTIC_CONSTANTS_BETA_4      EQU 40
MLAS_LOGISTIC_CONSTANTS_BETA_2      EQU 44
MLAS_LOGISTIC_CONSTANTS_BETA_0      EQU 48
MLAS_LOGISTIC_CONSTANTS_ONE_HALF    EQU 52

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
;   zmm0-zmm23 - Supplies the block accumulators.
;

ClearBlock MACRO FilterCount, OutputCount

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vpxord zmm0,zmm0,zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vpxord zmm4,zmm4,zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vpxord zmm8,zmm8,zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vpxord zmm12,zmm12,zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vpxord zmm16,zmm16,zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vpxord zmm20,zmm20,zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vpxord zmm1,zmm1,zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vpxord zmm5,zmm5,zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vpxord zmm9,zmm9,zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vpxord zmm13,zmm13,zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vpxord zmm17,zmm17,zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vpxord zmm21,zmm21,zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vpxord zmm2,zmm2,zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vpxord zmm6,zmm6,zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vpxord zmm10,zmm10,zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vpxord zmm14,zmm14,zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vpxord zmm18,zmm18,zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vpxord zmm22,zmm22,zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vpxord zmm3,zmm3,zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vpxord zmm7,zmm7,zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vpxord zmm11,zmm11,zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vpxord zmm15,zmm15,zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vpxord zmm19,zmm19,zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vpxord zmm23,zmm23,zmm23>

        ENDM

ApplySiluAvx512F MACRO Accumulator

        vmovaps zmm24,Accumulator
        vcmpps  k1,zmm24,zmm24,3

        vmaxps  Accumulator,Accumulator,DWORD BCST [r10+MLAS_LOGISTIC_CONSTANTS_LOWER_RANGE]
        vminps  Accumulator,Accumulator,DWORD BCST [r10+MLAS_LOGISTIC_CONSTANTS_UPPER_RANGE]
        vmulps  zmm25,Accumulator,Accumulator

        vbroadcastss zmm26,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ALPHA_9]
        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ALPHA_7]
        vfmadd213ps zmm26,zmm25,zmm27
        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ALPHA_5]
        vfmadd213ps zmm26,zmm25,zmm27
        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ALPHA_3]
        vfmadd213ps zmm26,zmm25,zmm27
        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ALPHA_1]
        vfmadd213ps zmm26,zmm25,zmm27
        vmulps  zmm26,zmm26,Accumulator

        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_10]
        vbroadcastss zmm28,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_8]
        vfmadd213ps zmm27,zmm25,zmm28
        vbroadcastss zmm28,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_6]
        vfmadd213ps zmm27,zmm25,zmm28
        vbroadcastss zmm28,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_4]
        vfmadd213ps zmm27,zmm25,zmm28
        vbroadcastss zmm28,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_2]
        vfmadd213ps zmm27,zmm25,zmm28
        vbroadcastss zmm28,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_BETA_0]
        vfmadd213ps zmm27,zmm25,zmm28

        vdivps  zmm26,zmm26,zmm27
        vbroadcastss zmm27,DWORD PTR [r10+MLAS_LOGISTIC_CONSTANTS_ONE_HALF]
        vaddps  zmm26,zmm26,zmm27
        vpxord  zmm28,zmm28,zmm28
        vmaxps  zmm26,zmm26,zmm28
        vaddps  zmm27,zmm27,zmm27
        vminps  zmm26,zmm26,zmm27
        vmulps  Accumulator,zmm24,zmm26
        vmovaps Accumulator{k1},zmm24

        ENDM

;
; Macro Description:
;
;   This macro multiplies and accumulates for a FilterCount by OutputCount
;   block of the output buffer using a native packed pointwise filter layout.
;
;   The filter layout is organized as:
;       [InputChannelBlock][InputChannelWithinBlock][FilterRow(4)][OutputLane(16)]
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
;   BroadcastOffset - Supplies the byte offset from the input buffer to fetch
;       elements.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the input buffer.
;
;   rdx - Supplies the address of the packed filter slice for the current input
;       channel within the current input block.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r14 - Supplies the address of the input buffer plus 3 * StrideWidth.
;
;   zmm0-zmm23 - Supplies the block accumulators.
;

ComputePackedPointwiseBlock MACRO FilterCount, OutputCount, BroadcastOffset

        EmitIfCountGE OutputCount, 1, <vbroadcastss zmm26,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vbroadcastss zmm27,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vbroadcastss zmm28,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vbroadcastss zmm29,DWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vbroadcastss zmm30,DWORD PTR [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vbroadcastss zmm31,DWORD PTR [r14+r9*2+BroadcastOffset]>

        EmitIfCountGE FilterCount, 1, <vmovups zmm24,ZMMWORD PTR [rdx+0*16*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vfmadd231ps zmm0,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vfmadd231ps zmm4,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vfmadd231ps zmm8,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vfmadd231ps zmm12,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vfmadd231ps zmm16,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vfmadd231ps zmm20,zmm31,zmm24>

        EmitIfCountGE FilterCount, 2, <vmovups zmm24,ZMMWORD PTR [rdx+1*16*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vfmadd231ps zmm1,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vfmadd231ps zmm5,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vfmadd231ps zmm9,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vfmadd231ps zmm13,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vfmadd231ps zmm17,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vfmadd231ps zmm21,zmm31,zmm24>

        EmitIfCountGE FilterCount, 3, <vmovups zmm24,ZMMWORD PTR [rdx+2*16*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vfmadd231ps zmm2,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vfmadd231ps zmm6,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vfmadd231ps zmm10,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vfmadd231ps zmm14,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vfmadd231ps zmm18,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vfmadd231ps zmm22,zmm31,zmm24>

        EmitIfCountGE FilterCount, 4, <vmovups zmm24,ZMMWORD PTR [rdx+3*16*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vfmadd231ps zmm3,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vfmadd231ps zmm7,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vfmadd231ps zmm11,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vfmadd231ps zmm15,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vfmadd231ps zmm19,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vfmadd231ps zmm23,zmm31,zmm24>

        add     rdx,4*16*4

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute a native packed pointwise
;   convolution for a specified number of filter rows.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;

ProcessPointwiseOutputCountPackedN MACRO FilterCount, OutputCount

        LOCAL   ProcessNextInputBlock

        mov     rcx,rdi
        mov     rdx,r12
        mov     r11,SconvKernelPointwiseFrame.InputChannels[rsp]
        ClearBlock FilterCount, OutputCount

ProcessNextInputBlock:
IF OutputCount GT 3
        lea     r14,[r9+r9*2]
        add     r14,rcx
ENDIF
        IRP     Index, <0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>
            ComputePackedPointwiseBlock FilterCount, OutputCount, Index*4
        ENDM
        add     rcx,rbp
        dec     r11
        jnz     ProcessNextInputBlock

        mov     edx,DWORD PTR SconvKernelPointwiseFrame.Flags[rsp]
IF FilterCount GT 1
        mov     rax,SconvKernelPointwiseFrame.OutputStride[rsp]
ENDIF
        mov     rcx,SconvKernelPointwiseFrame.Bias[rsp]
        call    MlasConvPostProcessFloatAvx512FPointwiseFilter&FilterCount&Output&OutputCount

        ENDM

ProcessPointwiseFilterCountPackedN MACRO FilterCount

        LOCAL   ProcessNextOutputCountBy6
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCountLessThan3
        LOCAL   ProcessRemainingOutputCount1

        sub     r10,6
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy6:
        ProcessPointwiseOutputCountPackedN FilterCount, 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]
        sub     r10,6
        jae     ProcessNextOutputCountBy6

ProcessRemainingOutputCount:
        add     r10,6
        jz      ExitPackedKernel
        cmp     r10,3
        jb      ProcessRemainingOutputCountLessThan3
        ProcessPointwiseOutputCountPackedN FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax
        sub     r10,3
        jz      ExitPackedKernel

ProcessRemainingOutputCountLessThan3:
        cmp     r10,2
        jb      ProcessRemainingOutputCount1
        ProcessPointwiseOutputCountPackedN FilterCount, 2
        jmp     ExitPackedKernel

ProcessRemainingOutputCount1:
        ProcessPointwiseOutputCountPackedN FilterCount, 1

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
;   r14 - Supplies the address of the input buffer plus 3 * StrideWidth.
;
;   zmm0-zmm23 - Supplies the block accumulators.
;

ComputeBlock MACRO KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

IFIDNI <KernelType>, <Depthwise>
        vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE OutputCount, 1, <vfmadd231ps zmm0,zmm24,ZMMWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vfmadd231ps zmm4,zmm24,ZMMWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vfmadd231ps zmm8,zmm24,ZMMWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vfmadd231ps zmm12,zmm24,ZMMWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vfmadd231ps zmm16,zmm24,ZMMWORD PTR [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vfmadd231ps zmm20,zmm24,ZMMWORD PTR [r14+r9*2+BroadcastOffset]>
ELSE
IF FilterCount EQ 1
        vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE OutputCount, 1, <vfmadd231ps zmm0,zmm24,DWORD BCST [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vfmadd231ps zmm4,zmm24,DWORD BCST [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vfmadd231ps zmm8,zmm24,DWORD BCST [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vfmadd231ps zmm12,zmm24,DWORD BCST [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vfmadd231ps zmm16,zmm24,DWORD BCST [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vfmadd231ps zmm20,zmm24,DWORD BCST [r14+r9*2+BroadcastOffset]>
ELSE
        EmitIfCountGE OutputCount, 1, <vbroadcastss zmm26,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vbroadcastss zmm27,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 3, <vbroadcastss zmm28,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE OutputCount, 4, <vbroadcastss zmm29,DWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE OutputCount, 5, <vbroadcastss zmm30,DWORD PTR [r14+r9+BroadcastOffset]>
        EmitIfCountGE OutputCount, 6, <vbroadcastss zmm31,DWORD PTR [r14+r9*2+BroadcastOffset]>
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vfmadd231ps zmm0,zmm26,ZMMWORD PTR [rdx+VectorOffset]>
        EmitIfCountGE FilterCount, 2, <vfmadd231ps zmm1,zmm26,ZMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 3, <vfmadd231ps zmm2,zmm26,ZMMWORD PTR [rbx+VectorOffset]>
        EmitIfCountGE FilterCount, 4, <vfmadd231ps zmm3,zmm26,ZMMWORD PTR [rbx+rsi+VectorOffset]>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vfmadd231ps zmm0,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vfmadd231ps zmm4,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vfmadd231ps zmm8,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vfmadd231ps zmm12,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vfmadd231ps zmm16,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vfmadd231ps zmm20,zmm31,zmm24>
        EmitIfCountGE FilterCount, 2, <vmovups zmm24,ZMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vfmadd231ps zmm1,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vfmadd231ps zmm5,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vfmadd231ps zmm9,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vfmadd231ps zmm13,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vfmadd231ps zmm17,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vfmadd231ps zmm21,zmm31,zmm24>
        EmitIfCountGE FilterCount, 3, <vmovups zmm24,ZMMWORD PTR [rbx+VectorOffset]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vfmadd231ps zmm2,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vfmadd231ps zmm6,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vfmadd231ps zmm10,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vfmadd231ps zmm14,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vfmadd231ps zmm18,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vfmadd231ps zmm22,zmm31,zmm24>
        EmitIfCountGE FilterCount, 4, <vmovups zmm24,ZMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vfmadd231ps zmm3,zmm26,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vfmadd231ps zmm7,zmm27,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vfmadd231ps zmm11,zmm28,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vfmadd231ps zmm15,zmm29,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vfmadd231ps zmm19,zmm30,zmm24>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vfmadd231ps zmm23,zmm31,zmm24>
ENDIF
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

        LOCAL   ProcessOutputCount
        LOCAL   ProcessNextOutputCountBy6
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCountLessThan3
        LOCAL   ProcessRemainingOutputCount1
        LOCAL   ProcessOutputCountRightPadAndRemaining

;
; Process the output blocks that include left padding.
;

        mov     r10,KernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      ProcessOutputCount
        call    MlasConv&KernelType&FloatSingleAvx512FFilter&FilterCount

;
; Process the output blocks that do not include any padding.
;

ProcessOutputCount:
        mov     r10,KernelFrame.OutputCount[rsp]
        sub     r10,6
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy6:
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             ; advance input by 6 elements
        sub     r10,6
        jae     ProcessNextOutputCountBy6

ProcessRemainingOutputCount:
        add     r10,6                       ; correct for over-subtract above
        jz      ProcessOutputCountRightPadAndRemaining
        cmp     r10,3
        jb      ProcessRemainingOutputCountLessThan3
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jz      ProcessOutputCountRightPadAndRemaining

ProcessRemainingOutputCountLessThan3:
        cmp     r10,1
        je      ProcessOutputCountRightPadAndRemaining
        ProcessOutputCountN Avx512F, KernelFrame, KernelType, 16, FilterCount, 2
        lea     rdi,[rdi+r9*2]              ; advance input by 2 elements
        sub     r10,2

;
; Process the output blocks that include right padding plus any remaining output
; blocks from above.
;

ProcessOutputCountRightPadAndRemaining:
        add     r10,KernelFrame.OutputCountRightPad[rsp]
        jz      ExitKernel
        call    MlasConv&KernelType&FloatSingleAvx512FFilter&FilterCount

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

        LOCAL   ProcessNextOutputCountBy6
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessRemainingOutputCountLessThan3
        LOCAL   ProcessRemainingOutputCount1

        sub     r10,6
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy6:
        ProcessPointwiseOutputCountN Avx512F, 16, FilterCount, 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             ; advance input by 6 elements
        sub     r10,6
        jae     ProcessNextOutputCountBy6

ProcessRemainingOutputCount:
        add     r10,6                       ; correct for over-subtract above
        jz      ExitKernel
        cmp     r10,3
        jb      ProcessRemainingOutputCountLessThan3
        ProcessPointwiseOutputCountN Avx512F, 16, FilterCount, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jz      ExitKernel

ProcessRemainingOutputCountLessThan3:
        cmp     r10,2
        jb      ProcessRemainingOutputCount1
        ProcessPointwiseOutputCountN Avx512F, 16, FilterCount, 2
        jmp     ExitKernel

ProcessRemainingOutputCount1:
        ProcessPointwiseOutputCountN Avx512F, 16, FilterCount, 1

        ENDM

;
; Macro Description:
;
;   This macro computes one reduction step for the dedicated AVX512 pointwise
;   hot tile for FilterCount=4 and OutputCount=6.
;

ComputePointwiseBlockHot4x6Avx512F MACRO VectorOffset, BroadcastOffset

        vbroadcastss zmm26,DWORD PTR [rcx+BroadcastOffset]
        vbroadcastss zmm27,DWORD PTR [rcx+r9+BroadcastOffset]
        vbroadcastss zmm28,DWORD PTR [rcx+r9*2+BroadcastOffset]
        vbroadcastss zmm29,DWORD PTR [r14+BroadcastOffset]
        vbroadcastss zmm30,DWORD PTR [r14+r9+BroadcastOffset]
        vbroadcastss zmm31,DWORD PTR [r14+r9*2+BroadcastOffset]

        vmovups zmm24,ZMMWORD PTR [rdx+VectorOffset]
        vmovups zmm25,ZMMWORD PTR [rdx+rsi+VectorOffset]

        vfmadd231ps zmm0,zmm26,zmm24
        vfmadd231ps zmm1,zmm26,zmm25
        vfmadd231ps zmm4,zmm27,zmm24
        vfmadd231ps zmm5,zmm27,zmm25
        vfmadd231ps zmm8,zmm28,zmm24
        vfmadd231ps zmm9,zmm28,zmm25
        vfmadd231ps zmm12,zmm29,zmm24
        vfmadd231ps zmm13,zmm29,zmm25
        vfmadd231ps zmm16,zmm30,zmm24
        vfmadd231ps zmm17,zmm30,zmm25
        vfmadd231ps zmm20,zmm31,zmm24
        vfmadd231ps zmm21,zmm31,zmm25

        vmovups zmm24,ZMMWORD PTR [rbx+VectorOffset]
        vmovups zmm25,ZMMWORD PTR [rbx+rsi+VectorOffset]

        vfmadd231ps zmm2,zmm26,zmm24
        vfmadd231ps zmm3,zmm26,zmm25
        vfmadd231ps zmm6,zmm27,zmm24
        vfmadd231ps zmm7,zmm27,zmm25
        vfmadd231ps zmm10,zmm28,zmm24
        vfmadd231ps zmm11,zmm28,zmm25
        vfmadd231ps zmm14,zmm29,zmm24
        vfmadd231ps zmm15,zmm29,zmm25
        vfmadd231ps zmm18,zmm30,zmm24
        vfmadd231ps zmm19,zmm30,zmm25
        vfmadd231ps zmm22,zmm31,zmm24
        vfmadd231ps zmm23,zmm31,zmm25

        ENDM

;
; Macro Description:
;
;   This macro computes the steady-state AVX512 pointwise hot tile for a
;   fixed 4 x 6 block:
;       - 4 output-channel blocks
;       - 6 output positions
;
;   This isolates the primary pointwise tile into a dedicated kernel entry so
;   it can be tuned independently from the generic pointwise path.
;

ProcessPointwiseOutputCountHot4x6Avx512F MACRO

        LOCAL   ProcessNextInputBlock

        mov     rcx,rdi
        mov     rdx,r12
        mov     r11,SconvKernelPointwiseFrame.InputChannels[rsp]
        ClearBlock 4, 6

ProcessNextInputBlock:
        lea     r14,[r9+r9*2]
        add     r14,rcx                     ; compute input plus 3 blocks
        lea     rbx,[rdx+rsi*2]             ; compute filter plus 2 rows

        ComputePointwiseBlockHot4x6Avx512F 0*16*4, 0*4
        ComputePointwiseBlockHot4x6Avx512F 1*16*4, 1*4
        ComputePointwiseBlockHot4x6Avx512F 2*16*4, 2*4
        ComputePointwiseBlockHot4x6Avx512F 3*16*4, 3*4
        ComputePointwiseBlockHot4x6Avx512F 4*16*4, 4*4
        ComputePointwiseBlockHot4x6Avx512F 5*16*4, 5*4
        ComputePointwiseBlockHot4x6Avx512F 6*16*4, 6*4
        ComputePointwiseBlockHot4x6Avx512F 7*16*4, 7*4
        ComputePointwiseBlockHot4x6Avx512F 8*16*4, 8*4
        ComputePointwiseBlockHot4x6Avx512F 9*16*4, 9*4
        ComputePointwiseBlockHot4x6Avx512F 10*16*4, 10*4
        ComputePointwiseBlockHot4x6Avx512F 11*16*4, 11*4
        ComputePointwiseBlockHot4x6Avx512F 12*16*4, 12*4
        ComputePointwiseBlockHot4x6Avx512F 13*16*4, 13*4
        ComputePointwiseBlockHot4x6Avx512F 14*16*4, 14*4
        ComputePointwiseBlockHot4x6Avx512F 15*16*4, 15*4

        add     rcx,rbp                     ; advance input to next channel block
        add     rdx,16*16*4                 ; advance filter by 16i16o block
        dec     r11                         ; decrement input blocks remaining
        jnz     ProcessNextInputBlock

        mov     edx,DWORD PTR SconvKernelPointwiseFrame.Flags[rsp]
        mov     rax,SconvKernelPointwiseFrame.OutputStride[rsp]
        mov     rcx,SconvKernelPointwiseFrame.Bias[rsp]
        call    MlasConvPostProcessFloatAvx512FPointwiseFilter4Output6

        ENDM

;
; Generate the convolution kernels.
;
; N.B. BiasFilter is not used here as the AVX-512 EVEX instruction encoding
; efficiently compresses aligned relative byte offsets.
;

SconvKernelFunction Nchw, 16, Avx512F
SconvKernelFunction Nchwc, 16, Avx512F
SconvKernelDepthwiseFunction 16, Avx512F
SconvKernelPointwiseFunction Avx512F

;
; Dedicated AVX512 pointwise kernel for the steady-state 4 x 6 tile.
;

        NESTED_ENTRY MlasConvPointwiseFloatKernelAvx512FHot4x6, _TEXT

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
        mov     r12,rdx
        mov     r10,SconvKernelPointwiseFrame.OutputCount[rsp]
        mov     rsi,SconvKernelPointwiseFrame.FilterStride[rsp]
        mov     rbp,SconvKernelPointwiseFrame.InputStride[rsp]

        sub     r10,6
        jb      ProcessHot4x6Remainder

ProcessHot4x6Loop:
        ProcessPointwiseOutputCountHot4x6Avx512F
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             ; advance input by 6 elements
        sub     r10,6
        jae     ProcessHot4x6Loop

ProcessHot4x6Remainder:
        add     r10,6                       ; correct for over-subtract above
        jz      ExitHot4x6Kernel
        cmp     r10,3
        jb      ProcessHot4x6RemainderLessThan3
        ProcessPointwiseOutputCountN Avx512F, 16, 4, 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     ; advance input by 3 elements
        sub     r10,3
        jz      ExitHot4x6Kernel

ProcessHot4x6RemainderLessThan3:
        cmp     r10,2
        jb      ProcessHot4x6Remainder1
        ProcessPointwiseOutputCountN Avx512F, 16, 4, 2
        jmp     ExitHot4x6Kernel

ProcessHot4x6Remainder1:
        ProcessPointwiseOutputCountN Avx512F, 16, 4, 1

ExitHot4x6Kernel:
        vzeroupper
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

        NESTED_END MlasConvPointwiseFloatKernelAvx512FHot4x6, _TEXT

;
; Experimental AVX512 pointwise kernel for the native packed filter layout.
;

        NESTED_ENTRY MlasConvPointwiseFloatKernelAvx512FNativePacked, _TEXT

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
        mov     r12,rdx
        mov     r10,SconvKernelPointwiseFrame.OutputCount[rsp]
        mov     r11,SconvKernelPointwiseFrame.FilterCount[rsp]
        mov     rsi,SconvKernelPointwiseFrame.FilterStride[rsp]
        mov     rbp,SconvKernelPointwiseFrame.InputStride[rsp]

        cmp     r11,3
        je      ProcessPackedFilterCount3
        jb      ProcessPackedFilterCountLessThan3
        ProcessPointwiseFilterCountPackedN 4
        jmp     ExitPackedKernel

ProcessPackedFilterCount3:
        ProcessPointwiseFilterCountPackedN 3
        jmp     ExitPackedKernel

ProcessPackedFilterCountLessThan3:
        cmp     r11,2
        jb      ProcessPackedFilterCount1
        ProcessPointwiseFilterCountPackedN 2
        jmp     ExitPackedKernel

ProcessPackedFilterCount1:
        ProcessPointwiseFilterCountPackedN 1

ExitPackedKernel:
        vzeroupper
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

        NESTED_END MlasConvPointwiseFloatKernelAvx512FNativePacked, _TEXT

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

GeneratePostProcessFloatAvx512F MACRO UseSilu

        LOCAL   SkipAccumulateOutput
        LOCAL   SkipBiasAddition
        LOCAL   SkipReluActivation
        LOCAL   SkipSiluActivation

        IRP     FilterCount, <1, 2, 3, 4>
        IRP     OutputCount, <1, 2, 3, 6>

IF UseSilu
        LEAF_ENTRY MlasConvPostProcessFloatAvx512FPointwiseFilter&FilterCount&Output&OutputCount, _TEXT
ELSE
        LEAF_ENTRY MlasConvPostProcessFloatAvx512FFilter&FilterCount&Output&OutputCount, _TEXT
ENDIF

IF FilterCount GT 2
        lea     rbx,[r8+rax*2]              ; compute output plus 2 rows
ENDIF

;
; Test if the existing contents of the output buffer should be accumulated
; with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        jz      SkipAccumulateOutput
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vaddps zmm0,zmm0,ZMMWORD PTR [r8]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vaddps zmm4,zmm4,ZMMWORD PTR [r8+16*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vaddps zmm8,zmm8,ZMMWORD PTR [r8+32*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vaddps zmm12,zmm12,ZMMWORD PTR [r8+48*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vaddps zmm16,zmm16,ZMMWORD PTR [r8+64*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vaddps zmm20,zmm20,ZMMWORD PTR [r8+80*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vaddps zmm1,zmm1,ZMMWORD PTR [r8+rax]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vaddps zmm5,zmm5,ZMMWORD PTR [r8+rax+16*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vaddps zmm9,zmm9,ZMMWORD PTR [r8+rax+32*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vaddps zmm13,zmm13,ZMMWORD PTR [r8+rax+48*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vaddps zmm17,zmm17,ZMMWORD PTR [r8+rax+64*4]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vaddps zmm21,zmm21,ZMMWORD PTR [r8+rax+80*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vaddps zmm2,zmm2,ZMMWORD PTR [rbx]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vaddps zmm6,zmm6,ZMMWORD PTR [rbx+16*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vaddps zmm10,zmm10,ZMMWORD PTR [rbx+32*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vaddps zmm14,zmm14,ZMMWORD PTR [rbx+48*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vaddps zmm18,zmm18,ZMMWORD PTR [rbx+64*4]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vaddps zmm22,zmm22,ZMMWORD PTR [rbx+80*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vaddps zmm3,zmm3,ZMMWORD PTR [rbx+rax]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vaddps zmm7,zmm7,ZMMWORD PTR [rbx+rax+16*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vaddps zmm11,zmm11,ZMMWORD PTR [rbx+rax+32*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vaddps zmm15,zmm15,ZMMWORD PTR [rbx+rax+48*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vaddps zmm19,zmm19,ZMMWORD PTR [rbx+rax+64*4]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vaddps zmm23,zmm23,ZMMWORD PTR [rbx+rax+80*4]>

SkipAccumulateOutput:

;
; Test if the bias buffer should be accumulated with the output block.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        jz      SkipBiasAddition
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vaddps zmm0,zmm0,ZMMWORD PTR [rcx]>
        EmitIfCountGE FilterCount, 2, <vaddps zmm1,zmm1,ZMMWORD PTR [rcx+16*4]>
        EmitIfCountGE FilterCount, 3, <vaddps zmm2,zmm2,ZMMWORD PTR [rcx+32*4]>
        EmitIfCountGE FilterCount, 4, <vaddps zmm3,zmm3,ZMMWORD PTR [rcx+48*4]>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups zmm28,ZMMWORD PTR [rcx]>
        EmitIfCountGE FilterCount, 2, <vmovups zmm29,ZMMWORD PTR [rcx+16*4]>
        EmitIfCountGE FilterCount, 3, <vmovups zmm30,ZMMWORD PTR [rcx+32*4]>
        EmitIfCountGE FilterCount, 4, <vmovups zmm31,ZMMWORD PTR [rcx+48*4]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vaddps zmm0,zmm0,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vaddps zmm4,zmm4,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vaddps zmm8,zmm8,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vaddps zmm12,zmm12,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vaddps zmm16,zmm16,zmm28>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vaddps zmm20,zmm20,zmm28>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vaddps zmm1,zmm1,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vaddps zmm5,zmm5,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vaddps zmm9,zmm9,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vaddps zmm13,zmm13,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vaddps zmm17,zmm17,zmm29>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vaddps zmm21,zmm21,zmm29>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vaddps zmm2,zmm2,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vaddps zmm6,zmm6,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vaddps zmm10,zmm10,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vaddps zmm14,zmm14,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vaddps zmm18,zmm18,zmm30>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vaddps zmm22,zmm22,zmm30>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vaddps zmm3,zmm3,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vaddps zmm7,zmm7,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vaddps zmm11,zmm11,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vaddps zmm15,zmm15,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vaddps zmm19,zmm19,zmm31>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vaddps zmm23,zmm23,zmm31>
ENDIF

SkipBiasAddition:

;
; Test for fused ReLU activation.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        jz      SkipReluActivation
        vpxord  zmm24,zmm24,zmm24
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vmaxps zmm0,zmm24,zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vmaxps zmm4,zmm24,zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vmaxps zmm8,zmm24,zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vmaxps zmm12,zmm24,zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vmaxps zmm16,zmm24,zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vmaxps zmm20,zmm24,zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vmaxps zmm1,zmm24,zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vmaxps zmm5,zmm24,zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vmaxps zmm9,zmm24,zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmaxps zmm13,zmm24,zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmaxps zmm17,zmm24,zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmaxps zmm21,zmm24,zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vmaxps zmm2,zmm24,zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vmaxps zmm6,zmm24,zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vmaxps zmm10,zmm24,zmm10>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmaxps zmm14,zmm24,zmm14>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmaxps zmm18,zmm24,zmm18>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmaxps zmm22,zmm24,zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vmaxps zmm3,zmm24,zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vmaxps zmm7,zmm24,zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vmaxps zmm11,zmm24,zmm11>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmaxps zmm15,zmm24,zmm15>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmaxps zmm19,zmm24,zmm19>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmaxps zmm23,zmm24,zmm23>

SkipReluActivation:

IF UseSilu
;
; Test for fused SiLU activation.
;

        test    dl,MLAS_CONV_KERNEL_FLAG_SILU_ACTIVATION
        jz      SkipSiluActivation
        lea     r10,MlasLogisticConstants
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <ApplySiluAvx512F zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <ApplySiluAvx512F zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <ApplySiluAvx512F zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <ApplySiluAvx512F zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <ApplySiluAvx512F zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <ApplySiluAvx512F zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <ApplySiluAvx512F zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <ApplySiluAvx512F zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <ApplySiluAvx512F zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <ApplySiluAvx512F zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <ApplySiluAvx512F zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <ApplySiluAvx512F zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <ApplySiluAvx512F zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <ApplySiluAvx512F zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <ApplySiluAvx512F zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <ApplySiluAvx512F zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <ApplySiluAvx512F zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <ApplySiluAvx512F zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <ApplySiluAvx512F zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <ApplySiluAvx512F zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <ApplySiluAvx512F zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <ApplySiluAvx512F zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <ApplySiluAvx512F zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <ApplySiluAvx512F zmm23>

SkipSiluActivation:
ENDIF

;
; Store the output block in the output buffer.
;

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vmovups ZMMWORD PTR [r8],zmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vmovups ZMMWORD PTR [r8+16*4],zmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vmovups ZMMWORD PTR [r8+32*4],zmm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 4, <vmovups ZMMWORD PTR [r8+48*4],zmm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 5, <vmovups ZMMWORD PTR [r8+64*4],zmm16>
        EmitIfCount2GE FilterCount, 1, OutputCount, 6, <vmovups ZMMWORD PTR [r8+80*4],zmm20>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vmovups ZMMWORD PTR [r8+rax],zmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vmovups ZMMWORD PTR [r8+rax+16*4],zmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vmovups ZMMWORD PTR [r8+rax+32*4],zmm9>
        EmitIfCount2GE FilterCount, 2, OutputCount, 4, <vmovups ZMMWORD PTR [r8+rax+48*4],zmm13>
        EmitIfCount2GE FilterCount, 2, OutputCount, 5, <vmovups ZMMWORD PTR [r8+rax+64*4],zmm17>
        EmitIfCount2GE FilterCount, 2, OutputCount, 6, <vmovups ZMMWORD PTR [r8+rax+80*4],zmm21>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vmovups ZMMWORD PTR [rbx],zmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vmovups ZMMWORD PTR [rbx+16*4],zmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vmovups ZMMWORD PTR [rbx+32*4],zmm10>
        EmitIfCount2GE FilterCount, 3, OutputCount, 4, <vmovups ZMMWORD PTR [rbx+48*4],zmm14>
        EmitIfCount2GE FilterCount, 3, OutputCount, 5, <vmovups ZMMWORD PTR [rbx+64*4],zmm18>
        EmitIfCount2GE FilterCount, 3, OutputCount, 6, <vmovups ZMMWORD PTR [rbx+80*4],zmm22>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vmovups ZMMWORD PTR [rbx+rax],zmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vmovups ZMMWORD PTR [rbx+rax+16*4],zmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vmovups ZMMWORD PTR [rbx+rax+32*4],zmm11>
        EmitIfCount2GE FilterCount, 4, OutputCount, 4, <vmovups ZMMWORD PTR [rbx+rax+48*4],zmm15>
        EmitIfCount2GE FilterCount, 4, OutputCount, 5, <vmovups ZMMWORD PTR [rbx+rax+64*4],zmm19>
        EmitIfCount2GE FilterCount, 4, OutputCount, 6, <vmovups ZMMWORD PTR [rbx+rax+80*4],zmm23>
        add_immed r8,OutputCount*16*4       ; advance output by N nchw16c blocks
        ret

IF UseSilu
        LEAF_END MlasConvPostProcessFloatAvx512FPointwiseFilter&FilterCount&Output&OutputCount, _TEXT
ELSE
        LEAF_END MlasConvPostProcessFloatAvx512FFilter&FilterCount&Output&OutputCount, _TEXT
ENDIF

        ENDM
        ENDM

        ENDM

GeneratePostProcessFloatAvx512F 0
GeneratePostProcessFloatAvx512F 1

        END
