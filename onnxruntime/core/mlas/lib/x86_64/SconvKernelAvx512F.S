/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelAvx512F.s

Abstract:

    This module implements the kernels for the single precision convolution
    operation.

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"
#include "SconvKernelCommon.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro generates code to clear the block accumulators.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    zmm0-zmm23 - Supplies the block accumulators.

--*/

        .macro ClearBlock FilterCount, OutputCount

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vpxord zmm0,zmm0,zmm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vpxord zmm4,zmm4,zmm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vpxord zmm8,zmm8,zmm8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vpxord zmm12,zmm12,zmm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vpxord zmm16,zmm16,zmm16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vpxord zmm20,zmm20,zmm20"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vpxord zmm1,zmm1,zmm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vpxord zmm5,zmm5,zmm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vpxord zmm9,zmm9,zmm9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vpxord zmm13,zmm13,zmm13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vpxord zmm17,zmm17,zmm17"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vpxord zmm21,zmm21,zmm21"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vpxord zmm2,zmm2,zmm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vpxord zmm6,zmm6,zmm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vpxord zmm10,zmm10,zmm10"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "vpxord zmm14,zmm14,zmm14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 5, "vpxord zmm18,zmm18,zmm18"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 6, "vpxord zmm22,zmm22,zmm22"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vpxord zmm3,zmm3,zmm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vpxord zmm7,zmm7,zmm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vpxord zmm11,zmm11,zmm11"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "vpxord zmm15,zmm15,zmm15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 5, "vpxord zmm19,zmm19,zmm19"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 6, "vpxord zmm23,zmm23,zmm23"

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for FilterCount by OutputCount block
    of the output buffer.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

    VectorOffset - Supplies the byte offset from the filter buffer to fetch
        elements.

    BroadcastOffset - Supplies the byte offset from the input buffer to fetch
        elements.

Implicit Arguments:

    rcx - Supplies the address of the input buffer.

    rdx - Supplies the address of the filter buffer.

    rsi - Supplies the FilterStride parameter (see function description).

    rbx - Supplies the address of the filter buffer plus 2 * FilterStride.

    r9 - Supplies the StrideWidth parameter (see function description).

    r14 - Supplies the address of the input buffer plus 3 * StrideWidth.

    zmm0-zmm23 - Supplies the block accumulators.

--*/

        .macro ComputeBlock KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

.ifeqs "\KernelType\()","Depthwise"
        vmovups zmm24,ZMMWORD PTR [rdx+\VectorOffset\()]
        EmitIfCountGE \OutputCount\(), 1, "vfmadd231ps zmm0,zmm24,ZMMWORD PTR [rcx+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 2, "vfmadd231ps zmm4,zmm24,ZMMWORD PTR [rcx+r9+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 3, "vfmadd231ps zmm8,zmm24,ZMMWORD PTR [rcx+r9*2+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 4, "vfmadd231ps zmm12,zmm24,ZMMWORD PTR [r14+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 5, "vfmadd231ps zmm16,zmm24,ZMMWORD PTR [r14+r9+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 6, "vfmadd231ps zmm20,zmm24,ZMMWORD PTR [r14+r9*2+\BroadcastOffset\()]"
.else
.if \FilterCount\() == 1
        vmovups zmm24,ZMMWORD PTR [rdx+\VectorOffset\()]
        EmitIfCountGE \OutputCount\(), 1, "vfmadd231ps zmm0,zmm24,DWORD PTR [rcx+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \OutputCount\(), 2, "vfmadd231ps zmm4,zmm24,DWORD PTR [rcx+r9+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \OutputCount\(), 3, "vfmadd231ps zmm8,zmm24,DWORD PTR [rcx+r9*2+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \OutputCount\(), 4, "vfmadd231ps zmm12,zmm24,DWORD PTR [r14+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \OutputCount\(), 5, "vfmadd231ps zmm16,zmm24,DWORD PTR [r14+r9+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \OutputCount\(), 6, "vfmadd231ps zmm20,zmm24,DWORD PTR [r14+r9*2+\BroadcastOffset\()]{1to16}"
.else
        EmitIfCountGE \OutputCount\(), 1, "vbroadcastss zmm26,DWORD PTR [rcx+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 2, "vbroadcastss zmm27,DWORD PTR [rcx+r9+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 3, "vbroadcastss zmm28,DWORD PTR [rcx+r9*2+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 4, "vbroadcastss zmm29,DWORD PTR [r14+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 5, "vbroadcastss zmm30,DWORD PTR [r14+r9+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 6, "vbroadcastss zmm31,DWORD PTR [r14+r9*2+\BroadcastOffset\()]"
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "vfmadd231ps zmm0,zmm26,ZMMWORD PTR [rdx+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 2, "vfmadd231ps zmm1,zmm26,ZMMWORD PTR [rdx+rsi+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 3, "vfmadd231ps zmm2,zmm26,ZMMWORD PTR [rbx+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 4, "vfmadd231ps zmm3,zmm26,ZMMWORD PTR [rbx+rsi+\VectorOffset\()]"
.else
        EmitIfCountGE \FilterCount\(), 1, "vmovups zmm24,ZMMWORD PTR [rdx+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfmadd231ps zmm0,zmm26,zmm24"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vfmadd231ps zmm4,zmm27,zmm24"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vfmadd231ps zmm8,zmm28,zmm24"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vfmadd231ps zmm12,zmm29,zmm24"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vfmadd231ps zmm16,zmm30,zmm24"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vfmadd231ps zmm20,zmm31,zmm24"
        EmitIfCountGE \FilterCount\(), 2, "vmovups zmm24,ZMMWORD PTR [rdx+rsi+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfmadd231ps zmm1,zmm26,zmm24"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vfmadd231ps zmm5,zmm27,zmm24"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vfmadd231ps zmm9,zmm28,zmm24"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vfmadd231ps zmm13,zmm29,zmm24"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vfmadd231ps zmm17,zmm30,zmm24"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vfmadd231ps zmm21,zmm31,zmm24"
        EmitIfCountGE \FilterCount\(), 3, "vmovups zmm24,ZMMWORD PTR [rbx+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfmadd231ps zmm2,zmm26,zmm24"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vfmadd231ps zmm6,zmm27,zmm24"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vfmadd231ps zmm10,zmm28,zmm24"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "vfmadd231ps zmm14,zmm29,zmm24"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 5, "vfmadd231ps zmm18,zmm30,zmm24"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 6, "vfmadd231ps zmm22,zmm31,zmm24"
        EmitIfCountGE \FilterCount\(), 4, "vmovups zmm24,ZMMWORD PTR [rbx+rsi+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfmadd231ps zmm3,zmm26,zmm24"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vfmadd231ps zmm7,zmm27,zmm24"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vfmadd231ps zmm11,zmm28,zmm24"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "vfmadd231ps zmm15,zmm29,zmm24"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 5, "vfmadd231ps zmm19,zmm30,zmm24"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 6, "vfmadd231ps zmm23,zmm31,zmm24"
.endif
.endif
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    rdi - Supplies the address of the input buffer.

    rsi - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    rbp - Supplies the DilationWidth parameter (see function description).

    r8 - Supplies the address of the output buffer.

    r9 - Supplies the StrideWidth parameter (see function description).

    r15 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessFilterCountN KernelFrame, KernelType, FilterCount

//
// Process the output blocks that include left padding.
//

        mov     r10,\KernelFrame\()_OutputCountLeftPad[rsp]
        test    r10,r10
        jz      .L\KernelType\().\FilterCount\().ProcessOutputCount
        call    MlasConv\KernelType\()FloatSingleAvx512FFilter\FilterCount\()

//
// Process the output blocks that do not include any padding.
//

.L\KernelType\().\FilterCount\().ProcessOutputCount:
        mov     r10,\KernelFrame\()_OutputCount[rsp]
        sub     r10,6
        jb      .L\KernelType\().\FilterCount\().ProcessRemainingOutputCount

.L\KernelType\().\FilterCount\().ProcessNextOutputCountBy6:
        ProcessOutputCountN Avx512F, \KernelFrame\(), \KernelType\(), 16, \FilterCount\(), 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             # advance input by 6 elements
        sub     r10,6
        jae     .L\KernelType\().\FilterCount\().ProcessNextOutputCountBy6

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCount:
        add     r10,6                       # correct for over-subtract above
        jz      .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining
        cmp     r10,3
        jb      .L\KernelType\().\FilterCount\().ProcessRemainingOutputCountLessThan3
        ProcessOutputCountN Avx512F, \KernelFrame\(), \KernelType\(), 16, \FilterCount\(), 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     # advance input by 3 elements
        sub     r10,3
        jz      .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCountLessThan3:
        cmp     r10,1
        je      .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining
        ProcessOutputCountN Avx512F, \KernelFrame\(), \KernelType\(), 16, \FilterCount\(), 2
        lea     rdi,[rdi+r9*2]              # advance input by 2 elements
        sub     r10,2

//
// Process the output blocks that include right padding plus any remaining output
// blocks from above.
//

.L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining:
        add     r10,\KernelFrame\()_OutputCountRightPad[rsp]
        jz      .L\KernelType\().ExitKernel
        call    MlasConv\KernelType\()FloatSingleAvx512FFilter\FilterCount\()

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows for a pointwise convolution.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    rdi - Supplies the address of the input buffer.

    rsi - Supplies the FilterStride parameter (see function description).

    rbp - Supplies the InputStride parameter (see function description).

    r8 - Supplies the address of the output buffer.

    r9 - Supplies the StrideWidth parameter (see function description).

    r10 - Supplies the OutputCount parameter (see function description).

    r12 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseFilterCountN FilterCount

        sub     r10,6
        jb      .LPointwise.\FilterCount\().ProcessRemainingOutputCount

.LPointwise.\FilterCount\().ProcessNextOutputCountBy6:
        ProcessPointwiseOutputCountN Avx512F, 16, \FilterCount\(), 6
        lea     rax,[r9*2+r9]
        lea     rdi,[rdi+rax*2]             # advance input by 6 elements
        sub     r10,6
        jae     .LPointwise.\FilterCount\().ProcessNextOutputCountBy6

.LPointwise.\FilterCount\().ProcessRemainingOutputCount:
        add     r10,6                       # correct for over-subtract above
        jz      .LPointwise.ExitKernel
        cmp     r10,3
        jb      .LPointwise.\FilterCount\().ProcessRemainingOutputCountLessThan3
        ProcessPointwiseOutputCountN Avx512F, 16, \FilterCount\(), 3
        lea     rax,[r9*2+r9]
        add     rdi,rax                     # advance input by 3 elements
        sub     r10,3
        jz      .LPointwise.ExitKernel

.LPointwise.\FilterCount\().ProcessRemainingOutputCountLessThan3:
        cmp     r10,2
        jb      .LPointwise.\FilterCount\().ProcessRemainingOutputCount1
        ProcessPointwiseOutputCountN Avx512F, 16, \FilterCount\(), 2
        jmp     .LPointwise.ExitKernel

.LPointwise.\FilterCount\().ProcessRemainingOutputCount1:
        ProcessPointwiseOutputCountN Avx512F, 16, \FilterCount\(), 1

        .endm

//
// Generate the convolution kernels.
//
// N.B. BiasFilter is not used here as the AVX-512 EVEX instruction encoding
// efficiently compresses aligned relative byte offsets.
//

        SconvKernelFunction Nchw, 16, Avx512F
        SconvKernelFunction Nchwc, 16, Avx512F
        SconvKernelDepthwiseFunction 16, Avx512F
        SconvKernelPointwiseFunction Avx512F

/*++

Macro Description:

    This macro generates code to process an output block after the inner
    convolution kernel has executed and then stores the output block to the
    output buffer.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

--*/

        .macro PostProcessBlock FilterCount, OutputCount

        .globl  MlasConvPostProcessFloatAvx512FFilter\FilterCount\()Output\OutputCount\()
#if !defined(__APPLE__)
        .hidden MlasConvPostProcessFloatAvx512FFilter\FilterCount\()Output\OutputCount\()
#endif
MlasConvPostProcessFloatAvx512FFilter\FilterCount\()Output\OutputCount\():

.if \FilterCount\() > 2
        lea     rbx,[r8+rax*2]              # compute output plus 2 rows
.endif

//
// Test if the existing contents of the output buffer should be accumulated
// with the output block.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vaddps zmm0,zmm0,ZMMWORD PTR [r8]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vaddps zmm4,zmm4,ZMMWORD PTR [r8+16*4]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vaddps zmm8,zmm8,ZMMWORD PTR [r8+32*4]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vaddps zmm12,zmm12,ZMMWORD PTR [r8+48*4]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vaddps zmm16,zmm16,ZMMWORD PTR [r8+64*4]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vaddps zmm20,zmm20,ZMMWORD PTR [r8+80*4]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vaddps zmm1,zmm1,ZMMWORD PTR [r8+rax]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vaddps zmm5,zmm5,ZMMWORD PTR [r8+rax+16*4]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vaddps zmm9,zmm9,ZMMWORD PTR [r8+rax+32*4]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vaddps zmm13,zmm13,ZMMWORD PTR [r8+rax+48*4]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vaddps zmm17,zmm17,ZMMWORD PTR [r8+rax+64*4]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vaddps zmm21,zmm21,ZMMWORD PTR [r8+rax+80*4]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vaddps zmm2,zmm2,ZMMWORD PTR [rbx]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vaddps zmm6,zmm6,ZMMWORD PTR [rbx+16*4]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vaddps zmm10,zmm10,ZMMWORD PTR [rbx+32*4]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "vaddps zmm14,zmm14,ZMMWORD PTR [rbx+48*4]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 5, "vaddps zmm18,zmm18,ZMMWORD PTR [rbx+64*4]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 6, "vaddps zmm22,zmm22,ZMMWORD PTR [rbx+80*4]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vaddps zmm3,zmm3,ZMMWORD PTR [rbx+rax]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vaddps zmm7,zmm7,ZMMWORD PTR [rbx+rax+16*4]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vaddps zmm11,zmm11,ZMMWORD PTR [rbx+rax+32*4]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "vaddps zmm15,zmm15,ZMMWORD PTR [rbx+rax+48*4]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 5, "vaddps zmm19,zmm19,ZMMWORD PTR [rbx+rax+64*4]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 6, "vaddps zmm23,zmm23,ZMMWORD PTR [rbx+rax+80*4]"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput:

//
// Test if the bias buffer should be accumulated with the output block.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "vaddps zmm0,zmm0,ZMMWORD PTR [rcx]"
        EmitIfCountGE \FilterCount\(), 2, "vaddps zmm1,zmm1,ZMMWORD PTR [rcx+16*4]"
        EmitIfCountGE \FilterCount\(), 3, "vaddps zmm2,zmm2,ZMMWORD PTR [rcx+32*4]"
        EmitIfCountGE \FilterCount\(), 4, "vaddps zmm3,zmm3,ZMMWORD PTR [rcx+48*4]"
.else
        EmitIfCountGE \FilterCount\(), 1, "vmovups zmm28,ZMMWORD PTR [rcx]"
        EmitIfCountGE \FilterCount\(), 2, "vmovups zmm29,ZMMWORD PTR [rcx+16*4]"
        EmitIfCountGE \FilterCount\(), 3, "vmovups zmm30,ZMMWORD PTR [rcx+32*4]"
        EmitIfCountGE \FilterCount\(), 4, "vmovups zmm31,ZMMWORD PTR [rcx+48*4]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vaddps zmm0,zmm0,zmm28"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vaddps zmm4,zmm4,zmm28"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vaddps zmm8,zmm8,zmm28"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vaddps zmm12,zmm12,zmm28"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vaddps zmm16,zmm16,zmm28"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vaddps zmm20,zmm20,zmm28"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vaddps zmm1,zmm1,zmm29"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vaddps zmm5,zmm5,zmm29"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vaddps zmm9,zmm9,zmm29"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vaddps zmm13,zmm13,zmm29"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vaddps zmm17,zmm17,zmm29"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vaddps zmm21,zmm21,zmm29"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vaddps zmm2,zmm2,zmm30"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vaddps zmm6,zmm6,zmm30"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vaddps zmm10,zmm10,zmm30"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "vaddps zmm14,zmm14,zmm30"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 5, "vaddps zmm18,zmm18,zmm30"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 6, "vaddps zmm22,zmm22,zmm30"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vaddps zmm3,zmm3,zmm31"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vaddps zmm7,zmm7,zmm31"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vaddps zmm11,zmm11,zmm31"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "vaddps zmm15,zmm15,zmm31"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 5, "vaddps zmm19,zmm19,zmm31"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 6, "vaddps zmm23,zmm23,zmm31"
.endif

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation
        vpxord  zmm24,zmm24,zmm24
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vmaxps zmm0,zmm24,zmm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vmaxps zmm4,zmm24,zmm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vmaxps zmm8,zmm24,zmm8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vmaxps zmm12,zmm24,zmm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vmaxps zmm16,zmm24,zmm16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vmaxps zmm20,zmm24,zmm20"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vmaxps zmm1,zmm24,zmm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vmaxps zmm5,zmm24,zmm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vmaxps zmm9,zmm24,zmm9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vmaxps zmm13,zmm24,zmm13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vmaxps zmm17,zmm24,zmm17"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vmaxps zmm21,zmm24,zmm21"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vmaxps zmm2,zmm24,zmm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vmaxps zmm6,zmm24,zmm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vmaxps zmm10,zmm24,zmm10"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vmaxps zmm14,zmm24,zmm14"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vmaxps zmm18,zmm24,zmm18"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vmaxps zmm22,zmm24,zmm22"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vmaxps zmm3,zmm24,zmm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vmaxps zmm7,zmm24,zmm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vmaxps zmm11,zmm24,zmm11"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vmaxps zmm15,zmm24,zmm15"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vmaxps zmm19,zmm24,zmm19"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vmaxps zmm23,zmm24,zmm23"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation:

//
// Store the output block in the output buffer.
//

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vmovups ZMMWORD PTR [r8],zmm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vmovups ZMMWORD PTR [r8+16*4],zmm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vmovups ZMMWORD PTR [r8+32*4],zmm8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "vmovups ZMMWORD PTR [r8+48*4],zmm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "vmovups ZMMWORD PTR [r8+64*4],zmm16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "vmovups ZMMWORD PTR [r8+80*4],zmm20"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vmovups ZMMWORD PTR [r8+rax],zmm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vmovups ZMMWORD PTR [r8+rax+16*4],zmm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vmovups ZMMWORD PTR [r8+rax+32*4],zmm9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "vmovups ZMMWORD PTR [r8+rax+48*4],zmm13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "vmovups ZMMWORD PTR [r8+rax+64*4],zmm17"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "vmovups ZMMWORD PTR [r8+rax+80*4],zmm21"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vmovups ZMMWORD PTR [rbx],zmm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vmovups ZMMWORD PTR [rbx+16*4],zmm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vmovups ZMMWORD PTR [rbx+32*4],zmm10"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "vmovups ZMMWORD PTR [rbx+48*4],zmm14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 5, "vmovups ZMMWORD PTR [rbx+64*4],zmm18"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 6, "vmovups ZMMWORD PTR [rbx+80*4],zmm22"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vmovups ZMMWORD PTR [rbx+rax],zmm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vmovups ZMMWORD PTR [rbx+rax+16*4],zmm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vmovups ZMMWORD PTR [rbx+rax+32*4],zmm11"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "vmovups ZMMWORD PTR [rbx+rax+48*4],zmm15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 5, "vmovups ZMMWORD PTR [rbx+rax+64*4],zmm19"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 6, "vmovups ZMMWORD PTR [rbx+rax+80*4],zmm23"
        add_immed r8,\OutputCount\()*16*4   # advance output by N nchw16c blocks
        ret

        .endm

        .irp    FilterCount, 1, 2, 3, 4
        .irp    OutputCount, 1, 2, 3, 6
            PostProcessBlock \FilterCount\(), \OutputCount\()
        .endr
        .endr

        .end
