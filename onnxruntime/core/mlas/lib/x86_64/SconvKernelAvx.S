/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelAvx.S

Abstract:

    This module implements the kernels for the single precision convolution
    operation.

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"
#include "SconvKernelAvxCommon.h"

        .intel_syntax noprefix

        .text

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

    ymm0-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlock KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

.ifeqs "\KernelType\()","Depthwise"
        vmovups ymm12,YMMWORD PTR [rdx]
        EmitIfCountGE \OutputCount\(), 1, "vmulps ymm8,ymm12,YMMWORD PTR [rcx]"
        EmitIfCountGE \OutputCount\(), 1, "vaddps ymm0,ymm0,ymm8"
        EmitIfCountGE \OutputCount\(), 2, "vmulps ymm9,ymm12,YMMWORD PTR [rcx+r9]"
        EmitIfCountGE \OutputCount\(), 2, "vaddps ymm4,ymm4,ymm9"
.else
        EmitIfCountGE \OutputCount\(), 1, "vbroadcastss ymm13,DWORD PTR [rcx+\BroadcastOffset\()]"
        EmitIfCountGE \OutputCount\(), 2, "vbroadcastss ymm14,DWORD PTR [rcx+r9+\BroadcastOffset\()]"
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "vmulps ymm8,ymm13,YMMWORD PTR [rdx+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 1, "vaddps ymm0,ymm0,ymm8"
        EmitIfCountGE \FilterCount\(), 2, "vmulps ymm9,ymm13,YMMWORD PTR [rdx+rsi+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 2, "vaddps ymm1,ymm1,ymm9"
        EmitIfCountGE \FilterCount\(), 3, "vmulps ymm10,ymm13,YMMWORD PTR [rbx+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 3, "vaddps ymm2,ymm2,ymm10"
        EmitIfCountGE \FilterCount\(), 4, "vmulps ymm11,ymm13,YMMWORD PTR [rbx+rsi+\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 4, "vaddps ymm3,ymm3,ymm11"
.else
        EmitIfCountGE \FilterCount\(), 1, "vmovups ymm12,YMMWORD PTR [rdx+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vmulps ymm8,ymm13,ymm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vaddps ymm0,ymm0,ymm8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vmulps ymm9,ymm14,ymm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vaddps ymm4,ymm4,ymm9"
        EmitIfCountGE \FilterCount\(), 2, "vmovups ymm12,YMMWORD PTR [rdx+rsi+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vmulps ymm10,ymm13,ymm12"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vaddps ymm1,ymm1,ymm10"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vmulps ymm11,ymm14,ymm12"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vaddps ymm5,ymm5,ymm11"
        EmitIfCountGE \FilterCount\(), 3, "vmovups ymm12,YMMWORD PTR [rbx+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vmulps ymm8,ymm13,ymm12"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vaddps ymm2,ymm2,ymm8"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vmulps ymm9,ymm14,ymm12"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vaddps ymm6,ymm6,ymm9"
        EmitIfCountGE \FilterCount\(), 4, "vmovups ymm12,YMMWORD PTR [rbx+rsi+\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vmulps ymm10,ymm13,ymm12"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vaddps ymm3,ymm3,ymm10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vmulps ymm11,ymm14,ymm12"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vaddps ymm7,ymm7,ymm11"
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
        call    MlasConv\KernelType\()FloatSingleAvxFilter\FilterCount\()

//
// Process the output blocks that do not include any padding.
//

.L\KernelType\().\FilterCount\().ProcessOutputCount:
        mov     r10,\KernelFrame\()_OutputCount[rsp]
        sub     r10,2
        jb      .L\KernelType\().\FilterCount\().ProcessRemainingOutputCount

.L\KernelType\().\FilterCount\().ProcessNextOutputCountBy2:
        ProcessOutputCountN Avx, \KernelFrame\(), \KernelType\(), 8, \FilterCount\(), 2
        lea     rdi,[rdi+r9*2]              # advance input by 2 elements
        sub     r10,2
        jae     .L\KernelType\().\FilterCount\().ProcessNextOutputCountBy2

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCount:
        add     r10,2                       # correct for over-subtract above

//
// Process the output blocks that include right padding plus any remaining output
// blocks from above.
//

.L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining:
        add     r10,\KernelFrame\()_OutputCountRightPad[rsp]
        jz      .L\KernelType\().ExitKernel
        call    MlasConv\KernelType\()FloatSingleAvxFilter\FilterCount\()

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

        sub     r10,2
        jb      .LPointwise.\FilterCount\().ProcessRemainingOutputCount

.LPointwise.\FilterCount\().ProcessNextOutputCountBy2:
        ProcessPointwiseOutputCountN Avx, 8, \FilterCount\(), 2
        lea     rdi,[rdi+r9*2]              # advance input by 2 elements
        sub     r10,2
        jae     .LPointwise.\FilterCount\().ProcessNextOutputCountBy2

.LPointwise.\FilterCount\().ProcessRemainingOutputCount:
        add     r10,2                       # correct for over-subtract above
        jz      .LPointwise.ExitKernel
        ProcessPointwiseOutputCountN Avx, 8, \FilterCount\(), 1

        .endm

//
// Generate the convolution kernels.
//

        SconvKernelFunction Nchw, 8, Avx
        SconvKernelFunction Nchwc, 8, Avx, BiasFilter
        SconvKernelDepthwiseFunction 8, Avx
        SconvKernelPointwiseFunction Avx, BiasFilter

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

        .globl  MlasConvPostProcessFloatAvxFilter\FilterCount\()Output\OutputCount\()
#if !defined(__APPLE__)
        .hidden MlasConvPostProcessFloatAvxFilter\FilterCount\()Output\OutputCount\()
#endif
MlasConvPostProcessFloatAvxFilter\FilterCount\()Output\OutputCount\():

        .globl  MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\()
#if !defined(__APPLE__)
        .hidden MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\()
#endif
MlasConvPostProcessFloatFma3Filter\FilterCount\()Output\OutputCount\():

.if \FilterCount\() > 2
        lea     rbx,[r8+rax*2]              # compute output plus 2 rows
.endif

//
// Test if the existing contents of the output buffer should be accumulated
// with the output block.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vaddps ymm0,ymm0,YMMWORD PTR [r8]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vaddps ymm4,ymm4,YMMWORD PTR [r8+32]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vaddps ymm8,ymm8,YMMWORD PTR [r8+64]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vaddps ymm1,ymm1,YMMWORD PTR [r8+rax]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vaddps ymm5,ymm5,YMMWORD PTR [r8+rax+32]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vaddps ymm9,ymm9,YMMWORD PTR [r8+rax+64]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vaddps ymm2,ymm2,YMMWORD PTR [rbx]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vaddps ymm6,ymm6,YMMWORD PTR [rbx+32]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vaddps ymm10,ymm10,YMMWORD PTR [rbx+64]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vaddps ymm3,ymm3,YMMWORD PTR [rbx+rax]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vaddps ymm7,ymm7,YMMWORD PTR [rbx+rax+32]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vaddps ymm11,ymm11,YMMWORD PTR [rbx+rax+64]"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput:

//
// Test if the bias buffer should be accumulated with the output block.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "vaddps ymm0,ymm0,YMMWORD PTR [rcx]"
        EmitIfCountGE \FilterCount\(), 2, "vaddps ymm1,ymm1,YMMWORD PTR [rcx+32]"
        EmitIfCountGE \FilterCount\(), 3, "vaddps ymm2,ymm2,YMMWORD PTR [rcx+64]"
        EmitIfCountGE \FilterCount\(), 4, "vaddps ymm3,ymm3,YMMWORD PTR [rcx+96]"
.else
        EmitIfCountGE \FilterCount\(), 1, "vmovups ymm12,YMMWORD PTR [rcx]"
        EmitIfCountGE \FilterCount\(), 2, "vmovups ymm13,YMMWORD PTR [rcx+32]"
        EmitIfCountGE \FilterCount\(), 3, "vmovups ymm14,YMMWORD PTR [rcx+64]"
        EmitIfCountGE \FilterCount\(), 4, "vmovups ymm15,YMMWORD PTR [rcx+96]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vaddps ymm0,ymm0,ymm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vaddps ymm4,ymm4,ymm12"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vaddps ymm8,ymm8,ymm12"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vaddps ymm1,ymm1,ymm13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vaddps ymm5,ymm5,ymm13"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vaddps ymm9,ymm9,ymm13"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vaddps ymm2,ymm2,ymm14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vaddps ymm6,ymm6,ymm14"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vaddps ymm10,ymm10,ymm14"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vaddps ymm3,ymm3,ymm15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vaddps ymm7,ymm7,ymm15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vaddps ymm11,ymm11,ymm15"
.endif

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

        test    dl,MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        jz      .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation
        vxorps  xmm15,xmm15,xmm15
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vmaxps ymm0,ymm15,ymm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vmaxps ymm4,ymm15,ymm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vmaxps ymm8,ymm15,ymm8"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vmaxps ymm1,ymm15,ymm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vmaxps ymm5,ymm15,ymm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vmaxps ymm9,ymm15,ymm9"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vmaxps ymm2,ymm15,ymm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vmaxps ymm6,ymm15,ymm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vmaxps ymm10,ymm15,ymm10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vmaxps ymm3,ymm15,ymm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vmaxps ymm7,ymm15,ymm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vmaxps ymm11,ymm15,ymm11"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation:

//
// Store the output block in the output buffer.
//

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vmovups YMMWORD PTR [r8],ymm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vmovups YMMWORD PTR [r8+32],ymm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vmovups YMMWORD PTR [r8+64],ymm8"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vmovups YMMWORD PTR [r8+rax],ymm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vmovups YMMWORD PTR [r8+rax+32],ymm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vmovups YMMWORD PTR [r8+rax+64],ymm9"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vmovups YMMWORD PTR [rbx],ymm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vmovups YMMWORD PTR [rbx+32],ymm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vmovups YMMWORD PTR [rbx+64],ymm10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vmovups YMMWORD PTR [rbx+rax],ymm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vmovups YMMWORD PTR [rbx+rax+32],ymm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vmovups YMMWORD PTR [rbx+rax+64],ymm11"
        add_immed r8,\OutputCount\()*8*4    # advance output by N nchw8c blocks
        ret

        .endm

        .irp    FilterCount, 1, 2, 3, 4
        .irp    OutputCount, 1, 2, 3
            PostProcessBlock \FilterCount\(), \OutputCount\()
        .endr
        .endr

        .end
