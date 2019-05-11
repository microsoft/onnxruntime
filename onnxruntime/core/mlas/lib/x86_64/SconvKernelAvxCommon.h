/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelAvxCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision convolution operation for the AVX and FMA3 kernels.

--*/

#include "SconvKernelCommon.h"

/*++

Macro Description:

    This macro generates code to clear the block accumulators.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    ymm0-ymm11 - Supplies the block accumulators.

--*/

        .macro ClearBlock FilterCount, OutputCount

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vxorps xmm0,xmm0,xmm0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "vxorps xmm4,xmm4,xmm4"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "vxorps xmm8,xmm8,xmm8"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vxorps xmm1,xmm1,xmm1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "vxorps xmm5,xmm5,xmm5"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "vxorps xmm9,xmm9,xmm9"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vxorps xmm2,xmm2,xmm2"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "vxorps xmm6,xmm6,xmm6"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "vxorps xmm10,xmm10,xmm10"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vxorps xmm3,xmm3,xmm3"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "vxorps xmm7,xmm7,xmm7"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "vxorps xmm11,xmm11,xmm11"

        .endm

/*++

Macro Description:

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

--*/

        .macro PostProcessBlock KernelFrame, KernelType, FilterCount, OutputCount

        mov     edx,DWORD PTR \KernelFrame\()_Flags[rsp]
.if \FilterCount\() > 1
        mov     rax,\KernelFrame\()_OutputStride[rsp]
.endif
.if \FilterCount\() > 2
        lea     rbx,[r8+rax*2]              # compute output plus 2 rows
.endif

//
// Test if the existing contents of the output buffer should be accumulated
// with the output block.
//

        test    dl,1
        jz      .L\KernelType\().\FilterCount\().\OutputCount\().SkipAccumulateOutput
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

.L\KernelType\().\FilterCount\().\OutputCount\().SkipAccumulateOutput:

//
// Test if the bias buffer should be accumulated with the output block.
//

        test    dl,2
        jz      .L\KernelType\().\FilterCount\().\OutputCount\().SkipBiasAddition
        mov     rcx,\KernelFrame\()_Bias[rsp]
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

.L\KernelType\().\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

        test    dl,4
        jz      .L\KernelType\().\FilterCount\().\OutputCount\().SkipReluActivation
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

.L\KernelType\().\FilterCount\().\OutputCount\().SkipReluActivation:

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
        add     r8,\OutputCount\()*8*4      # advance output by N nchw8c blocks

        .endm
