/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SoftmaxKernelAvx512F.s

Abstract:

    This module implements the kernels for the single precision softmax
    operation.

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

/*++

Routine Description:

    This routine implements a vectorized kernel to find the maximum value of
    the supplied buffer.

Arguments:

    Input (rdi) - Supplies the input buffer.

    N (rsi) - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/

        FUNCTION_ENTRY MlasReduceMaximumF32KernelAvx512F

        vbroadcastss ymm0,DWORD PTR C_UNDERSCORE(MlasMinimumF32Value)[rip]
        test    rsi,rsi
        jz      .LReduceMaximum.ExitKernel
        cmp     rsi,8
        jb      .LReduceMaximum.ProcessRemainingCountBy1
        cmp     rsi,32
        jb      .LReduceMaximum.ProcessRemainingCountBy8
        vmovaps ymm1,ymm0
        vmovaps ymm2,ymm0
        vmovaps ymm3,ymm0

.LReduceMaximum.ProcessRemainingCountBy32:
        vmaxps  ymm0,ymm0,YMMWORD PTR [rdi]
        vmaxps  ymm1,ymm1,YMMWORD PTR [rdi+8*4]
        sub     rsi,32
        vmaxps  ymm2,ymm2,YMMWORD PTR [rdi+16*4]
        vmaxps  ymm3,ymm3,YMMWORD PTR [rdi+24*4]
        add     rdi,32*4                        # advance input by 32 elements
        cmp     rsi,32
        jae     .LReduceMaximum.ProcessRemainingCountBy32
        vmaxps  ymm0,ymm0,ymm1                  # reduce to single vector
        vmaxps  ymm2,ymm2,ymm3
        vmaxps  ymm0,ymm0,ymm2

.LReduceMaximum.ProcessRemainingCountBy8:
        cmp     rsi,8
        jb      .LReduceMaximum.ProcessRemainingCountLessThan8
        vmaxps  ymm0,ymm0,YMMWORD PTR [rdi]
        sub     rsi,8
        add     rdi,8*4                         # advance input by 8 elements
        jmp     .LReduceMaximum.ProcessRemainingCountBy8

.LReduceMaximum.ProcessRemainingCountLessThan8:
        vextractf128 xmm1,ymm0,1                # reduce to single scalar
        vmaxps  xmm0,xmm0,xmm1
        vshufps xmm1,xmm0,xmm0,0xEE
        vmaxps  xmm0,xmm0,xmm1
        vshufps xmm1,xmm0,xmm0,0x55
        vmaxss  xmm0,xmm0,xmm1
        test    rsi,rsi
        jz      .LReduceMaximum.ExitKernel

.LReduceMaximum.ProcessRemainingCountBy1:
        vmaxss  xmm0,xmm0,DWORD PTR [rdi]
        add     rdi,4                           # advance input by 1 element
        dec     esi
        jnz     .LReduceMaximum.ProcessRemainingCountBy1

.LReduceMaximum.ExitKernel:
        vzeroupper
        ret

        .end
