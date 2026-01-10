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

        vbroadcastss zmm0,DWORD PTR C_UNDERSCORE(MlasMinimumF32Value)[rip]
        test    rsi,rsi
        jz      .LReduceMaximum.ExitKernel
        cmp     rsi,16
        jb      .LReduceMaximum.ProcessRemainingCountBy1
        cmp     rsi,64
        jb      .LReduceMaximum.ProcessRemainingCountBy16
        vmovaps zmm1,zmm0
        vmovaps zmm2,zmm0
        vmovaps zmm3,zmm0

.LReduceMaximum.ProcessRemainingCountBy64:
        vmaxps  zmm0,zmm0,ZMMWORD PTR [rdi]
        vmaxps  zmm1,zmm1,ZMMWORD PTR [rdi+16*4]
        sub     rsi,64
        vmaxps  zmm2,zmm2,ZMMWORD PTR [rdi+32*4]
        vmaxps  zmm3,zmm3,ZMMWORD PTR [rdi+48*4]
        add     rdi,64*4                        # advance input by 64 elements
        cmp     rsi,64
        jae     .LReduceMaximum.ProcessRemainingCountBy64
        vmaxps  zmm0,zmm0,zmm1                  # reduce to single vector
        vmaxps  zmm2,zmm2,zmm3
        vmaxps  zmm0,zmm0,zmm2

.LReduceMaximum.ProcessRemainingCountBy16:
        cmp     rsi,16
        jb      .LReduceMaximum.ProcessRemainingCountLessThan16
        vmaxps  zmm0,zmm0,ZMMWORD PTR [rdi]
        sub     rsi,16
        add     rdi,16*4                         # advance input by 16 elements
        jmp     .LReduceMaximum.ProcessRemainingCountBy16

.LReduceMaximum.ProcessRemainingCountLessThan16:
        vextractf32x8     ymm1,zmm0,1           # reduce to single scalar
        vmaxps  ymm0,ymm0,ymm1
        vextractf128 xmm1,ymm0,1
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
