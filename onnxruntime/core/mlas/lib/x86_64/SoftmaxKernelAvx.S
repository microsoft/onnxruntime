/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SoftmaxKernelAvx.s

Abstract:

    This module implements the kernels for the single precision softmax
    operation.

    This implementation uses AVX instructions.

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

        FUNCTION_ENTRY MlasReduceMaximumF32KernelAvx

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

/*++

Routine Description:

    This routine implements a vectorized kernel to produce the final output for
    the softmax operation.

Arguments:

    Output (rdi) - Supplies the output buffer.

    N (rsi) - Supplies the number of elements to process.

    Parameters (rdx) - Supplies an array containing the scale value.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeSoftmaxOutputF32KernelAvx

        vbroadcastss ymm4,DWORD PTR [rdx]       # broadcast scale value
        cmp     rsi,32
        jb      .LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountBy32:
        vmulps  ymm0,ymm4,YMMWORD PTR [rdi]
        vmulps  ymm1,ymm4,YMMWORD PTR [rdi+8*4]
        sub     rsi,32
        vmulps  ymm2,ymm4,YMMWORD PTR [rdi+16*4]
        vmulps  ymm3,ymm4,YMMWORD PTR [rdi+24*4]
        vmovups YMMWORD PTR [rdi],ymm0
        vmovups YMMWORD PTR [rdi+8*4],ymm1
        vmovups YMMWORD PTR [rdi+16*4],ymm2
        vmovups YMMWORD PTR [rdi+24*4],ymm3
        add     rdi,32*4                        # advance output by 32 elements
        cmp     rsi,32
        jae     .LComputeSoftmaxOutput.ProcessRemainingCountBy32

.LComputeSoftmaxOutput.ProcessRemainingCountBy8:
        cmp     rsi,8
        jb      .LComputeSoftmaxOutput.ProcessRemainingCountLessThan8
        vmulps  ymm0,ymm4,YMMWORD PTR [rdi]
        sub     rsi,8
        vmovups YMMWORD PTR [rdi],ymm0
        add     rdi,8*4                         # advance output by 8 elements
        jmp     .LComputeSoftmaxOutput.ProcessRemainingCountBy8

.LComputeSoftmaxOutput.ProcessRemainingCountLessThan8:
        test    rsi,rsi
        jz      .LComputeSoftmaxOutput.ExitKernel

.LComputeSoftmaxOutput.ProcessRemainingCountBy1:
        vmulss  xmm0,xmm4,DWORD PTR [rdi]
        vmovss  DWORD PTR [rdi],xmm0
        add     rdi,4                           # advance output by 1 element
        dec     esi
        jnz     .LComputeSoftmaxOutput.ProcessRemainingCountBy1

.LComputeSoftmaxOutput.ExitKernel:
        vzeroupper
        ret

/*++

Routine Description:

    This routine implements a vectorized kernel to produce the final output for
    the log softmax operation.

Arguments:

    Input (rdi) - Supplies the output buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx) - Supplies the number of elements to process.

    Parameters (rcx) - Supplies an array containing the negative maximum and
        logarithm values.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeLogSoftmaxOutputF32KernelAvx

        vbroadcastss ymm4,DWORD PTR [rcx]       # broadcast negative minimum value
        vbroadcastss ymm5,DWORD PTR [rcx+4]     # broadcast log(SumExp)
        cmp     rdx,32
        jb      .LComputeLogSoftmaxOutput.ProcessRemainingCountBy8

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy32:
        vaddps  ymm0,ymm4,YMMWORD PTR [rdi]
        vaddps  ymm1,ymm4,YMMWORD PTR [rdi+8*4]
        sub     rdx,32
        vaddps  ymm2,ymm4,YMMWORD PTR [rdi+16*4]
        vaddps  ymm3,ymm4,YMMWORD PTR [rdi+24*4]
        add     rdi,32*4                        # advance input by 32 elements
        vsubps  ymm0,ymm0,ymm5                  # do as two steps for numeric stability
        vsubps  ymm1,ymm1,ymm5
        vsubps  ymm2,ymm2,ymm5
        vsubps  ymm3,ymm3,ymm5
        vmovups YMMWORD PTR [rsi],ymm0
        vmovups YMMWORD PTR [rsi+8*4],ymm1
        vmovups YMMWORD PTR [rsi+16*4],ymm2
        vmovups YMMWORD PTR [rsi+24*4],ymm3
        add     rsi,32*4                        # advance output by 32 elements
        cmp     rdx,32
        jae     .LComputeLogSoftmaxOutput.ProcessRemainingCountBy32

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy8:
        cmp     rdx,8
        jb      .LComputeLogSoftmaxOutput.ProcessRemainingCountLessThan8
        vaddps  ymm0,ymm4,YMMWORD PTR [rdi]
        add     rdi,8*4                         # advance input by 8 elements
        vsubps  ymm0,ymm0,ymm5                  # do as two steps for numeric stability
        sub     rdx,8
        vmovups YMMWORD PTR [rsi],ymm0
        add     rsi,8*4                         # advance output by 8 elements
        jmp     .LComputeLogSoftmaxOutput.ProcessRemainingCountBy8

.LComputeLogSoftmaxOutput.ProcessRemainingCountLessThan8:
        test    rdx,rdx
        jz      .LComputeLogSoftmaxOutput.ExitKernel

.LComputeLogSoftmaxOutput.ProcessRemainingCountBy1:
        vaddss  xmm0,xmm4,DWORD PTR [rdi]
        add     rdi,4                           # advance input by 1 element
        vsubss  xmm0,xmm0,xmm5
        vmovss  DWORD PTR [rsi],xmm0
        add     rsi,4                           # advance output by 1 element
        dec     edx
        jnz     .LComputeLogSoftmaxOutput.ProcessRemainingCountBy1

.LComputeLogSoftmaxOutput.ExitKernel:
        vzeroupper
        ret

        .end
