;++
;
;Copyright (c) Microsoft Corporation. All rights reserved.
;
;Licensed under the MIT License.
;
;Module Name:
;
;    SoftmaxKernelAvx512F.asm
;
;Abstract:
;
;    This module implements the kernels for the single precision softmax
;    operation.
;
;    This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMinimumF32Value:NEAR

;++
;
;Routine Description:
;
;    This routine implements a vectorized kernel to find the maximum value of
;    the supplied buffer.
;
;Arguments:
;
;    Input (rcx) - Supplies the input buffer.
;
;    N (rdx) - Supplies the number of elements to process.
;
;Return Value:
;
;    Returns the maximum value of the supplied buffer.
;
;--

        LEAF_ENTRY MlasReduceMaximumF32KernelAvx512F, _TEXT

        vbroadcastss zmm0,DWORD PTR [MlasMinimumF32Value]
        test    rdx,rdx
        jz      ExitKernel
        cmp     rdx,16
        jb      ProcessRemainingCountBy1
        cmp     rdx,64
        jb      ProcessRemainingCountBy16
        vmovaps zmm1,zmm0
        vmovaps zmm2,zmm0
        vmovaps zmm3,zmm0

ProcessRemainingCountBy64:
        vmaxps  zmm0,zmm0,ZMMWORD PTR [rcx]
        vmaxps  zmm1,zmm1,ZMMWORD PTR [rcx+16*4]
        sub     rdx,64
        vmaxps  zmm2,zmm2,ZMMWORD PTR [rcx+32*4]
        vmaxps  zmm3,zmm3,ZMMWORD PTR [rcx+48*4]
        add     rcx,64*4                        ; advance input by 64 elements
        cmp     rdx,64
        jae     ProcessRemainingCountBy64
        vmaxps  zmm0,zmm0,zmm1                  ; reduce to single vector
        vmaxps  zmm2,zmm2,zmm3
        vmaxps  zmm0,zmm0,zmm2

ProcessRemainingCountBy16:
        cmp     rdx,16
        jb      ProcessRemainingCountLessThan16
        vmaxps  zmm0,zmm0,ZMMWORD PTR [rcx]
        sub     rdx,16
        add     rcx,16*4                         ; advance input by 16 elements
        jmp     ProcessRemainingCountBy16

ProcessRemainingCountLessThan16:
        vextractf32x8     ymm1,zmm0,1           ; reduce to single scalar
        vmaxps  ymm0,ymm0,ymm1
        vextractf128 xmm1,ymm0,1
        vmaxps  xmm0,xmm0,xmm1
        vshufps xmm1,xmm0,xmm0,0EEh
        vmaxps  xmm0,xmm0,xmm1
        vshufps xmm1,xmm0,xmm0,055h
        vmaxss  xmm0,xmm0,xmm1
        test    rdx,rdx
        jz      ExitKernel

ProcessRemainingCountBy1:
        vmaxss  xmm0,xmm0,DWORD PTR [rcx]
        add     rcx,4                           ; advance input by 1 element
        dec     edx
        jnz     ProcessRemainingCountBy1

ExitKernel:
        vzeroupper
        ret

        LEAF_END MlasReduceMaximumF32KernelAvx512F, _TEXT

        END
