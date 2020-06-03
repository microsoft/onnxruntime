;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SoftmaxKernelAvx.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision softmax
;   operation.
;
;   This implementation uses AVX instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMinimumF32Value:NEAR

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel to find the maximum value of
;   the supplied buffer.
;
; Arguments:
;
;   Input (rcx) - Supplies the input buffer.
;
;   N (rdx) - Supplies the number of elements to process.
;
; Return Value:
;
;   Returns the maximum value of the supplied buffer.
;
;--

        LEAF_ENTRY MlasReduceMaximumF32KernelAvx, _TEXT

        vbroadcastss ymm0,DWORD PTR [MlasMinimumF32Value]
        test    rdx,rdx
        jz      ExitKernel
        cmp     rdx,8
        jb      ProcessRemainingCountBy1
        cmp     rdx,32
        jb      ProcessRemainingCountBy8
        vmovaps ymm1,ymm0
        vmovaps ymm2,ymm0
        vmovaps ymm3,ymm0

ProcessRemainingCountBy32:
        vmaxps  ymm0,ymm0,YMMWORD PTR [rcx]
        vmaxps  ymm1,ymm1,YMMWORD PTR [rcx+8*4]
        sub     rdx,32
        vmaxps  ymm2,ymm2,YMMWORD PTR [rcx+16*4]
        vmaxps  ymm3,ymm3,YMMWORD PTR [rcx+24*4]
        add     rcx,32*4                        ; advance input by 32 elements
        cmp     rdx,32
        jae     ProcessRemainingCountBy32
        vmaxps  ymm0,ymm0,ymm1                  ; reduce to single vector
        vmaxps  ymm2,ymm2,ymm3
        vmaxps  ymm0,ymm0,ymm2

ProcessRemainingCountBy8:
        cmp     rdx,8
        jb      ProcessRemainingCountLessThan8
        vmaxps  ymm0,ymm0,YMMWORD PTR [rcx]
        sub     rdx,8
        add     rcx,8*4                         ; advance input by 8 elements
        jmp     ProcessRemainingCountBy8

ProcessRemainingCountLessThan8:
        vextractf128 xmm1,ymm0,1                ; reduce to single scalar
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

        LEAF_END MlasReduceMaximumF32KernelAvx, _TEXT

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel to produce the final output for
;   the softmax operation.
;
; Arguments:
;
;   Output (rcx) - Supplies the output buffer.
;
;   N (rdx) - Supplies the number of elements to process.
;
;   Parameters (r8) - Supplies an array containing the scale value.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY MlasComputeSoftmaxOutputF32KernelAvx, _TEXT

        vbroadcastss ymm4,DWORD PTR [r8]        ; broadcast scale value
        cmp     rdx,32
        jb      ProcessRemainingCountBy8

ProcessRemainingCountBy32:
        vmulps  ymm0,ymm4,YMMWORD PTR [rcx]
        vmulps  ymm1,ymm4,YMMWORD PTR [rcx+8*4]
        sub     rdx,32
        vmulps  ymm2,ymm4,YMMWORD PTR [rcx+16*4]
        vmulps  ymm3,ymm4,YMMWORD PTR [rcx+24*4]
        vmovups YMMWORD PTR [rcx],ymm0
        vmovups YMMWORD PTR [rcx+8*4],ymm1
        vmovups YMMWORD PTR [rcx+16*4],ymm2
        vmovups YMMWORD PTR [rcx+24*4],ymm3
        add     rcx,32*4                        ; advance output by 32 elements
        cmp     rdx,32
        jae     ProcessRemainingCountBy32

ProcessRemainingCountBy8:
        cmp     rdx,8
        jb      ProcessRemainingCountLessThan8
        vmulps  ymm0,ymm4,YMMWORD PTR [rcx]
        sub     rdx,8
        vmovups YMMWORD PTR [rcx],ymm0
        add     rcx,8*4                         ; advance output by 8 elements
        jmp     ProcessRemainingCountBy8

ProcessRemainingCountLessThan8:
        test    rdx,rdx
        jz      ExitKernel

ProcessRemainingCountBy1:
        vmulss  xmm0,xmm4,DWORD PTR [rcx]
        vmovss  DWORD PTR [rcx],xmm0
        add     rcx,4                           ; advance output by 1 element
        dec     edx
        jnz     ProcessRemainingCountBy1

ExitKernel:
        vzeroupper
        ret

        LEAF_END MlasComputeSoftmaxOutputF32KernelAvx, _TEXT

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel to produce the final output for
;   the log softmax operation.
;
; Arguments:
;
;   Input (rcx) - Supplies the output buffer.
;
;   Output (rdx) - Supplies the output buffer.
;
;   N (r8) - Supplies the number of elements to process.
;
;   Parameters (r9) - Supplies an array containing the negative maximum and
;       logarithm values.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY MlasComputeLogSoftmaxOutputF32KernelAvx, _TEXT

        vbroadcastss ymm4,DWORD PTR [r9]        ; broadcast negative minimum value
        vbroadcastss ymm5,DWORD PTR [r9+4]      ; broadcast log(SumExp)
        cmp     r8,32
        jb      ProcessRemainingCountBy8

ProcessRemainingCountBy32:
        vaddps  ymm0,ymm4,YMMWORD PTR [rcx]
        vaddps  ymm1,ymm4,YMMWORD PTR [rcx+8*4]
        sub     r8,32
        vaddps  ymm2,ymm4,YMMWORD PTR [rcx+16*4]
        vaddps  ymm3,ymm4,YMMWORD PTR [rcx+24*4]
        add     rcx,32*4                        ; advance input by 32 elements
        vsubps  ymm0,ymm0,ymm5                  ; do as two steps for numeric stability
        vsubps  ymm1,ymm1,ymm5
        vsubps  ymm2,ymm2,ymm5
        vsubps  ymm3,ymm3,ymm5
        vmovups YMMWORD PTR [rdx],ymm0
        vmovups YMMWORD PTR [rdx+8*4],ymm1
        vmovups YMMWORD PTR [rdx+16*4],ymm2
        vmovups YMMWORD PTR [rdx+24*4],ymm3
        add     rdx,32*4                        ; advance output by 32 elements
        cmp     r8,32
        jae     ProcessRemainingCountBy32

ProcessRemainingCountBy8:
        cmp     r8,8
        jb      ProcessRemainingCountLessThan8
        vaddps  ymm0,ymm4,YMMWORD PTR [rcx]
        add     rcx,8*4                         ; advance input by 8 elements
        vsubps  ymm0,ymm0,ymm5                  ; do as two steps for numeric stability
        sub     r8,8
        vmovups YMMWORD PTR [rdx],ymm0
        add     rdx,8*4                         ; advance output by 8 elements
        jmp     ProcessRemainingCountBy8

ProcessRemainingCountLessThan8:
        test    r8,r8
        jz      ExitKernel

ProcessRemainingCountBy1:
        vaddss  xmm0,xmm4,DWORD PTR [rcx]
        add     rcx,4                           ; advance input by 1 element
        vsubss  xmm0,xmm0,xmm5
        vmovss  DWORD PTR [rdx],xmm0
        add     rdx,4                           ; advance output by 1 element
        dec     r8d
        jnz     ProcessRemainingCountBy1

ExitKernel:
        vzeroupper
        ret

        LEAF_END MlasComputeLogSoftmaxOutputF32KernelAvx, _TEXT

        END
