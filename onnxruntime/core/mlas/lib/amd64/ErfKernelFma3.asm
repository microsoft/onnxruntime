;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   ErfKernelFma3.asm
;
; Abstract:
;
;   This module implements a kernel for computing the error function for a
;   buffer of elements.
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMaskMoveAvx:NEAR
        EXTERN  MlasErfConstants:NEAR

;
; Structure layout for the erf constants block.
;

ErfConstants STRUCT

        ErfUpperAbsRange DWORD ?
        ErfSplitBoundary DWORD ?
        ErfSMALL_P0 DWORD ?
        ErfSMALL_P1 DWORD ?
        ErfSMALL_P2 DWORD ?
        ErfSMALL_P3 DWORD ?
        ErfSMALL_P4 DWORD ?
        ErfSMALL_P5_Minus_One DWORD ?
        ErfReserve0 DWORD ?
        ErfBIG_P0 DWORD ?
        ErfBIG_P1 DWORD ?
        ErfBIG_P2 DWORD ?
        ErfBIG_P3 DWORD ?
        ErfBIG_P4 DWORD ?
        ErfBIG_P5 DWORD ?
        ErfBIG_P6_Minus_One DWORD ?
        ErfNegZero DWORD ?
        ErfOne DWORD ?

        Exp_UpperRange DWORD ?
        Exp_LowerRange DWORD ?
        Exp_Log2Reciprocal DWORD ?
        Exp_log2_hi DWORD ?
        Exp_log2_lo DWORD ?
        Exp_P0 DWORD ?
        Exp_P1 DWORD ?
        Exp_P2 DWORD ?
        Exp_P3 DWORD ?
        Exp_P4 DWORD ?
        Exp_P5 DWORD ?
        Exp_P6 DWORD ?
        Exp_C DWORD ?
        Exp_X7F DWORD ?

ErfConstants ENDS

;
; Stack frame layout for the erf kernel.
;

ErfKernelFrame STRUCT

        ErfBuffer0 OWORD 8 DUP(?)
        ErfBuffer1 OWORD 8 DUP(?)
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding0 QWORD ?
        Padding1 QWORD ?
        CountN QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?

ErfKernelFrame ENDS

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel for the error function.
;
; Arguments:
;
;   Input (rcx) - Supplies the input buffer.
;
;   Output (rdx) - Supplies the output buffer.
;
;   N (r8)  - Supplies the number of elements to process.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasErfKernelFma3, _TEXT

        alloc_stack (ErfKernelFrame.ReturnAddress)

        save_xmm128 xmm6,ErfKernelFrame.SavedXmm6
        save_xmm128 xmm7,ErfKernelFrame.SavedXmm7
        save_xmm128 xmm8,ErfKernelFrame.SavedXmm8
        save_xmm128 xmm9,ErfKernelFrame.SavedXmm9
        save_xmm128 xmm10,ErfKernelFrame.SavedXmm10
        save_xmm128 xmm11,ErfKernelFrame.SavedXmm11
        save_xmm128 xmm12,ErfKernelFrame.SavedXmm12
        save_xmm128 xmm13,ErfKernelFrame.SavedXmm13
        save_xmm128 xmm14,ErfKernelFrame.SavedXmm14
        save_xmm128 xmm15,ErfKernelFrame.SavedXmm15

        END_PROLOGUE

        lea     rax,MlasErfConstants
        sub     r8,8*4
        jb      LErfProcessRemainingCount

LComputeErf4x8Loop:
        vbroadcastss ymm15,ErfConstants.ErfNegZero[rax]
        vmovups ymm0,YMMWORD PTR [rcx]          ; original input vx0
        vmovups ymm1,YMMWORD PTR [rcx+32]       ; original input vx1
        vmovups ymm2,YMMWORD PTR [rcx+64]       ; original input vx2
        vmovups ymm3,YMMWORD PTR [rcx+96]       ; original input vx3

        vandps  ymm4,ymm0,ymm15                 ; vsign0
        vandps  ymm5,ymm1,ymm15                 ; vsign1
        vandps  ymm6,ymm2,ymm15                 ; vsign2
        vandps  ymm7,ymm3,ymm15                 ; vsign3
        vandnps ymm0,ymm15,ymm0                 ; abs(vx0)  va0
        vandnps ymm1,ymm15,ymm1                 ; abs(vx1)  va1
        vandnps ymm2,ymm15,ymm2                 ; abs(vx2)  va2
        vandnps ymm3,ymm15,ymm3                 ; abs(vx3)  va3

        vbroadcastss ymm14,ErfConstants.ErfUpperAbsRange[rax]
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp],ymm4
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp+32],ymm5
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp+64],ymm6
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp+96],ymm7

        vbroadcastss ymm8,ErfConstants.ErfSMALL_P0[rax]
        vminps  ymm0,ymm0,ymm14                 ; force abs value in range
        vminps  ymm1,ymm1,ymm14
        vminps  ymm2,ymm2,ymm14
        vminps  ymm3,ymm3,ymm14
        vmovaps ymm9,ymm8
        vmovaps ymm10,ymm8
        vmovaps ymm11,ymm8

        vbroadcastss ymm15,ErfConstants.ErfSMALL_P1[rax]
        vmulps  ymm4,ymm0,ymm0                  ; vs0 (square)
        vmulps  ymm5,ymm1,ymm1                  ; vs1
        vmulps  ymm6,ymm2,ymm2                  ; vs2
        vmulps  ymm7,ymm3,ymm3                  ; vs3

        vbroadcastss ymm14,ErfConstants.ErfSMALL_P2[rax]
        vfmadd213ps ymm8,ymm4,ymm15
        vfmadd213ps ymm9,ymm5,ymm15
        vfmadd213ps ymm10,ymm6,ymm15
        vfmadd213ps ymm11,ymm7,ymm15

        vbroadcastss ymm13,ErfConstants.ErfSMALL_P3[rax]
        vfmadd213ps ymm8,ymm4,ymm14
        vfmadd213ps ymm9,ymm5,ymm14
        vfmadd213ps ymm10,ymm6,ymm14
        vfmadd213ps ymm11,ymm7,ymm14

        vbroadcastss ymm15,ErfConstants.ErfSMALL_P4[rax]
        vfmadd213ps ymm8,ymm4,ymm13
        vfmadd213ps ymm9,ymm5,ymm13
        vfmadd213ps ymm10,ymm6,ymm13
        vfmadd213ps ymm11,ymm7,ymm13

        vbroadcastss ymm14,ErfConstants.ErfSMALL_P5_Minus_One[rax]
        vfmadd213ps ymm8,ymm4,ymm15
        vfmadd213ps ymm9,ymm5,ymm15
        vfmadd213ps ymm10,ymm6,ymm15
        vfmadd213ps ymm11,ymm7,ymm15

        vfmadd213ps ymm8,ymm4,ymm14
        vfmadd213ps ymm9,ymm5,ymm14
        vfmadd213ps ymm10,ymm6,ymm14
        vfmadd213ps ymm11,ymm7,ymm14

        vbroadcastss ymm12,ErfConstants.ErfSplitBoundary[rax]
        vfmadd213ps ymm8,ymm0,ymm0
        vfmadd213ps ymm9,ymm1,ymm1
        vfmadd213ps ymm10,ymm2,ymm2
        vfmadd213ps ymm11,ymm3,ymm3

        vcmpgtps ymm4,ymm0,ymm12                ; vmask0
        vcmpgtps ymm5,ymm1,ymm12                ; vmask1
        vcmpgtps ymm6,ymm2,ymm12                ; vmask2
        vcmpgtps ymm7,ymm3,ymm12                ; vmask3

        vandnps ymm8,ymm4,ymm8
        vandnps ymm9,ymm5,ymm9
        vandnps ymm10,ymm6,ymm10
        vandnps ymm11,ymm7,ymm11

        vbroadcastss ymm15,ErfConstants.ErfBIG_P1[rax]
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp],ymm8
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+32],ymm9
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+64],ymm10
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+96],ymm11

LBiggerNumbers:
        vbroadcastss ymm8,ErfConstants.ErfBIG_P0[rax]
        vandps  ymm0,ymm4,ymm0
        vandps  ymm1,ymm5,ymm1
        vandps  ymm2,ymm6,ymm2
        vandps  ymm3,ymm7,ymm3
        vmovaps ymm9,ymm8
        vmovaps ymm10,ymm8
        vmovaps ymm11,ymm8

        vbroadcastss ymm14,ErfConstants.ErfBIG_P2[rax]
        vfmadd213ps ymm8,ymm0,ymm15
        vfmadd213ps ymm9,ymm1,ymm15
        vfmadd213ps ymm10,ymm2,ymm15
        vfmadd213ps ymm11,ymm3,ymm15

        vbroadcastss ymm13,ErfConstants.ErfBIG_P3[rax]
        vfmadd213ps ymm8,ymm0,ymm14
        vfmadd213ps ymm9,ymm1,ymm14
        vfmadd213ps ymm10,ymm2,ymm14
        vfmadd213ps ymm11,ymm3,ymm14

        vbroadcastss ymm15,ErfConstants.ErfBIG_P4[rax]
        vfmadd213ps ymm8,ymm0,ymm13
        vfmadd213ps ymm9,ymm1,ymm13
        vfmadd213ps ymm10,ymm2,ymm13
        vfmadd213ps ymm11,ymm3,ymm13

        vbroadcastss ymm14,ErfConstants.ErfBIG_P5[rax]
        vfmadd213ps ymm8,ymm0,ymm15
        vfmadd213ps ymm9,ymm1,ymm15
        vfmadd213ps ymm10,ymm2,ymm15
        vfmadd213ps ymm11,ymm3,ymm15

        vbroadcastss ymm13,ErfConstants.ErfBIG_P6_Minus_One[rax]
        vfmadd213ps ymm8,ymm0,ymm14
        vfmadd213ps ymm9,ymm1,ymm14
        vfmadd213ps ymm10,ymm2,ymm14
        vfmadd213ps ymm11,ymm3,ymm14

        vbroadcastss ymm15,ErfConstants.ErfNegZero[rax]
        vfmadd213ps ymm8,ymm0,ymm13
        vfmadd213ps ymm9,ymm1,ymm13
        vfmadd213ps ymm10,ymm2,ymm13
        vfmadd213ps ymm11,ymm3,ymm13

        vbroadcastss ymm14,ErfConstants.Exp_LowerRange[rax]
        vfmadd213ps ymm8,ymm0,ymm0
        vfmadd213ps ymm9,ymm1,ymm1
        vfmadd213ps ymm10,ymm2,ymm2
        vfmadd213ps ymm11,ymm3,ymm3

        vbroadcastss ymm4,ErfConstants.Exp_Log2Reciprocal[rax]
        vxorps  ymm8,ymm8,ymm15
        vxorps  ymm9,ymm9,ymm15
        vxorps  ymm10,ymm10,ymm15
        vxorps  ymm11,ymm11,ymm15

        vbroadcastss ymm13,ErfConstants.Exp_C[rax]
        vmovaps ymm5,ymm4
        vmovaps ymm6,ymm4
        vmovaps ymm7,ymm4

        ; expf(ymm8 -- ymm11)
        vmaxps  ymm8,ymm8,ymm14
        vmaxps  ymm9,ymm9,ymm14
        vmaxps  ymm10,ymm10,ymm14
        vmaxps  ymm11,ymm11,ymm14

        vbroadcastss ymm0,ErfConstants.Exp_log2_hi[rax]
        vfmadd213ps ymm4,ymm8,ymm13
        vfmadd213ps ymm5,ymm9,ymm13
        vfmadd213ps ymm6,ymm10,ymm13
        vfmadd213ps ymm7,ymm11,ymm13

        vbroadcastss ymm15,ErfConstants.Exp_log2_lo[rax]
        vmovaps ymm1,ymm0
        vmovaps ymm2,ymm0
        vmovaps ymm3,ymm0

        vsubps  ymm4,ymm4,ymm13                 ; vr = round()
        vsubps  ymm5,ymm5,ymm13
        vsubps  ymm6,ymm6,ymm13
        vsubps  ymm7,ymm7,ymm13

        vfmadd213ps ymm0,ymm4,ymm8              ; vf = vr * log2_hi + ve
        vfmadd213ps ymm1,ymm5,ymm9
        vfmadd213ps ymm2,ymm6,ymm10
        vfmadd213ps ymm3,ymm7,ymm11

        vbroadcastss ymm8,ErfConstants.Exp_P0[rax]
        vfmadd231ps ymm0,ymm4,ymm15             ; vf += vr * log_2_lo
        vfmadd231ps ymm1,ymm5,ymm15
        vfmadd231ps ymm2,ymm6,ymm15
        vfmadd231ps ymm3,ymm7,ymm15
        vmovaps ymm9,ymm8
        vmovaps ymm10,ymm8
        vmovaps ymm11,ymm8

        vbroadcastss ymm14,ErfConstants.Exp_P1[rax]
        vbroadcastss ymm13,ErfConstants.Exp_P2[rax]
        vfmadd213ps ymm8,ymm0,ymm14             ; *+ exp_p1
        vfmadd213ps ymm9,ymm1,ymm14
        vfmadd213ps ymm10,ymm2,ymm14
        vfmadd213ps ymm11,ymm3,ymm14

        vbroadcastss ymm12,ErfConstants.Exp_P3[rax]
        vfmadd213ps ymm8,ymm0,ymm13             ; *+ exp_p2
        vfmadd213ps ymm9,ymm1,ymm13
        vfmadd213ps ymm10,ymm2,ymm13
        vfmadd213ps ymm11,ymm3,ymm13

        vbroadcastss ymm15,ErfConstants.Exp_P4[rax]
        vfmadd213ps ymm8,ymm0,ymm12             ; *+ exp_p3
        vfmadd213ps ymm9,ymm1,ymm12
        vfmadd213ps ymm10,ymm2,ymm12
        vfmadd213ps ymm11,ymm3,ymm12

        vbroadcastss ymm14,ErfConstants.Exp_P5[rax]
        vfmadd213ps ymm8,ymm0,ymm15             ; *+ exp_p4
        vfmadd213ps ymm9,ymm1,ymm15
        vfmadd213ps ymm10,ymm2,ymm15
        vfmadd213ps ymm11,ymm3,ymm15

        vbroadcastss ymm13,ErfConstants.Exp_P6[rax]
        vfmadd213ps ymm8,ymm0,ymm14             ; *+ exp_p5
        vfmadd213ps ymm9,ymm1,ymm14
        vfmadd213ps ymm10,ymm2,ymm14
        vfmadd213ps ymm11,ymm3,ymm14

        vbroadcastss ymm12,ErfConstants.Exp_X7F[rax]
        vfmadd213ps ymm8,ymm0,ymm13             ; *+ exp_p6
        vfmadd213ps ymm9,ymm1,ymm13
        vfmadd213ps ymm10,ymm2,ymm13
        vfmadd213ps ymm11,ymm3,ymm13

        vcvttps2dq  ymm4,ymm4
        vcvttps2dq  ymm5,ymm5
        vcvttps2dq  ymm6,ymm6
        vcvttps2dq  ymm7,ymm7

        vbroadcastss ymm15,ErfConstants.ErfOne[rax]
        vpaddd  ymm4,ymm4,ymm12                 ; +127
        vpaddd  ymm5,ymm5,ymm12
        vpaddd  ymm6,ymm6,ymm12
        vpaddd  ymm7,ymm7,ymm12

        vpslld  ymm4,ymm4,23
        vpslld  ymm5,ymm5,23
        vpslld  ymm6,ymm6,23
        vpslld  ymm7,ymm7,23

        vmulps  ymm8,ymm8,ymm4                  ; 2^i * exp(vf)
        vmulps  ymm9,ymm9,ymm5
        vmulps  ymm10,ymm10,ymm6
        vmulps  ymm11,ymm11,ymm7

        vsubps  ymm8,ymm15,ymm8
        vsubps  ymm9,ymm15,ymm9
        vsubps  ymm10,ymm15,ymm10
        vsubps  ymm11,ymm15,ymm11

        ; merge small numbers' result
        vorps   ymm8,ymm8,YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp]
        vorps   ymm9,ymm9,YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+32]
        vorps   ymm10,ymm10,YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+64]
        vorps   ymm11,ymm11,YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp+96]

        ; copy sign
        vorps   ymm0,ymm8,YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp]
        vorps   ymm1,ymm9,YMMWORD PTR 32+ErfKernelFrame.ErfBuffer0[rsp]
        vorps   ymm2,ymm10,YMMWORD PTR 64+ErfKernelFrame.ErfBuffer0[rsp]
        vorps   ymm3,ymm11,YMMWORD PTR 96+ErfKernelFrame.ErfBuffer0[rsp]

        vmovups YMMWORD PTR [rdx],ymm0
        vmovups YMMWORD PTR [rdx+32],ymm1
        vmovups YMMWORD PTR [rdx+64],ymm2
        vmovups YMMWORD PTR [rdx+96],ymm3

        add     rcx,32*4                        ; advance by 4*8 elements
        add     rdx,32*4
        sub     r8,32
        jae     LComputeErf4x8Loop

LErfProcessRemainingCount:
        add     r8,32                           ; correct for over-subtract above
        jz      LErfBatchExp

LErfProcess1x8:
        mov     DWORD PTR ErfKernelFrame.CountN[rsp],r8d
        vbroadcastss ymm3,DWORD PTR ErfKernelFrame.CountN[rsp]

        vpcmpgtd ymm3,ymm3,YMMWORD PTR [MlasMaskMoveAvx]
        vbroadcastss ymm15,ErfConstants.ErfNegZero[rax]
        vmaskmovps ymm0,ymm3,YMMWORD PTR [rcx]  ; original input vx0

        vandps  ymm4,ymm0,ymm15                 ; vsign0
        vandnps ymm0,ymm15,ymm0                 ; abs(vx0)  va0

        vbroadcastss ymm14,ErfConstants.ErfUpperAbsRange[rax]
        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp],ymm4

        vbroadcastss ymm8,ErfConstants.ErfSMALL_P0[rax]
        vminps  ymm0,ymm0,ymm14                 ; force abs value in range

        vbroadcastss ymm15,ErfConstants.ErfSMALL_P1[rax]
        vmulps  ymm4,ymm0,ymm0                  ; vs0 (square)

        vbroadcastss ymm14,ErfConstants.ErfSMALL_P2[rax]
        vfmadd213ps ymm8,ymm4,ymm15

        vbroadcastss ymm13,ErfConstants.ErfSMALL_P3[rax]
        vfmadd213ps ymm8,ymm4,ymm14

        vbroadcastss ymm15,ErfConstants.ErfSMALL_P4[rax]
        vfmadd213ps ymm8,ymm4,ymm13

        vbroadcastss ymm14,ErfConstants.ErfSMALL_P5_Minus_One[rax]
        vfmadd213ps ymm8,ymm4,ymm15

        vfmadd213ps ymm8,ymm4,ymm14

        vbroadcastss ymm12,ErfConstants.ErfSplitBoundary[rax]
        vfmadd213ps ymm8,ymm0,ymm0

        vcmpgtps ymm4,ymm0,ymm12                ; vmask0

        vandnps ymm8,ymm4,ymm8

        vmovups YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp],ymm8

LBiggerNumbersRemaining:
        vbroadcastss ymm15,ErfConstants.ErfBIG_P1[rax]
        vbroadcastss ymm8,ErfConstants.ErfBIG_P0[rax]
        vandps  ymm0,ymm4,ymm0

        vbroadcastss ymm14,ErfConstants.ErfBIG_P2[rax]
        vfmadd213ps ymm8,ymm0,ymm15

        vbroadcastss ymm13,ErfConstants.ErfBIG_P3[rax]
        vfmadd213ps ymm8,ymm0,ymm14

        vbroadcastss ymm15,ErfConstants.ErfBIG_P4[rax]
        vfmadd213ps ymm8,ymm0,ymm13

        vbroadcastss ymm14,ErfConstants.ErfBIG_P5[rax]
        vfmadd213ps ymm8,ymm0,ymm15

        vbroadcastss ymm13,ErfConstants.ErfBIG_P6_Minus_One[rax]
        vfmadd213ps ymm8,ymm0,ymm14

        vbroadcastss ymm15,ErfConstants.ErfNegZero[rax]
        vfmadd213ps ymm8,ymm0,ymm13

        vbroadcastss ymm14,ErfConstants.Exp_LowerRange[rax]
        vfmadd213ps ymm8,ymm0,ymm0

        vbroadcastss ymm4,ErfConstants.Exp_Log2Reciprocal[rax]
        vxorps  ymm8,ymm8,ymm15

        vbroadcastss ymm13,ErfConstants.Exp_C[rax]

        ; expf(ymm8 -- ymm11)
        vmaxps  ymm8,ymm8,ymm14

        vbroadcastss ymm0,ErfConstants.Exp_log2_hi[rax]
        vfmadd213ps ymm4,ymm8,ymm13

        vbroadcastss ymm15,ErfConstants.Exp_log2_lo[rax]

        vsubps  ymm4,ymm4,ymm13                 ; vr = round()

        vfmadd213ps ymm0,ymm4,ymm8              ; vf = vr * log2_hi + ve

        vbroadcastss ymm8,ErfConstants.Exp_P0[rax]

        vfmadd231ps ymm0,ymm4,ymm15             ; vf += vr * log_2_lo

        vbroadcastss ymm14,ErfConstants.Exp_P1[rax]

        vbroadcastss ymm13,ErfConstants.Exp_P2[rax]
        vfmadd213ps ymm8,ymm0,ymm14             ; *+ exp_p1

        vbroadcastss ymm12,ErfConstants.Exp_P3[rax]
        vfmadd213ps ymm8,ymm0,ymm13             ; *+ exp_p2

        vbroadcastss ymm15,ErfConstants.Exp_P4[rax]
        vfmadd213ps ymm8,ymm0,ymm12             ; *+ exp_p3

        vbroadcastss ymm14,ErfConstants.Exp_P5[rax]
        vfmadd213ps ymm8,ymm0,ymm15             ; *+ exp_p4

        vbroadcastss ymm13,ErfConstants.Exp_P6[rax]
        vfmadd213ps ymm8,ymm0,ymm14             ; *+ exp_p5

        vbroadcastss ymm12,ErfConstants.Exp_X7F[rax]
        vfmadd213ps ymm8,ymm0,ymm13             ; *+ exp_p6

        vcvttps2dq ymm4,ymm4

        vbroadcastss ymm15,ErfConstants.ErfOne[rax]
        vpaddd  ymm4,ymm4,ymm12                 ; +127

        vpslld  ymm4,ymm4,23

        vmulps  ymm8,ymm8,ymm4                  ; 2^i * exp(vf)

        vsubps  ymm8,ymm15,ymm8

        ; merge small numbers' result
        vorps   ymm8,ymm8,YMMWORD PTR ErfKernelFrame.ErfBuffer1[rsp]

        ; copy sign
        vorps   ymm0,ymm8,YMMWORD PTR ErfKernelFrame.ErfBuffer0[rsp]

        vmaskmovps YMMWORD PTR [rdx],ymm3,ymm0

        add     rcx,8*4
        add     rdx,8*4
        sub     r8,8
        jg      LErfProcess1x8

LErfBatchExp:
        vzeroupper
        movaps  xmm6,ErfKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,ErfKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,ErfKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,ErfKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,ErfKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,ErfKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,ErfKernelFrame.SavedXmm12[rsp]
        movaps  xmm13,ErfKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,ErfKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,ErfKernelFrame.SavedXmm15[rsp]
        add     rsp,(ErfKernelFrame.ReturnAddress)

        BEGIN_EPILOGUE

        ret

        NESTED_END MlasErfKernelFma3, _TEXT

        END
