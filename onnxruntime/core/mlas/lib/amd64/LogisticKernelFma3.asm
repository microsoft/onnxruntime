;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   LogisticKernelFma3.asm
;
; Abstract:
;
;   This module implements a kernel for computing the logistic function for a
;   buffer of elements.
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE TransKernelCommon.inc
        .list

        EXTERN  MlasMaskMoveTableAvx:NEAR
        EXTERN  MlasLogisticConstants:NEAR

;++
;
; Routine Description:
;
;   This routine implements the a vectorized kernel for the logistic function.
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

        NESTED_ENTRY MlasComputeLogisticF32KernelFma3, _TEXT

        alloc_stack (TransKernelFrame.ReturnAddress)

        save_xmm128 xmm6,TransKernelFrame.SavedXmm6
        save_xmm128 xmm7,TransKernelFrame.SavedXmm7
        save_xmm128 xmm8,TransKernelFrame.SavedXmm8
        save_xmm128 xmm9,TransKernelFrame.SavedXmm9
        save_xmm128 xmm10,TransKernelFrame.SavedXmm10
        save_xmm128 xmm11,TransKernelFrame.SavedXmm11
        save_xmm128 xmm12,TransKernelFrame.SavedXmm12
        save_xmm128 xmm13,TransKernelFrame.SavedXmm13
        save_xmm128 xmm14,TransKernelFrame.SavedXmm14
        save_xmm128 xmm15,TransKernelFrame.SavedXmm15

        END_PROLOGUE

        lea     rax,MlasLogisticConstants
        vbroadcastss ymm4,LogisticConstants.LowerRange[rax]
        vbroadcastss ymm5,LogisticConstants.UpperRange[rax]
        vbroadcastss ymm6,LogisticConstants.alpha_9[rax]
        vbroadcastss ymm7,LogisticConstants.alpha_7[rax]
        vbroadcastss ymm8,LogisticConstants.alpha_5[rax]
        vbroadcastss ymm9,LogisticConstants.alpha_3[rax]
        vbroadcastss ymm10,LogisticConstants.alpha_1[rax]
        vbroadcastss ymm11,LogisticConstants.beta_10[rax]
        vbroadcastss ymm12,LogisticConstants.beta_6[rax]
        vbroadcastss ymm13,LogisticConstants.beta_4[rax]
        vbroadcastss ymm14,LogisticConstants.beta_2[rax]
        vbroadcastss ymm15,LogisticConstants.beta_0[rax]

        sub     r8,8
        jb      ProcessRemainingCount

ComputeLogisticBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rcx]     ; clamp lower bound
        vmovaps ymm2,ymm7
        vminps  ymm0,ymm5,ymm0                  ; clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  ; x2
        vbroadcastss ymm3,LogisticConstants.beta_8[rax]
        vfmadd231ps ymm2,ymm1,ymm6              ; p = x2 * alpha_9 + alpha_7
        vfmadd213ps ymm2,ymm1,ymm8              ; p = x2 * p + alpha_5
        vfmadd213ps ymm2,ymm1,ymm9              ; p = x2 * p + alpha_3
        vfmadd213ps ymm2,ymm1,ymm10             ; p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm11             ; q = x2 * beta_10 + beta_8
        vfmadd213ps ymm3,ymm1,ymm12             ; q = x2 * q + beta_6
        vfmadd213ps ymm3,ymm1,ymm13             ; q = x2 * q + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             ; q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             ; q = x2 * q + beta_0
        vmulps  ymm2,ymm0,ymm2                  ; p = x * p
        vbroadcastss ymm0,LogisticConstants.one_half[rax]
        vdivps  ymm2,ymm2,ymm3
        vxorps  ymm3,ymm3,ymm3
        vaddps  ymm0,ymm2,ymm0                  ; logistic = p / q + 0.5
        vmaxps  ymm0,ymm3,ymm0                  ; clamp lower bound
        add     rcx,8*4                         ; advance input by 8 elements
        vmovups YMMWORD PTR [rdx],ymm0
        add     rdx,8*4                         ; advance output by 8 elements
        sub     r8,8
        jae     ComputeLogisticBy8Loop

ProcessRemainingCount:
        add     r8,8                            ; correct for over-subtract above
        jz      ExitKernel
        neg     r8
        lea     r10,MlasMaskMoveTableAvx+8*4
        vmovups ymm2,YMMWORD PTR [r10+r8*4]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rcx]
        vmaxps  ymm0,ymm4,ymm0                  ; clamp lower bound
        vminps  ymm0,ymm5,ymm0                  ; clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  ; x2
        vbroadcastss ymm3,LogisticConstants.beta_8[rax]
        vfmadd231ps ymm7,ymm1,ymm6              ; p = x2 * alpha_9 + alpha_7
        vfmadd213ps ymm7,ymm1,ymm8              ; p = x2 * p + alpha_5
        vfmadd213ps ymm7,ymm1,ymm9              ; p = x2 * p + alpha_3
        vfmadd213ps ymm7,ymm1,ymm10             ; p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm11             ; q = x2 * beta_10 + beta_8
        vfmadd213ps ymm3,ymm1,ymm12             ; q = x2 * q + beta_6
        vfmadd213ps ymm3,ymm1,ymm13             ; q = x2 * q + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             ; q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             ; q = x2 * q + beta_0
        vmulps  ymm7,ymm0,ymm7                  ; p = x * p
        vbroadcastss ymm0,LogisticConstants.one_half[rax]
        vdivps  ymm7,ymm7,ymm3
        vxorps  ymm3,ymm3,ymm3
        vaddps  ymm0,ymm7,ymm0                  ; logistic = p / q + 0.5
        vmaxps  ymm0,ymm3,ymm0                  ; clamp lower bound
        vmaskmovps YMMWORD PTR [rdx],ymm2,ymm0

ExitKernel:
        vzeroupper
        movaps  xmm6,TransKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,TransKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,TransKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,TransKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,TransKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,TransKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,TransKernelFrame.SavedXmm12[rsp]
        movaps  xmm13,TransKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,TransKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,TransKernelFrame.SavedXmm15[rsp]
        add     rsp,(TransKernelFrame.ReturnAddress)

        BEGIN_EPILOGUE

        ret

        NESTED_END MlasComputeLogisticF32KernelFma3, _TEXT

        END
