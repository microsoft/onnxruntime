;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   TanhKernelFma3.asm
;
; Abstract:
;
;   This module implements a kernel for computing the hyperbolic tangent
;   function for a buffer of elements.
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE TransKernelCommon.inc
        .list

        EXTERN  MlasMaskMoveTableAvx:NEAR
        EXTERN  MlasTanhConstants:NEAR

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel for the hyperbolic tangent
;   function.
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

        NESTED_ENTRY MlasComputeTanhF32KernelFma3, _TEXT

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

        lea     rax,MlasTanhConstants
        vbroadcastss ymm4,TanhConstants.LowerRange[rax]
        vbroadcastss ymm5,TanhConstants.UpperRange[rax]
        vbroadcastss ymm6,TanhConstants.alpha_13[rax]
        vbroadcastss ymm7,TanhConstants.alpha_11[rax]
        vbroadcastss ymm8,TanhConstants.alpha_9[rax]
        vbroadcastss ymm9,TanhConstants.alpha_7[rax]
        vbroadcastss ymm10,TanhConstants.alpha_5[rax]
        vbroadcastss ymm11,TanhConstants.alpha_3[rax]
        vbroadcastss ymm12,TanhConstants.alpha_1[rax]
        vbroadcastss ymm13,TanhConstants.beta_6[rax]
        vbroadcastss ymm14,TanhConstants.beta_2[rax]
        vbroadcastss ymm15,TanhConstants.beta_0[rax]

        sub     r8,8
        jb      ProcessRemainingCount

ComputeTanhBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rcx]     ; clamp lower bound
        vmovaps ymm2,ymm7
        vminps  ymm0,ymm5,ymm0                  ; clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  ; x2
        vbroadcastss ymm3,TanhConstants.beta_4[rax]
        vfmadd231ps ymm2,ymm1,ymm6              ; p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm2,ymm1,ymm8              ; p = x2 * p + alpha_9
        vfmadd213ps ymm2,ymm1,ymm9              ; p = x2 * p + alpha_7
        vfmadd213ps ymm2,ymm1,ymm10             ; p = x2 * p + alpha_5
        vfmadd213ps ymm2,ymm1,ymm11             ; p = x2 * p + alpha_3
        vfmadd213ps ymm2,ymm1,ymm12             ; p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             ; q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             ; q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             ; q = x2 * q + beta_0
        vmulps  ymm2,ymm0,ymm2                  ; p = x * p
        vdivps  ymm0,ymm2,ymm3                  ; tanh = p / q
        add     rcx,8*4                         ; advance input by 8 elements
        vmovups YMMWORD PTR [rdx],ymm0
        add     rdx,8*4                         ; advance output by 8 elements
        sub     r8,8
        jae     ComputeTanhBy8Loop

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
        vbroadcastss ymm3,TanhConstants.beta_4[rax]
        vfmadd231ps ymm7,ymm1,ymm6              ; p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm7,ymm1,ymm8              ; p = x2 * p + alpha_9
        vfmadd213ps ymm7,ymm1,ymm9              ; p = x2 * p + alpha_7
        vfmadd213ps ymm7,ymm1,ymm10             ; p = x2 * p + alpha_5
        vfmadd213ps ymm7,ymm1,ymm11             ; p = x2 * p + alpha_3
        vfmadd213ps ymm7,ymm1,ymm12             ; p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             ; q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             ; q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             ; q = x2 * q + beta_0
        vmulps  ymm7,ymm0,ymm7                  ; p = x * p
        vdivps  ymm0,ymm7,ymm3                  ; tanh = p / q
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

        NESTED_END MlasComputeTanhF32KernelFma3, _TEXT

        END
