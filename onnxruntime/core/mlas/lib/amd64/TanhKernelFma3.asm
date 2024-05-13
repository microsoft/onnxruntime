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
        vbroadcastss ymm5,  DWORD PTR [rax + 0]   ; nc2
        vbroadcastss ymm6,  DWORD PTR [rax + 4]   ; nc1
        vbroadcastss ymm4,  DWORD PTR [rax + 8]   ; nc0
        vbroadcastss ymm7,  DWORD PTR [rax + 12]  ; dc2
        vbroadcastss ymm8,  DWORD PTR [rax + 16]  ; dc1
        vbroadcastss ymm9,  DWORD PTR [rax + 20]  ; dc0
        vbroadcastss ymm10, DWORD PTR [rax + 24]  ; absmask
        vbroadcastss ymm11, DWORD PTR [rax + 28]  ; bound

        sub     r8,8
        jb      ProcessRemainingCount

ComputeTanhBy8Loop:
        vandps      ymm0,ymm10,YMMWORD PTR [rcx]
        vmovaps     ymm3, ymm5
        vmovaps     ymm13, ymm7
        vxorps      ymm1, ymm0, YMMWORD PTR [rcx]
        vmulps      ymm2, ymm0, ymm0
        vcmpps      ymm12, ymm0, ymm11, 29
        vfmadd132ps ymm3, ymm6, ymm2
        vfmadd132ps ymm13, ymm8, ymm2
        vfmadd132ps ymm3, ymm4, ymm2
        vfmadd132ps ymm2, ymm9, ymm13
        vfmadd132ps ymm0, ymm0, ymm2
        vdivps      ymm0, ymm0, ymm3
        vblendvps   ymm0, ymm0, ymm4, ymm12
        vxorps      ymm0, ymm0, ymm1
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
        vmovups     ymm15,YMMWORD PTR [r10+r8*4]
        vmaskmovps  ymm0,ymm15,YMMWORD PTR [rcx]
        vandps      ymm0,ymm10,ymm0
        vmovaps     ymm3, ymm5
        vmovaps     ymm13, ymm7
        vxorps      ymm1, ymm0, YMMWORD PTR [rcx]
        vmulps      ymm2, ymm0, ymm0
        vcmpps      ymm12, ymm0, ymm11, 29
        vfmadd132ps ymm3, ymm6, ymm2
        vfmadd132ps ymm13, ymm8, ymm2
        vfmadd132ps ymm3, ymm4, ymm2
        vfmadd132ps ymm2, ymm9, ymm13
        vfmadd132ps ymm0, ymm0, ymm2
        vdivps      ymm0, ymm0, ymm3
        vblendvps   ymm0, ymm0, ymm4, ymm12
        vxorps      ymm0, ymm0, ymm1
        vmaskmovps  YMMWORD PTR [rdx],ymm15,ymm0

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
