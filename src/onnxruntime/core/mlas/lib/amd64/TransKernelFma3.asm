;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   TransKernelFma3.asm
;
; Abstract:
;
;   This module implements kernels for various transcendental functions.
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE TransKernelCommon.inc
        .list

        EXTERN  MlasMaskMoveTableAvx:NEAR
        EXTERN  MlasExpConstants:NEAR

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel for the exponential function.
;
; Arguments:
;
;   Input (rcx) - Supplies the input buffer.
;
;   Output (rdx) - Supplies the output buffer.
;
;   N (r8) - Supplies the number of elements to process.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasComputeExpF32KernelFma3, _TEXT

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

        lea     rax,MlasExpConstants
        vbroadcastss ymm4,ExpConstants.LowerRange[rax]
        vbroadcastss ymm5,ExpConstants.UpperRange[rax]
        vbroadcastss ymm6,ExpConstants.MinimumExponent[rax]
        vbroadcastss ymm7,ExpConstants.MaximumExponent[rax]
        vbroadcastss ymm8,ExpConstants.RoundingBias[rax]
        vbroadcastss ymm9,ExpConstants.Log2Low[rax]
        vbroadcastss ymm10,ExpConstants.poly_0[rax]
        vbroadcastss ymm11,ExpConstants.poly_1[rax]
        vbroadcastss ymm12,ExpConstants.poly_2[rax]
        vbroadcastss ymm13,ExpConstants.poly_3[rax]
        vbroadcastss ymm14,ExpConstants.poly_4[rax]
        vbroadcastss ymm15,ExpConstants.poly_56[rax]

        sub     r8,8
        jb      ProcessRemainingCount

ComputeExpBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rcx]     ; clamp lower bound
        vbroadcastss ymm2,ExpConstants.Log2Reciprocal[rax]
        vminps  ymm0,ymm5,ymm0                  ; clamp upper bound
        vbroadcastss ymm3,ExpConstants.Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm8              ; (x / ln2) plus rounding bias
        vsubps  ymm1,ymm2,ymm8                  ; m = round(x / ln2)
        vfmadd231ps ymm0,ymm1,ymm3              ; range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm9              ; range reduce: x -= (m * ln2_low)
        vmovaps ymm1,ymm10                      ; p = poly_0
        vfmadd213ps ymm1,ymm0,ymm11             ; p = p * x + poly_1
        vpslld  ymm2,ymm2,23                    ; shift m to exponent field
        vfmadd213ps ymm1,ymm0,ymm12             ; p = p * x + poly_2
        vpminsd  ymm3,ymm2,ymm7                 ; clamp upper normal exponent to +127
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_3
        vpmaxsd  ymm3,ymm3,ymm6                 ; clamp lower normal exponent to -126
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_4
        vpsubd  ymm2,ymm2,ymm3                  ; compute overflow exponent
        vpaddd  ymm3,ymm3,ymm7                  ; add exponent bias to normal scale
        vpaddd  ymm2,ymm2,ymm7                  ; add exponent bias to overflow scale
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_56
        vmulps  ymm0,ymm0,ymm2                  ; scale x with overflow exponent
        vfmadd213ps ymm1,ymm0,ymm2              ; p = p * (x * overflow) + overflow
        vmulps  ymm1,ymm1,ymm3                  ; scale p with normal exponent
        add     rcx,8*4                         ; advance input by 8 elements
        vmovups YMMWORD PTR [rdx],ymm1
        add     rdx,8*4                         ; advance output by 8 elements
        sub     r8,8
        jae     ComputeExpBy8Loop

ProcessRemainingCount:
        add     r8,8                            ; correct for over-subtract above
        jz      ExitKernel
        neg     r8
        lea     r10,MlasMaskMoveTableAvx+8*4
        vmovups ymm2,YMMWORD PTR [r10+r8*4]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rcx]
        vmaxps  ymm0,ymm4,ymm0                  ; clamp lower bound
        vbroadcastss ymm4,ExpConstants.Log2Reciprocal[rax]
        vminps  ymm0,ymm5,ymm0                  ; clamp upper bound
        vbroadcastss ymm3,ExpConstants.Log2High[rax]
        vfmadd213ps ymm4,ymm0,ymm8              ; (x / ln2) plus rounding bias
        vsubps  ymm1,ymm4,ymm8                  ; m = round(x / ln2)
        vfmadd231ps ymm0,ymm1,ymm3              ; range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm9              ; range reduce: x -= (m * ln2_low)
        vmovaps ymm1,ymm10                      ; p = poly_0
        vfmadd213ps ymm1,ymm0,ymm11             ; p = p * x + poly_1
        vpslld  ymm4,ymm4,23                    ; shift m to exponent field
        vfmadd213ps ymm1,ymm0,ymm12             ; p = p * x + poly_2
        vpminsd  ymm3,ymm4,ymm7                 ; clamp upper normal exponent to +127
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_3
        vpmaxsd  ymm3,ymm3,ymm6                 ; clamp lower normal exponent to -126
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_4
        vpsubd  ymm4,ymm4,ymm3                  ; compute overflow exponent
        vpaddd  ymm3,ymm3,ymm7                  ; add exponent bias to normal scale
        vpaddd  ymm4,ymm4,ymm7                  ; add exponent bias to overflow scale
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_5
        vmulps  ymm0,ymm0,ymm4                  ; scale x with overflow exponent
        vfmadd213ps ymm1,ymm0,ymm4              ; p = p * (x * overflow) + overflow
        vmulps  ymm1,ymm1,ymm3                  ; scale p with normal exponent
        vmaskmovps YMMWORD PTR [rdx],ymm2,ymm1

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

        NESTED_END MlasComputeExpF32KernelFma3, _TEXT

;++
;
; Routine Description:
;
;   This routine implements a vectorized kernel for the sum of exponential
;   functions.
;
; Arguments:
;
;   Input (rcx) - Supplies the input buffer.
;
;   Output (rdx) - Optionally supplies the output buffer. When used for Softmax,
;       the output buffer is used to store the intermediate exp() results. When
;       used for LogSoftmax, the intermediate exp() results are not required.
;
;   N (r8) - Supplies the number of elements to process.
;
;   NegativeMaximum (r9) - Supplies the address of the negative maximum value
;       that is added to each element before computing the exponential function.
;
; Return Value:
;
;   Returns the sum of the exponential functions.
;
;--

        NESTED_ENTRY MlasComputeSumExpF32KernelFma3, _TEXT

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

        lea     rax,MlasExpConstants
        vbroadcastss ymm9,DWORD PTR [r9]        ; broadcast negative maximum value
        vxorps  xmm10,xmm10,xmm10               ; clear exp() accumulator
        sub     r8,24
        jb      ProcessRemainingCount

ComputeExpBy24Loop:
        vbroadcastss ymm11,ExpConstants.LowerRangeSumExp[rax]
        vbroadcastss ymm2,ExpConstants.Log2Reciprocal[rax]
        vaddps  ymm0,ymm9,YMMWORD PTR [rcx]     ; bias by negative maximum value
        vaddps  ymm3,ymm9,YMMWORD PTR [rcx+32]
        vaddps  ymm6,ymm9,YMMWORD PTR [rcx+64]
        vbroadcastss ymm15,ExpConstants.RoundingBias[rax]
        vmaxps  ymm0,ymm11,ymm0                 ; clamp lower bound
        vmovaps ymm5,ymm2
        vmaxps  ymm3,ymm11,ymm3
        vmovaps ymm8,ymm2
        vmaxps  ymm6,ymm11,ymm6
        vbroadcastss ymm13,ExpConstants.Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm15             ; (x / ln2) plus rounding bias
        vfmadd213ps ymm5,ymm3,ymm15
        vfmadd213ps ymm8,ymm6,ymm15
        vbroadcastss ymm14,ExpConstants.Log2Low[rax]
        vsubps  ymm1,ymm2,ymm15                 ; m = round(x / ln2)
        vsubps  ymm4,ymm5,ymm15
        vsubps  ymm7,ymm8,ymm15
        vfmadd231ps ymm0,ymm1,ymm13             ; range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm3,ymm4,ymm13
        vfmadd231ps ymm6,ymm7,ymm13
        vfmadd231ps ymm0,ymm1,ymm14             ; range reduce: x -= (m * ln2_low)
        vfmadd231ps ymm3,ymm4,ymm14
        vfmadd231ps ymm6,ymm7,ymm14
        vbroadcastss ymm1,ExpConstants.poly_0[rax]
        vbroadcastss ymm13,ExpConstants.poly_1[rax]
        vmovaps ymm4,ymm1
        vmovaps ymm7,ymm1
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_1
        vfmadd213ps ymm4,ymm3,ymm13
        vfmadd213ps ymm7,ymm6,ymm13
        vbroadcastss ymm14,ExpConstants.poly_2[rax]
        vpslld  ymm2,ymm2,23                    ; shift m to exponent field
        vpslld  ymm5,ymm5,23
        vpslld  ymm8,ymm8,23
        vbroadcastss ymm15,ExpConstants.MaximumExponent[rax]
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_2
        vfmadd213ps ymm4,ymm3,ymm14
        vfmadd213ps ymm7,ymm6,ymm14
        vbroadcastss ymm13,ExpConstants.poly_3[rax]
        vpaddd  ymm2,ymm2,ymm15                 ; add exponent bias to scale
        vpaddd  ymm5,ymm5,ymm15
        vpaddd  ymm8,ymm8,ymm15
        vbroadcastss ymm14,ExpConstants.poly_4[rax]
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_3
        vfmadd213ps ymm4,ymm3,ymm13
        vfmadd213ps ymm7,ymm6,ymm13
        vbroadcastss ymm15,ExpConstants.poly_56[rax]
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_4
        vfmadd213ps ymm4,ymm3,ymm14
        vfmadd213ps ymm7,ymm6,ymm14
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_5
        vfmadd213ps ymm4,ymm3,ymm15
        vfmadd213ps ymm7,ymm6,ymm15
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_6
        vfmadd213ps ymm4,ymm3,ymm15
        vfmadd213ps ymm7,ymm6,ymm15
        vmulps  ymm1,ymm1,ymm2                  ; scale p with exponent
        vmulps  ymm4,ymm4,ymm5
        vaddps  ymm10,ymm10,ymm1                ; accumulate exp() results
        vmulps  ymm7,ymm7,ymm8
        vaddps  ymm10,ymm10,ymm4
        add     rcx,24*4                        ; advance input by 24 elements
        vaddps  ymm10,ymm10,ymm7
        test    rdx,rdx
        jz      SkipStoreResultsBy24
        vmovups YMMWORD PTR [rdx],ymm1
        vmovups YMMWORD PTR [rdx+32],ymm4
        vmovups YMMWORD PTR [rdx+64],ymm7
        add     rdx,24*4                        ; advance output by 24 elements

SkipStoreResultsBy24:
        sub     r8,24
        jae     ComputeExpBy24Loop

ProcessRemainingCount:
        add     r8,24                           ; correct for over-subtract above
        jz      ReduceAccumulator
        vbroadcastss ymm11,ExpConstants.LowerRangeSumExp[rax]

ComputeExpBy8Loop:
        cmp     r8,8                            ; remaining count < 8?
        jb      LoadPartialVector
        vmovups ymm0,YMMWORD PTR [rcx]
        jmp     ProcessSingleVector

LoadPartialVector:
        lea     r10,MlasMaskMoveTableAvx+8*4
        neg     r8                              ; carry flag unchanged
        vmovups ymm3,YMMWORD PTR [r10+r8*4]
        vmaskmovps ymm0,ymm3,YMMWORD PTR [rcx]
        vandps  ymm9,ymm9,ymm3                  ; mask unused maximum value to 0.0

ProcessSingleVector:
        vbroadcastss ymm2,ExpConstants.Log2Reciprocal[rax]
        vaddps  ymm0,ymm9,ymm0                  ; bias by negative maximum value
        vbroadcastss ymm15,ExpConstants.RoundingBias[rax]
        vmaxps  ymm0,ymm11,ymm0                 ; clamp lower bound
        vbroadcastss ymm13,ExpConstants.Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm15             ; (input / ln2) plus rounding bias
        vbroadcastss ymm14,ExpConstants.Log2Low[rax]
        vsubps  ymm1,ymm2,ymm15                 ; round(input / ln2)
        vfmadd231ps ymm0,ymm1,ymm13             ; range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm14             ; range reduce: x -= (m * ln2_low)
        vbroadcastss ymm1,ExpConstants.poly_0[rax]
        vbroadcastss ymm13,ExpConstants.poly_1[rax]
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_1
        vbroadcastss ymm14,ExpConstants.poly_2[rax]
        vpslld  ymm2,ymm2,23                    ; shift m to exponent field
        vbroadcastss ymm15,ExpConstants.MaximumExponent[rax]
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_2
        vbroadcastss ymm13,ExpConstants.poly_3[rax]
        vpaddd  ymm2,ymm2,ymm15                 ; add exponent bias to scale
        vbroadcastss ymm14,ExpConstants.poly_4[rax]
        vfmadd213ps ymm1,ymm0,ymm13             ; p = p * x + poly_3
        vbroadcastss ymm15,ExpConstants.poly_56[rax]
        vfmadd213ps ymm1,ymm0,ymm14             ; p = p * x + poly_4
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_5
        vfmadd213ps ymm1,ymm0,ymm15             ; p = p * x + poly_6
        vmulps  ymm1,ymm1,ymm2
        jb      StorePartialVector              ; remaining count < 8?
        vaddps  ymm10,ymm10,ymm1                ; accumulate exp() results
        test    rdx,rdx                         ; store exp() results?
        jz      SkipStoreResultsBy8
        vmovups YMMWORD PTR [rdx],ymm1
        add     rdx,8*4                         ; advance output by 8 elements

SkipStoreResultsBy8:
        add     rcx,8*4                         ; advance input by 8 elements
        sub     r8,8
        jnz     ComputeExpBy8Loop
        jmp     ReduceAccumulator

StorePartialVector:
        vandps  ymm1,ymm1,ymm3                  ; mask unused exp() results to 0.0
        vaddps  ymm10,ymm10,ymm1                ; accumulate exp() results
        test    rdx,rdx                         ; store exp() results?
        jz      ReduceAccumulator
        vmaskmovps YMMWORD PTR [rdx],ymm3,ymm1

ReduceAccumulator:
        vhaddps ymm10,ymm10,ymm10
        vhaddps ymm10,ymm10,ymm10
        vextractf128 xmm0,ymm10,1
        vaddss  xmm0,xmm0,xmm10

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

        NESTED_END MlasComputeSumExpF32KernelFma3, _TEXT

        END
