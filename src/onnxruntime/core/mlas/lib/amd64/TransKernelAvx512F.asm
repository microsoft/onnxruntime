;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   TransKernelAvx512F.asm
;
; Abstract:
;
;   This module implements kernels for various transcendental functions.
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE TransKernelCommon.inc
        .list

        EXTERN  MlasExpConstants:NEAR
        EXTERN  MlasOpmask16BitTableAvx512:NEAR

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

        LEAF_ENTRY MlasComputeExpF32KernelAvx512F, _TEXT

        lea     rax,MlasExpConstants
        vbroadcastss zmm21,ExpConstants.LowerRange[rax]
        vbroadcastss zmm22,ExpConstants.RoundingBias[rax]
        vbroadcastss zmm23,ExpConstants.Log2Reciprocal[rax]
        vbroadcastss zmm24,ExpConstants.Log2High[rax]
        vbroadcastss zmm25,ExpConstants.Log2Low[rax]
        vbroadcastss zmm26,ExpConstants.poly_0[rax]
        vbroadcastss zmm27,ExpConstants.poly_1[rax]
        vbroadcastss zmm28,ExpConstants.poly_2[rax]
        vbroadcastss zmm29,ExpConstants.poly_3[rax]
        vbroadcastss zmm30,ExpConstants.poly_4[rax]
        vbroadcastss zmm31,ExpConstants.poly_56[rax]

        sub     r8,16
        jb      ProcessRemainingCount

ComputeExpBy16Loop:
        vmaxps  zmm16,zmm21,ZMMWORD PTR [rcx]   ; clamp lower bound
        vmovaps zmm18,zmm23
        vfmadd213ps zmm18,zmm16,zmm22           ; (input / ln2) plus rounding bias
        vmovaps zmm17,zmm26                     ; p = poly_0
        vsubps  zmm18,zmm18,zmm22               ; m = round(input / ln2)
        vfmadd231ps zmm16,zmm18,zmm24           ; range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm16,zmm18,zmm25           ; range reduce: x -= (m * ln2_low)
        vmovaps zmm17,zmm26                     ; p = poly_0
        vfmadd213ps zmm17,zmm16,zmm27           ; p = p * x + poly_1
        vfmadd213ps zmm17,zmm16,zmm28           ; p = p * x + poly_2
        vfmadd213ps zmm17,zmm16,zmm29           ; p = p * x + poly_3
        vfmadd213ps zmm17,zmm16,zmm30           ; p = p * x + poly_4
        vfmadd213ps zmm17,zmm16,zmm31           ; p = p * x + poly_5
        vfmadd213ps zmm17,zmm16,zmm31           ; p = p * x + poly_6
        vscalefps zmm17,zmm17,zmm18             ; scale p with exponent
        add     rcx,16*4                        ; advance input by 16 elements
        vmovups ZMMWORD PTR [rdx],zmm17
        add     rdx,16*4                        ; advance output by 16 elements
        sub     r8,16
        jae     ComputeExpBy16Loop

ProcessRemainingCount:
        add     r8,16                           ; correct for over-subtract above
        jz      ExitKernel
        lea     r10,MlasOpmask16BitTableAvx512
        kmovw   k1,WORD PTR [r10+r8*2]
        vmaxps  zmm16{k1}{z},zmm21,ZMMWORD PTR [rcx]
                                                ; clamp lower bound
        vfmadd213ps zmm23,zmm16,zmm22           ; (input / ln2) plus rounding bias
        vsubps  zmm23,zmm23,zmm22               ; round(input / ln2)
        vfmadd231ps zmm16,zmm23,zmm24           ; range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm16,zmm23,zmm25           ; range reduce: x -= (m * ln2_low)
        vfmadd213ps zmm26,zmm16,zmm27           ; p = p * x + poly_1
        vfmadd213ps zmm26,zmm16,zmm28           ; p = p * x + poly_2
        vfmadd213ps zmm26,zmm16,zmm29           ; p = p * x + poly_3
        vfmadd213ps zmm26,zmm16,zmm30           ; p = p * x + poly_4
        vfmadd213ps zmm26,zmm16,zmm31           ; p = p * x + poly_5
        vfmadd213ps zmm26,zmm16,zmm31           ; p = p * x + poly_6
        vscalefps zmm26,zmm26,zmm23             ; scale p with exponent
        vmovups ZMMWORD PTR [rdx]{k1},zmm26

ExitKernel:
        ret

        LEAF_END MlasComputeExpF32KernelAvx512F, _TEXT

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

        LEAF_ENTRY MlasComputeSumExpF32KernelAvx512F, _TEXT

        lea     rax,MlasExpConstants
        vbroadcastss zmm21,ExpConstants.LowerRange[rax]
        vbroadcastss zmm22,ExpConstants.RoundingBias[rax]
        vbroadcastss zmm23,ExpConstants.Log2Reciprocal[rax]
        vbroadcastss zmm24,ExpConstants.Log2High[rax]
        vbroadcastss zmm25,ExpConstants.Log2Low[rax]
        vbroadcastss zmm26,ExpConstants.poly_0[rax]
        vbroadcastss zmm27,ExpConstants.poly_1[rax]
        vbroadcastss zmm28,ExpConstants.poly_2[rax]
        vbroadcastss zmm29,ExpConstants.poly_3[rax]
        vbroadcastss zmm30,ExpConstants.poly_4[rax]
        vbroadcastss zmm31,ExpConstants.poly_56[rax]

        vbroadcastss zmm19,DWORD PTR [r9]       ; broadcast negative maximum value
        vpxord  zmm20,zmm20,zmm20               ; clear exp() accumulator
        sub     r8,48
        jb      ProcessRemainingCount

ComputeExpBy48Loop:
        vaddps  zmm0,zmm19,ZMMWORD PTR [rcx]    ; bias by negative maximum value
        vaddps  zmm3,zmm19,ZMMWORD PTR [rcx+64]
        vaddps  zmm16,zmm19,ZMMWORD PTR [rcx+128]
        vmaxps  zmm0,zmm21,zmm0                 ; clamp lower bound
        vmovaps zmm2,zmm23
        vmaxps  zmm3,zmm21,zmm3
        vmovaps zmm5,zmm23
        vmaxps  zmm16,zmm21,zmm16
        vmovaps zmm18,zmm23
        vfmadd213ps zmm2,zmm0,zmm22             ; (input / ln2) plus rounding bias
        vfmadd213ps zmm5,zmm3,zmm22
        vfmadd213ps zmm18,zmm16,zmm22
        vmovaps zmm1,zmm26                      ; p = poly_0
        vmovaps zmm4,zmm26
        vmovaps zmm17,zmm26
        vsubps  zmm2,zmm2,zmm22                 ; m = round(input / ln2)
        vsubps  zmm5,zmm5,zmm22
        vsubps  zmm18,zmm18,zmm22
        vfmadd231ps zmm0,zmm2,zmm24             ; range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm3,zmm5,zmm24
        vfmadd231ps zmm16,zmm18,zmm24
        vfmadd231ps zmm0,zmm2,zmm25             ; range reduce: x -= (m * ln2_low)
        vfmadd231ps zmm3,zmm5,zmm25
        vfmadd231ps zmm16,zmm18,zmm25
        vfmadd213ps zmm1,zmm0,zmm27             ; p = p * x + poly_1
        vfmadd213ps zmm4,zmm3,zmm27
        vfmadd213ps zmm17,zmm16,zmm27
        vfmadd213ps zmm1,zmm0,zmm28             ; p = p * x + poly_2
        vfmadd213ps zmm4,zmm3,zmm28
        vfmadd213ps zmm17,zmm16,zmm28
        vfmadd213ps zmm1,zmm0,zmm29             ; p = p * x + poly_3
        vfmadd213ps zmm4,zmm3,zmm29
        vfmadd213ps zmm17,zmm16,zmm29
        vfmadd213ps zmm1,zmm0,zmm30             ; p = p * x + poly_4
        vfmadd213ps zmm4,zmm3,zmm30
        vfmadd213ps zmm17,zmm16,zmm30
        vfmadd213ps zmm1,zmm0,zmm31             ; p = p * x + poly_5
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31
        vfmadd213ps zmm1,zmm0,zmm31             ; p = p * x + poly_6
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31
        vscalefps zmm1,zmm1,zmm2
        vscalefps zmm4,zmm4,zmm5
        vscalefps zmm17,zmm17,zmm18
        vaddps  zmm20,zmm20,zmm1                ; accumulate exp() results
        vaddps  zmm20,zmm20,zmm4
        vaddps  zmm20,zmm20,zmm17
        add     rcx,48*4                        ; advance input by 48 elements
        test    rdx,rdx
        jz      SkipStoreResultsBy48
        vmovups ZMMWORD PTR [rdx],zmm1
        vmovups ZMMWORD PTR [rdx+64],zmm4
        vmovups ZMMWORD PTR [rdx+128],zmm17
        add     rdx,48*4                        ; advance output by 48 elements

SkipStoreResultsBy48:
        sub     r8,48
        jae     ComputeExpBy48Loop

ProcessRemainingCount:
        add     r8,48                           ; correct for over-subtract above
        jz      ReduceAccumulator
        mov     eax,-1
        kmovw   k1,eax                          ; update mask to access all elements

ComputeExpBy16Loop:
        cmp     r8,16
        jae     ProcessSingleVector
        lea     r10,MlasOpmask16BitTableAvx512
        kmovw   k1,WORD PTR [r10+r8*2]

ProcessSingleVector:
        vaddps  zmm0{k1}{z},zmm19,ZMMWORD PTR [rcx]
                                                ; bias by negative maximum value
        vmaxps  zmm0,zmm21,zmm0                 ; clamp lower bound
        vmovaps zmm2,zmm23
        vfmadd213ps zmm2,zmm0,zmm22             ; (input / ln2) plus rounding bias
        vmovaps zmm1,zmm26                      ; p = poly_0
        vsubps  zmm2,zmm2,zmm22                 ; m = round(input / ln2)
        vfmadd231ps zmm0,zmm2,zmm24             ; range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm0,zmm2,zmm25             ; range reduce: x -= (m * ln2_low)
        vfmadd213ps zmm1,zmm0,zmm27             ; p = p * x + poly_1
        vfmadd213ps zmm1,zmm0,zmm28             ; p = p * x + poly_2
        vfmadd213ps zmm1,zmm0,zmm29             ; p = p * x + poly_3
        vfmadd213ps zmm1,zmm0,zmm30             ; p = p * x + poly_4
        vfmadd213ps zmm1,zmm0,zmm31             ; p = p * x + poly_5
        vfmadd213ps zmm1,zmm0,zmm31             ; p = p * x + poly_6
        vscalefps zmm1,zmm1,zmm2
        vaddps  zmm20{k1},zmm20,zmm1            ; accumulate exp() results
        add     rcx,16*4                        ; advance input by 16 elements
        test    rdx,rdx
        jz      SkipStoreResultsBy16
        vmovups ZMMWORD PTR [rdx]{k1},zmm1
        add     rdx,16*4                        ; advance output by 16 elements

SkipStoreResultsBy16:
        sub     r8,16
        ja      ComputeExpBy16Loop

ReduceAccumulator:
        vextractf64x4 ymm0,zmm20,1
        vaddps  zmm0,zmm0,zmm20
        vhaddps ymm0,ymm0,ymm0
        vhaddps ymm0,ymm0,ymm0
        vextractf128 xmm1,ymm0,1
        vaddss  xmm0,xmm0,xmm1

        vzeroupper
        ret

        LEAF_END MlasComputeSumExpF32KernelAvx512F, _TEXT

        END
