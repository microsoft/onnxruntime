/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    TransKernelAvx512F.s

Abstract:

    This module implements kernels for various transcendental functions.

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"
#include "TransKernelCommon.h"

        .intel_syntax noprefix

        .text

/*++

Routine Description:

    This routine implements a vectorized kernel for the exponential function.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx) - Supplies the number of elements to process.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeExpF32KernelAvx512F

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss zmm21,.LExpConstants_LowerRange[rax]
        vbroadcastss zmm22,.LExpConstants_RoundingBias[rax]
        vbroadcastss zmm23,.LExpConstants_Log2Reciprocal[rax]
        vbroadcastss zmm24,.LExpConstants_Log2High[rax]
        vbroadcastss zmm25,.LExpConstants_Log2Low[rax]
        vbroadcastss zmm26,.LExpConstants_poly_0[rax]
        vbroadcastss zmm27,.LExpConstants_poly_1[rax]
        vbroadcastss zmm28,.LExpConstants_poly_2[rax]
        vbroadcastss zmm29,.LExpConstants_poly_3[rax]
        vbroadcastss zmm30,.LExpConstants_poly_4[rax]
        vbroadcastss zmm31,.LExpConstants_poly_56[rax]

        sub     rdx,16
        jb      .LComputeExp.ProcessRemainingCount

.LComputeExp.ComputeExpBy16Loop:
        vmaxps  zmm16,zmm21,ZMMWORD PTR [rdi]   # clamp lower bound
        vmovaps zmm18,zmm23
        vfmadd213ps zmm18,zmm16,zmm22           # (input / ln2) plus rounding bias
        vmovaps zmm17,zmm26                     # p = poly_0
        vsubps  zmm18,zmm18,zmm22               # m = round(input / ln2)
        vfmadd231ps zmm16,zmm18,zmm24           # range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm16,zmm18,zmm25           # range reduce: x -= (m * ln2_low)
        vmovaps zmm17,zmm26                     # p = poly_0
        vfmadd213ps zmm17,zmm16,zmm27           # p = p * x + poly_1
        vfmadd213ps zmm17,zmm16,zmm28           # p = p * x + poly_2
        vfmadd213ps zmm17,zmm16,zmm29           # p = p * x + poly_3
        vfmadd213ps zmm17,zmm16,zmm30           # p = p * x + poly_4
        vfmadd213ps zmm17,zmm16,zmm31           # p = p * x + poly_5
        vfmadd213ps zmm17,zmm16,zmm31           # p = p * x + poly_6
        vscalefps zmm17,zmm17,zmm18             # scale p with exponent
        add     rdi,16*4                        # advance input by 16 elements
        vmovups ZMMWORD PTR [rsi],zmm17
        add     rsi,16*4                        # advance output by 16 elements
        sub     rdx,16
        jae     .LComputeExp.ComputeExpBy16Loop

.LComputeExp.ProcessRemainingCount:
        add     rdx,16                          # correct for over-subtract above
        jz      .LComputeExp.ExitKernel
        lea     r10,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip]
        kmovw   k1,WORD PTR [r10+rdx*2]
        vmaxps  zmm16{k1}{z},zmm21,ZMMWORD PTR [rdi]
                                                # clamp lower bound
        vfmadd213ps zmm23,zmm16,zmm22           # (input / ln2) plus rounding bias
        vsubps  zmm23,zmm23,zmm22               # round(input / ln2)
        vfmadd231ps zmm16,zmm23,zmm24           # range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm16,zmm23,zmm25           # range reduce: x -= (m * ln2_low)
        vfmadd213ps zmm26,zmm16,zmm27           # p = p * x + poly_1
        vfmadd213ps zmm26,zmm16,zmm28           # p = p * x + poly_2
        vfmadd213ps zmm26,zmm16,zmm29           # p = p * x + poly_3
        vfmadd213ps zmm26,zmm16,zmm30           # p = p * x + poly_4
        vfmadd213ps zmm26,zmm16,zmm31           # p = p * x + poly_5
        vfmadd213ps zmm26,zmm16,zmm31           # p = p * x + poly_6
        vscalefps zmm26,zmm26,zmm23             # scale p with exponent
        vmovups ZMMWORD PTR [rsi]{k1},zmm26

.LComputeExp.ExitKernel:
        ret

/*++

Routine Description:

    This routine implements a vectorized kernel for the sum of exponential
    functions.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Optionally supplies the output buffer. When used for Softmax,
        the output buffer is used to store the intermediate exp() results. When
        used for LogSoftmax, the intermediate exp() results are not required.

    N (rdx) - Supplies the number of elements to process.

    NegativeMaximum (rcx) - Supplies the address of the negative maximum that
        is added to each element before computing the exponential function.

Return Value:

    Returns the sum of the exponential functions.

--*/

        FUNCTION_ENTRY MlasComputeSumExpF32KernelAvx512F

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss zmm21,.LExpConstants_LowerRange[rax]
        vbroadcastss zmm22,.LExpConstants_RoundingBias[rax]
        vbroadcastss zmm23,.LExpConstants_Log2Reciprocal[rax]
        vbroadcastss zmm24,.LExpConstants_Log2High[rax]
        vbroadcastss zmm25,.LExpConstants_Log2Low[rax]
        vbroadcastss zmm26,.LExpConstants_poly_0[rax]
        vbroadcastss zmm27,.LExpConstants_poly_1[rax]
        vbroadcastss zmm28,.LExpConstants_poly_2[rax]
        vbroadcastss zmm29,.LExpConstants_poly_3[rax]
        vbroadcastss zmm30,.LExpConstants_poly_4[rax]
        vbroadcastss zmm31,.LExpConstants_poly_56[rax]

        vbroadcastss zmm19,DWORD PTR [rcx]      # broadcast negative maximum value
        vpxord  zmm20,zmm20,zmm20               # clear exp() accumulator
        sub     rdx,48
        jb      .LComputeSumExp.ProcessRemainingCount

.LComputeSumExp.ComputeExpBy48Loop:
        vaddps  zmm0,zmm19,ZMMWORD PTR [rdi]    # bias by negative maximum value
        vaddps  zmm3,zmm19,ZMMWORD PTR [rdi+64]
        vaddps  zmm16,zmm19,ZMMWORD PTR [rdi+128]
        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound
        vmovaps zmm2,zmm23
        vmaxps  zmm3,zmm21,zmm3
        vmovaps zmm5,zmm23
        vmaxps  zmm16,zmm21,zmm16
        vmovaps zmm18,zmm23
        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias
        vfmadd213ps zmm5,zmm3,zmm22
        vfmadd213ps zmm18,zmm16,zmm22
        vmovaps zmm1,zmm26                      # p = poly_0
        vmovaps zmm4,zmm26
        vmovaps zmm17,zmm26
        vsubps  zmm2,zmm2,zmm22                 # m = round(input / ln2)
        vsubps  zmm5,zmm5,zmm22
        vsubps  zmm18,zmm18,zmm22
        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm3,zmm5,zmm24
        vfmadd231ps zmm16,zmm18,zmm24
        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: x -= (m * ln2_low)
        vfmadd231ps zmm3,zmm5,zmm25
        vfmadd231ps zmm16,zmm18,zmm25
        vfmadd213ps zmm1,zmm0,zmm27             # p = p * x + poly_1
        vfmadd213ps zmm4,zmm3,zmm27
        vfmadd213ps zmm17,zmm16,zmm27
        vfmadd213ps zmm1,zmm0,zmm28             # p = p * x + poly_2
        vfmadd213ps zmm4,zmm3,zmm28
        vfmadd213ps zmm17,zmm16,zmm28
        vfmadd213ps zmm1,zmm0,zmm29             # p = p * x + poly_3
        vfmadd213ps zmm4,zmm3,zmm29
        vfmadd213ps zmm17,zmm16,zmm29
        vfmadd213ps zmm1,zmm0,zmm30             # p = p * x + poly_4
        vfmadd213ps zmm4,zmm3,zmm30
        vfmadd213ps zmm17,zmm16,zmm30
        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_5
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31
        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_6
        vfmadd213ps zmm4,zmm3,zmm31
        vfmadd213ps zmm17,zmm16,zmm31
        vscalefps zmm1,zmm1,zmm2
        vscalefps zmm4,zmm4,zmm5
        vscalefps zmm17,zmm17,zmm18
        vaddps  zmm20,zmm20,zmm1                # accumulate exp() results
        vaddps  zmm20,zmm20,zmm4
        vaddps  zmm20,zmm20,zmm17
        add     rdi,48*4                        # advance input by 48 elements
        test    rsi,rsi
        jz      .LComputeSumExp.SkipStoreResultsBy48
        vmovups ZMMWORD PTR [rsi],zmm1
        vmovups ZMMWORD PTR [rsi+64],zmm4
        vmovups ZMMWORD PTR [rsi+128],zmm17
        add     rsi,48*4                        # advance output by 48 elements

.LComputeSumExp.SkipStoreResultsBy48:
        sub     rdx,48
        jae     .LComputeSumExp.ComputeExpBy48Loop

.LComputeSumExp.ProcessRemainingCount:
        add     rdx,48                          # correct for over-subtract above
        jz      .LComputeSumExp.ReduceAccumulator
        mov     eax,-1
        kmovw   k1,eax                          # update mask to access all elements

.LComputeSumExp.ComputeExpBy16Loop:
        cmp     rdx,16
        jae     .LComputeSumExp.ProcessSingleVector
        lea     r10,C_UNDERSCORE(MlasOpmask16BitTableAvx512)[rip]
        kmovw   k1,WORD PTR [r10+rdx*2]

.LComputeSumExp.ProcessSingleVector:
        vaddps  zmm0{k1}{z},zmm19,ZMMWORD PTR [rdi]
                                                # bias by negative maximum value
        vmaxps  zmm0,zmm21,zmm0                 # clamp lower bound
        vmovaps zmm2,zmm23
        vfmadd213ps zmm2,zmm0,zmm22             # (input / ln2) plus rounding bias
        vmovaps zmm1,zmm26                      # p = poly_0
        vsubps  zmm2,zmm2,zmm22                 # m = round(input / ln2)
        vfmadd231ps zmm0,zmm2,zmm24             # range reduce: x -= (m * ln2_high)
        vfmadd231ps zmm0,zmm2,zmm25             # range reduce: x -= (m * ln2_low)
        vfmadd213ps zmm1,zmm0,zmm27             # p = p * x + poly_1
        vfmadd213ps zmm1,zmm0,zmm28             # p = p * x + poly_2
        vfmadd213ps zmm1,zmm0,zmm29             # p = p * x + poly_3
        vfmadd213ps zmm1,zmm0,zmm30             # p = p * x + poly_4
        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_5
        vfmadd213ps zmm1,zmm0,zmm31             # p = p * x + poly_6
        vscalefps zmm1,zmm1,zmm2
        vaddps  zmm20{k1},zmm20,zmm1            # accumulate exp() results
        add     rdi,16*4                        # advance input by 16 elements
        test    rsi,rsi
        jz      .LComputeSumExp.SkipStoreResultsBy16
        vmovups ZMMWORD PTR [rsi]{k1},zmm1
        add     rsi,16*4                        # advance output by 16 elements

.LComputeSumExp.SkipStoreResultsBy16:
        sub     rdx,16
        ja      .LComputeSumExp.ComputeExpBy16Loop

.LComputeSumExp.ReduceAccumulator:
        vextractf64x4 ymm0,zmm20,1
        vaddps  zmm0,zmm0,zmm20
        vhaddps ymm0,ymm0,ymm0
        vhaddps ymm0,ymm0,ymm0
        vextractf128 xmm1,ymm0,1
        vaddss  xmm0,xmm0,xmm1

        vzeroupper
        ret

        .end
