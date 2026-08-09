/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    TransKernelFma3.s

Abstract:

    This module implements kernels for various transcendental functions.

    This implementation uses AVX fused multiply/add instructions.

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

        FUNCTION_ENTRY MlasComputeExpF32KernelFma3

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss ymm4,.LExpConstants_LowerRange[rax]
        vbroadcastss ymm5,.LExpConstants_UpperRange[rax]
        vbroadcastss ymm6,.LExpConstants_MinimumExponent[rax]
        vbroadcastss ymm7,.LExpConstants_MaximumExponent[rax]
        vbroadcastss ymm8,.LExpConstants_RoundingBias[rax]
        vbroadcastss ymm9,.LExpConstants_Log2Low[rax]
        vbroadcastss ymm10,.LExpConstants_poly_0[rax]
        vbroadcastss ymm11,.LExpConstants_poly_1[rax]
        vbroadcastss ymm12,.LExpConstants_poly_2[rax]
        vbroadcastss ymm13,.LExpConstants_poly_3[rax]
        vbroadcastss ymm14,.LExpConstants_poly_4[rax]
        vbroadcastss ymm15,.LExpConstants_poly_56[rax]

        sub     rdx,8
        jb      .LComputeExp.ProcessRemainingCount

.LComputeExp.ComputeExpBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rdi]     # clamp lower bound
        vbroadcastss ymm2,.LExpConstants_Log2Reciprocal[rax]
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vbroadcastss ymm3,.LExpConstants_Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm8              # (x / ln2) plus rounding bias
        vsubps  ymm1,ymm2,ymm8                  # m = round(x / ln2)
        vfmadd231ps ymm0,ymm1,ymm3              # range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm9              # range reduce: x -= (m * ln2_low)
        vmovaps ymm1,ymm10                      # p = poly_0
        vfmadd213ps ymm1,ymm0,ymm11             # p = p * x + poly_1
        vpslld  ymm2,ymm2,23                    # shift m to exponent field
        vfmadd213ps ymm1,ymm0,ymm12             # p = p * x + poly_2
        vpminsd  ymm3,ymm2,ymm7                 # clamp upper normal exponent to +127
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_3
        vpmaxsd  ymm3,ymm3,ymm6                 # clamp lower normal exponent to -126
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_4
        vpsubd  ymm2,ymm2,ymm3                  # compute overflow exponent
        vpaddd  ymm3,ymm3,ymm7                  # add exponent bias to normal scale
        vpaddd  ymm2,ymm2,ymm7                  # add exponent bias to overflow scale
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_56
        vmulps  ymm0,ymm0,ymm2                  # scale x with overflow exponent
        vfmadd213ps ymm1,ymm0,ymm2              # p = p * (x * overflow) + overflow
        vmulps  ymm1,ymm1,ymm3                  # scale p with normal exponent
        add     rdi,8*4                         # advance input by 8 elements
        vmovups YMMWORD PTR [rsi],ymm1
        add     rsi,8*4                         # advance output by 8 elements
        sub     rdx,8
        jae     .LComputeExp.ComputeExpBy8Loop

.LComputeExp.ProcessRemainingCount:
        add     rdx,8                            # correct for over-subtract above
        jz      .LComputeExp.ExitKernel
        neg     rdx
        lea     r10,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovups ymm2,YMMWORD PTR [r10+rdx*4]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rdi]
        vmaxps  ymm0,ymm4,ymm0                  # clamp lower bound
        vbroadcastss ymm4,.LExpConstants_Log2Reciprocal[rax]
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vbroadcastss ymm3,.LExpConstants_Log2High[rax]
        vfmadd213ps ymm4,ymm0,ymm8              # (x / ln2) plus rounding bias
        vsubps  ymm1,ymm4,ymm8                  # m = round(x / ln2)
        vfmadd231ps ymm0,ymm1,ymm3              # range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm9              # range reduce: x -= (m * ln2_low)
        vmovaps ymm1,ymm10                      # p = poly_0
        vfmadd213ps ymm1,ymm0,ymm11             # p = p * x + poly_1
        vpslld  ymm4,ymm4,23                    # shift m to exponent field
        vfmadd213ps ymm1,ymm0,ymm12             # p = p * x + poly_2
        vpminsd  ymm3,ymm4,ymm7                 # clamp upper normal exponent to +127
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_3
        vpmaxsd  ymm3,ymm3,ymm6                 # clamp lower normal exponent to -126
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_4
        vpsubd  ymm4,ymm4,ymm3                  # compute overflow exponent
        vpaddd  ymm3,ymm3,ymm7                  # add exponent bias to normal scale
        vpaddd  ymm4,ymm4,ymm7                  # add exponent bias to overflow scale
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_5
        vmulps  ymm0,ymm0,ymm4                  # scale x with overflow exponent
        vfmadd213ps ymm1,ymm0,ymm4              # p = p * (x * overflow) + overflow
        vmulps  ymm1,ymm1,ymm3                  # scale p with normal exponent
        vmaskmovps YMMWORD PTR [rsi],ymm2,ymm1

.LComputeExp.ExitKernel:
        vzeroupper
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

    NegativeMaximum (rcx) - Supplies the address of the negative maximum value
        that is added to each element before computing the exponential function.

Return Value:

    Returns the sum of the exponential functions.

--*/

        FUNCTION_ENTRY MlasComputeSumExpF32KernelFma3

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss ymm9,DWORD PTR [rcx]       # broadcast negative maximum value
        vxorps  xmm10,xmm10,xmm10               # clear exp() accumulator
        sub     rdx,24
        jb      .LComputeSumExp.ProcessRemainingCount

.LComputeSumExp.ComputeExpBy24Loop:
        vbroadcastss ymm11,.LExpConstants_LowerRangeSumExp[rax]
        vbroadcastss ymm2,.LExpConstants_Log2Reciprocal[rax]
        vaddps  ymm0,ymm9,YMMWORD PTR [rdi]     # bias by negative maximum value
        vaddps  ymm3,ymm9,YMMWORD PTR [rdi+32]
        vaddps  ymm6,ymm9,YMMWORD PTR [rdi+64]
        vbroadcastss ymm15,.LExpConstants_RoundingBias[rax]
        vmaxps  ymm0,ymm11,ymm0                 # clamp lower bound
        vmovaps ymm5,ymm2
        vmaxps  ymm3,ymm11,ymm3
        vmovaps ymm8,ymm2
        vmaxps  ymm6,ymm11,ymm6
        vbroadcastss ymm13,.LExpConstants_Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm15             # (x / ln2) plus rounding bias
        vfmadd213ps ymm5,ymm3,ymm15
        vfmadd213ps ymm8,ymm6,ymm15
        vbroadcastss ymm14,.LExpConstants_Log2Low[rax]
        vsubps  ymm1,ymm2,ymm15                 # m = round(x / ln2)
        vsubps  ymm4,ymm5,ymm15
        vsubps  ymm7,ymm8,ymm15
        vfmadd231ps ymm0,ymm1,ymm13             # range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm3,ymm4,ymm13
        vfmadd231ps ymm6,ymm7,ymm13
        vfmadd231ps ymm0,ymm1,ymm14             # range reduce: x -= (m * ln2_low)
        vfmadd231ps ymm3,ymm4,ymm14
        vfmadd231ps ymm6,ymm7,ymm14
        vbroadcastss ymm1,.LExpConstants_poly_0[rax]
        vbroadcastss ymm13,.LExpConstants_poly_1[rax]
        vmovaps ymm4,ymm1
        vmovaps ymm7,ymm1
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_1
        vfmadd213ps ymm4,ymm3,ymm13
        vfmadd213ps ymm7,ymm6,ymm13
        vbroadcastss ymm14,.LExpConstants_poly_2[rax]
        vpslld  ymm2,ymm2,23                    # shift m to exponent field
        vpslld  ymm5,ymm5,23
        vpslld  ymm8,ymm8,23
        vbroadcastss ymm15,.LExpConstants_MaximumExponent[rax]
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_2
        vfmadd213ps ymm4,ymm3,ymm14
        vfmadd213ps ymm7,ymm6,ymm14
        vbroadcastss ymm13,.LExpConstants_poly_3[rax]
        vpaddd  ymm2,ymm2,ymm15                 # add exponent bias to scale
        vpaddd  ymm5,ymm5,ymm15
        vpaddd  ymm8,ymm8,ymm15
        vbroadcastss ymm14,.LExpConstants_poly_4[rax]
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_3
        vfmadd213ps ymm4,ymm3,ymm13
        vfmadd213ps ymm7,ymm6,ymm13
        vbroadcastss ymm15,.LExpConstants_poly_56[rax]
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_4
        vfmadd213ps ymm4,ymm3,ymm14
        vfmadd213ps ymm7,ymm6,ymm14
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_5
        vfmadd213ps ymm4,ymm3,ymm15
        vfmadd213ps ymm7,ymm6,ymm15
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_6
        vfmadd213ps ymm4,ymm3,ymm15
        vfmadd213ps ymm7,ymm6,ymm15
        vmulps  ymm1,ymm1,ymm2                  # scale p with exponent
        vmulps  ymm4,ymm4,ymm5
        vaddps  ymm10,ymm10,ymm1                # accumulate exp() results
        vmulps  ymm7,ymm7,ymm8
        vaddps  ymm10,ymm10,ymm4
        add     rdi,24*4                        # advance input by 24 elements
        vaddps  ymm10,ymm10,ymm7
        test    rsi,rsi
        jz      .LComputeSumExp.SkipStoreResultsBy24
        vmovups YMMWORD PTR [rsi],ymm1
        vmovups YMMWORD PTR [rsi+32],ymm4
        vmovups YMMWORD PTR [rsi+64],ymm7
        add     rsi,24*4                        # advance output by 24 elements

.LComputeSumExp.SkipStoreResultsBy24:
        sub     rdx,24
        jae     .LComputeSumExp.ComputeExpBy24Loop

.LComputeSumExp.ProcessRemainingCount:
        add     rdx,24                          # correct for over-subtract above
        jz      .LComputeSumExp.ReduceAccumulator
        vbroadcastss ymm11,.LExpConstants_LowerRangeSumExp[rax]

.LComputeSumExp.ComputeExpBy8Loop:
        cmp     rdx,8                           # remaining count < 8?
        jb      .LComputeSumExp.LoadPartialVector
        vmovups ymm0,YMMWORD PTR [rdi]
        jmp     .LComputeSumExp.ProcessSingleVector

.LComputeSumExp.LoadPartialVector:
        lea     r10,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        neg     rdx                             # carry flag unchanged
        vmovups ymm3,YMMWORD PTR [r10+rdx*4]
        vmaskmovps ymm0,ymm3,YMMWORD PTR [rdi]
        vandps  ymm9,ymm9,ymm3                  # mask unused maximum value to 0.0

.LComputeSumExp.ProcessSingleVector:
        vbroadcastss ymm2,.LExpConstants_Log2Reciprocal[rax]
        vaddps  ymm0,ymm9,ymm0                  # bias by negative maximum value
        vbroadcastss ymm15,.LExpConstants_RoundingBias[rax]
        vmaxps  ymm0,ymm11,ymm0                 # clamp lower bound
        vbroadcastss ymm13,.LExpConstants_Log2High[rax]
        vfmadd213ps ymm2,ymm0,ymm15             # (input / ln2) plus rounding bias
        vbroadcastss ymm14,.LExpConstants_Log2Low[rax]
        vsubps  ymm1,ymm2,ymm15                 # round(input / ln2)
        vfmadd231ps ymm0,ymm1,ymm13             # range reduce: x -= (m * ln2_high)
        vfmadd231ps ymm0,ymm1,ymm14             # range reduce: x -= (m * ln2_low)
        vbroadcastss ymm1,.LExpConstants_poly_0[rax]
        vbroadcastss ymm13,.LExpConstants_poly_1[rax]
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_1
        vbroadcastss ymm14,.LExpConstants_poly_2[rax]
        vpslld  ymm2,ymm2,23                    # # shift m to exponent field
        vbroadcastss ymm15,.LExpConstants_MaximumExponent[rax]
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_2
        vbroadcastss ymm13,.LExpConstants_poly_3[rax]
        vpaddd  ymm2,ymm2,ymm15                 # add exponent bias to scale
        vbroadcastss ymm14,.LExpConstants_poly_4[rax]
        vfmadd213ps ymm1,ymm0,ymm13             # p = p * x + poly_3
        vbroadcastss ymm15,.LExpConstants_poly_56[rax]
        vfmadd213ps ymm1,ymm0,ymm14             # p = p * x + poly_4
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_5
        vfmadd213ps ymm1,ymm0,ymm15             # p = p * x + poly_6
        vmulps  ymm1,ymm1,ymm2
        jb      .LComputeSumExp.StorePartialVector
                                                # remaining count < 8?
        vaddps  ymm10,ymm10,ymm1                # accumulate exp() results
        test    rsi,rsi                         # store exp() results?
        jz      .LComputeSumExp.SkipStoreResultsBy8
        vmovups YMMWORD PTR [rsi],ymm1
        add     rsi,8*4                         # advance output by 8 elements

.LComputeSumExp.SkipStoreResultsBy8:
        add     rdi,8*4                         # advance input by 8 elements
        sub     rdx,8
        jnz     .LComputeSumExp.ComputeExpBy8Loop
        jmp     .LComputeSumExp.ReduceAccumulator

.LComputeSumExp.StorePartialVector:
        vandps  ymm1,ymm1,ymm3                  # mask unused exp() results to 0.0
        vaddps  ymm10,ymm10,ymm1                # accumulate exp() results
        test    rsi,rsi                         # store exp() results?
        jz      .LComputeSumExp.ReduceAccumulator
        vmaskmovps YMMWORD PTR [rsi],ymm3,ymm1

.LComputeSumExp.ReduceAccumulator:
        vhaddps ymm10,ymm10,ymm10
        vhaddps ymm10,ymm10,ymm10
        vextractf128 xmm0,ymm10,1
        vaddss  xmm0,xmm0,xmm10

        vzeroupper
        ret

        .end
