/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    LogisticKernelFma3.s

Abstract:

    This module implements a kernel for computing the logistic function for a
    buffer of elements.

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"
#include "TransKernelCommon.h"

        .intel_syntax noprefix

        .text

/*++

Routine Description:

    This routine implements a vectorized kernel for the logistic function.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx)  - Supplies the number of elements to process.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasComputeLogisticF32KernelFma3

        lea     rax,C_UNDERSCORE(MlasLogisticConstants)[rip]
        vbroadcastss ymm4,.LLogisticConstants_LowerRange[rax]
        vbroadcastss ymm5,.LLogisticConstants_UpperRange[rax]
        vbroadcastss ymm6,.LLogisticConstants_alpha_9[rax]
        vbroadcastss ymm7,.LLogisticConstants_alpha_7[rax]
        vbroadcastss ymm8,.LLogisticConstants_alpha_5[rax]
        vbroadcastss ymm9,.LLogisticConstants_alpha_3[rax]
        vbroadcastss ymm10,.LLogisticConstants_alpha_1[rax]
        vbroadcastss ymm11,.LLogisticConstants_beta_10[rax]
        vbroadcastss ymm12,.LLogisticConstants_beta_6[rax]
        vbroadcastss ymm13,.LLogisticConstants_beta_4[rax]
        vbroadcastss ymm14,.LLogisticConstants_beta_2[rax]
        vbroadcastss ymm15,.LLogisticConstants_beta_0[rax]

        sub     rdx,8
        jb      .LProcessRemainingCount

.LComputeLogisticBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rdi]     # clamp lower bound
        vmovaps ymm2,ymm7
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,.LLogisticConstants_beta_8[rax]
        vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7
        vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_5
        vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_3
        vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8
        vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6
        vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm2,ymm0,ymm2                  # p = x * p
        vbroadcastss ymm0,.LLogisticConstants_one_half[rax]
        vdivps  ymm2,ymm2,ymm3
        vxorps  ymm3,ymm3,ymm3
        vaddps  ymm0,ymm2,ymm0                  # logistic = p / q + 0.5
        vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound
        add     rdi,8*4                         # advance input by 8 elements
        vmovups YMMWORD PTR [rsi],ymm0
        add     rsi,8*4                         # advance output by 8 elements
        sub     rdx,8
        jae     .LComputeLogisticBy8Loop

.LProcessRemainingCount:
        add     rdx,8                           # correct for over-subtract above
        jz      .LExitKernel
        neg     rdx
        lea     r10,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovups ymm2,YMMWORD PTR [r10+rdx*4]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rdi]
        vmaxps  ymm0,ymm4,ymm0                  # clamp lower bound
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,.LLogisticConstants_beta_8[rax]
        vfmadd231ps ymm7,ymm1,ymm6              # p = x2 * alpha_9 + alpha_7
        vfmadd213ps ymm7,ymm1,ymm8              # p = x2 * p + alpha_5
        vfmadd213ps ymm7,ymm1,ymm9              # p = x2 * p + alpha_3
        vfmadd213ps ymm7,ymm1,ymm10             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm11             # q = x2 * beta_10 + beta_8
        vfmadd213ps ymm3,ymm1,ymm12             # q = x2 * q + beta_6
        vfmadd213ps ymm3,ymm1,ymm13             # q = x2 * q + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm7,ymm0,ymm7                  # p = x * p
        vbroadcastss ymm0,.LLogisticConstants_one_half[rax]
        vdivps  ymm7,ymm7,ymm3
        vxorps  ymm3,ymm3,ymm3
        vaddps  ymm0,ymm7,ymm0                  # logistic = p / q + 0.5
        vmaxps  ymm0,ymm3,ymm0                  # clamp lower bound
        vmaskmovps YMMWORD PTR [rsi],ymm2,ymm0

.LExitKernel:
        vzeroupper
        ret

        .end
