/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    TanhKernelFma3.s

Abstract:

    This module implements a kernel for computing the hyperbolic tangent
    function for a buffer of elements.

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

//
// Structure layout for the tanh constants block.
//

        .equ    TanhConstants_LowerRange, 0
        .equ    TanhConstants_UpperRange, 4
        .equ    TanhConstants_alpha_13, 8
        .equ    TanhConstants_alpha_11, 12
        .equ    TanhConstants_alpha_9, 16
        .equ    TanhConstants_alpha_7, 20
        .equ    TanhConstants_alpha_5, 24
        .equ    TanhConstants_alpha_3, 28
        .equ    TanhConstants_alpha_1, 32
        .equ    TanhConstants_beta_6, 36
        .equ    TanhConstants_beta_4, 40
        .equ    TanhConstants_beta_2, 44
        .equ    TanhConstants_beta_0, 48

//
// Stack frame layout for the tanh kernel.
//

        .equ    TanhKernelFrame_CountN, -8
        .equ    TanhKernelFrame_ReturnAddress, 0

/*++

Routine Description:

    This routine implements a vectorized kernel for the hyperbolic tangent
    function.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx)  - Supplies the number of elements to process.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasTanhKernelFma3)
C_UNDERSCORE(MlasTanhKernelFma3):

        lea     rax,C_UNDERSCORE(MlasTanhConstants)[rip]
        vbroadcastss ymm4,TanhConstants_LowerRange[rax]
        vbroadcastss ymm5,TanhConstants_UpperRange[rax]
        vbroadcastss ymm6,TanhConstants_alpha_13[rax]
        vbroadcastss ymm7,TanhConstants_alpha_11[rax]
        vbroadcastss ymm8,TanhConstants_alpha_9[rax]
        vbroadcastss ymm9,TanhConstants_alpha_7[rax]
        vbroadcastss ymm10,TanhConstants_alpha_5[rax]
        vbroadcastss ymm11,TanhConstants_alpha_3[rax]
        vbroadcastss ymm12,TanhConstants_alpha_1[rax]
        vbroadcastss ymm13,TanhConstants_beta_6[rax]
        vbroadcastss ymm14,TanhConstants_beta_2[rax]
        vbroadcastss ymm15,TanhConstants_beta_0[rax]

        sub     rdx,8
        jb      .LProcessRemainingCount

.LComputeTanhBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rdi]     # clamp lower bound
        vmovaps ymm2,ymm7
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,TanhConstants_beta_4[rax]
        vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9
        vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7
        vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5
        vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3
        vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm2,ymm0,ymm2                  # p = x * p
        vdivps  ymm0,ymm2,ymm3                  # tanh = p / q
        add     rdi,8*4                         # advance input by 8 elements
        vmovups YMMWORD PTR [rsi],ymm0
        add     rsi,8*4                         # advance output by 8 elements
        sub     rdx,8
        jae     .LComputeTanhBy8Loop

.LProcessRemainingCount:
        add     rdx,8                           # correct for over-subtract above
        jz      .LExitKernel
        mov     DWORD PTR TanhKernelFrame_CountN[rsp],edx
        vbroadcastss ymm2,DWORD PTR TanhKernelFrame_CountN[rsp]
        vpcmpgtd ymm2,ymm2,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rdi]
        vmaxps  ymm0,ymm4,ymm0                  # clamp lower bound
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,TanhConstants_beta_4[rax]
        vfmadd231ps ymm7,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm7,ymm1,ymm8              # p = x2 * p + alpha_9
        vfmadd213ps ymm7,ymm1,ymm9              # p = x2 * p + alpha_7
        vfmadd213ps ymm7,ymm1,ymm10             # p = x2 * p + alpha_5
        vfmadd213ps ymm7,ymm1,ymm11             # p = x2 * p + alpha_3
        vfmadd213ps ymm7,ymm1,ymm12             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm7,ymm0,ymm7                  # p = x * p
        vdivps  ymm0,ymm7,ymm3                  # tanh = p / q
        vmaskmovps YMMWORD PTR [rsi],ymm2,ymm0

.LExitKernel:
        vzeroupper
        ret

        .end
