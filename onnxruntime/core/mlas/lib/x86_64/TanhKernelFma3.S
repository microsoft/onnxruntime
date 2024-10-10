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
#include "TransKernelCommon.h"


        .intel_syntax noprefix

        .text

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

        FUNCTION_ENTRY MlasComputeTanhF32KernelFma3

        lea     rax,C_UNDERSCORE(MlasTanhConstants)[rip]
        vbroadcastss ymm5,  0x00[rax]  // nc2
        vbroadcastss ymm6,  0x04[rax]  // nc1
        vbroadcastss ymm4,  0x08[rax]  // nc0
        vbroadcastss ymm7,  0x0c[rax]  // dc2
        vbroadcastss ymm8,  0x10[rax]  // dc1
        vbroadcastss ymm9,  0x14[rax]  // dc0
        vbroadcastss ymm10, 0x18[rax]  // absmask
        vbroadcastss ymm11, 0x1c[rax]  // bound
        sub     rdx,8
        jb      .LProcessRemainingCount

.LComputeTanhBy8Loop:
        vandps      ymm0,ymm10,YMMWORD PTR [rdi]
        vmovaps     ymm3, ymm5
        vmovaps     ymm13, ymm7
        vxorps      ymm1, ymm0, YMMWORD PTR [rdi]
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
        add         rdi,8*4                         # advance input by 8 elements
        vmovups     YMMWORD PTR [rsi],ymm0
        add         rsi,8*4                         # advance output by 8 elements
        sub         rdx,8
        jae     .LComputeTanhBy8Loop

.LProcessRemainingCount:
        add         rdx,8                           # correct for over-subtract above
        jz          .LExitKernel
        neg         rdx
        lea         r10,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovups     ymm15,YMMWORD PTR [r10+rdx*4]
        vmaskmovps  ymm0,ymm15,YMMWORD PTR [rdi]
        vandps      ymm0,ymm10,ymm0
        vmovaps     ymm3, ymm5
        vmovaps     ymm13, ymm7
        vxorps      ymm1, ymm0, YMMWORD PTR [rdi]
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
        vmaskmovps  YMMWORD PTR [rsi],ymm15,ymm0

.LExitKernel:
        vzeroupper
        ret
