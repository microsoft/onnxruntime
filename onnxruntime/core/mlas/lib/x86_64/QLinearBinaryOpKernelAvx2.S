/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QLinearBinaryOpKernelAvx2.S

Abstract:

    This module implements the kernels for the quantized linear add
    for element type int8_t and uint8_t.

    This implementation uses AVX2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

// Used share data as below
// MLasPackBytesMM256VpshufbControl:
//         .byte 0,4,8,12,        255,255,255,255, 255,255,255,255, 255,255,255,255
//         .byte 255,255,255,255, 0,4,8,12,        255,255,255,255, 255,255,255,255
// MLasPackBytesMM256VpermpsControl:
//         .int  0, 5, 2, 3, 4, 1, 6, 7

//
// Stack frame layout for the QLinearBinaryElementwise kernel.
//
        .equ    .LQLinearBinaryElementwiseFrame_ReturnAddress, 0
        .equ    .LQLinearBinaryElementwiseFrame_N, 8

        .text

/*++

Macro Description:

   This macro generates code to unpack 8 x (s/u)int8 to 8 x int32,
   according to signed/unsigned.

Arguments:

   Source - Supplies address of 8 x 8bits integers.

   Target - target ymm register.

   DataType - S8 or U8

--*/

        .macro UnpackBytesDWords Target, Source, DataType
.ifeqs "\DataType\()","S8"
        vpmovsxbd \Target\(),\Source\()
.else
        vpmovzxbd \Target\(),\Source\()
.endif
        .endm


/*++

Macro Description:

   This macro generates code to set Target 64bits register with the
   max value of signed/unsigned int8 specified by DataType.

Arguments:

   Target - target 64bits register.

   DataType - S8 or U8.

--*/

        .macro SetMax8BitsValue Target, DataType
.ifeqs "\DataType\()","S8"
        mov \Target\(),QWORD PTR 127
.else
        mov \Target\(),QWORD PTR 255
.endif
        .endm


/*++

Macro Description:

   This macro generates code to set Target 64bits register with the
   min value of signed/unsigned int8 specified by DataType.

Arguments:

   Target - target 64bits register.

   DataType - S8 or U8.

--*/

        .macro SetMin8BitsValue Target, DataType
.ifeqs "\DataType\()","S8"
        mov \Target\(),QWORD PTR -128
.else
        mov \Target\(),QWORD PTR 0
.endif
        .endm


/*++

Macro Description:

   This macro generates code for function QLinearOpName()
   on the specified signed/unsigned int8 DataType.

Arguments:

   DataType - S8 or U8.

   OpName - Name of the QLinearOp, like Add, Mul, etc.

   OpInstruction - the assembly code prefix which op() two ymm vector of floats,
                   like vaddps, vmulps, etc

--*/

        .macro QLinearBinaryOpAvx2 DataType, OpName, OpInstruction

/*++
Routine Description:

    This routine implements the kernels for the Quantize Linear \OpName\()
    for element type \DataType\(), vector on vector.

Arguments:

    InputA (rdi) - Supplies the address of InputA.

    ScaleA (xmm0) - Supplies A's Scale value in float.

    ZeroPointA (rsi) - Supplies A's zero point value.

    InputB (rdx) - Supplies the address of InputB.

    ScaleB (xmm1) - Supplies B's Scale value in float.

    ZeroPointB (rcx) - Supplies B's zero point value.

    ScaleC (xmm2) - Supplies C's Scale value in float.

    ZeroPointC (r8) - Supplies C's zero point value.

    OutputC (r9) - Supplies the address of OutputC.

    N - Supplies the number of elements to calculate.

Return Value:

    None.
--*/
        .globl  C_UNDERSCORE(MlasQLinear\OpName\()\DataType\()KernelAvx2)
C_UNDERSCORE(MlasQLinear\OpName\()\DataType\()KernelAvx2):
        vbroadcastss ymm0,xmm0                  # Vector of ScaleA
        vbroadcastss ymm1,xmm1                  # Vector of ScaleB
        vbroadcastss ymm2,xmm2                  # Vector of ScaleC
        movq    xmm3,rsi                        # ZeroPointA, lower 32bits are zero/sign extended already.
        movq    xmm4,rcx                        # ZeroPointB, lower 32bits are zero/sign extended already.
        movq    xmm5,r8                         # ZeroPointC, lower 32bits are zero/sign extended already.
        vbroadcastss ymm3,xmm3                  # Vector of ZeroPointA
        vbroadcastss ymm4,xmm4                  # Vector of ZeroPointB
        vbroadcastss ymm5,xmm5                  # Vector of ZeroPointC

        lea     r8,C_UNDERSCORE(MLasPackBytesMM256VpshufbControl)[rip]
        vmovaps ymm10,[r8]
        lea     rsi,C_UNDERSCORE(MLasPackBytesMM256VpermpsControl)[rip]
        vmovaps ymm11,[rsi]

        SetMax8BitsValue r8,\DataType\()
        movq         xmm6,r8
        vbroadcastss ymm6,xmm6
        vpsubd       ymm6,ymm6,ymm5

        SetMin8BitsValue rsi,\DataType\()
        movq         xmm7,rsi
        vbroadcastss ymm7,xmm7
        vpsubd       ymm7,ymm7,ymm5

        mov     rcx,.LQLinearBinaryElementwiseFrame_N[rsp]
        mov     rax,QWORD PTR 8

.LQLinear\OpName\()\DataType\()Avx2Process8Loop:
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit

        UnpackBytesDWords ymm8,[rdi],\DataType\()
        UnpackBytesDWords ymm9,[rdx],\DataType\()
        vpsubd  ymm9,ymm9,ymm4                  # - Zero Points respectively
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8                     # convert to float
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm8,ymm8,ymm0                  # * Scales respectively 
        vmulps  ymm9,ymm9,ymm1

        \OpInstruction  ymm8,ymm8,ymm9          # \OpName\() two float values

        vdivps  ymm8,ymm8,ymm2                  # Quantize 8 values, / ScaleC
        add     rdi,rax                         # two out-of-order instructions
        add     rdx,rax
        vcvtps2dq ymm8,ymm8                     # round()
        vpmaxsd ymm8,ymm8,ymm7
        vpminsd ymm8,ymm8,ymm6
        vpaddd  ymm8,ymm8,ymm5                  # + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 # pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        sub     rcx,rax
        jb      .LQLinear\OpName\()\DataType\()Avx2StoreLessThan8

        movsd   QWORD PTR [r9],xmm8
        add     r9,rax
        jmp     .LQLinear\OpName\()\DataType\()Avx2Process8Loop

.LQLinear\OpName\()\DataType\()Avx2StoreLessThan8:
        add     rcx,rax
        pextrq  rax,xmm8,0

.LQLinear\OpName\()\DataType\()Avx2StoreLoop:
        mov     BYTE PTR [r9],al
        shr     rax,8
        inc     r9
        dec     rcx
        jnz     .LQLinear\OpName\()\DataType\()Avx2StoreLoop

.LQLinear\OpName\()\DataType\()Avx2Exit:
        vzeroupper
        ret

        .endm


//
// Generate the QLinearAdd Avx2 S8 kernel.
//

QLinearBinaryOpAvx2 S8,Add,vaddps

//
// Generate the QLinearAdd Avx2 U8 kernel.
//

QLinearBinaryOpAvx2 U8,Add,vaddps

        .end
