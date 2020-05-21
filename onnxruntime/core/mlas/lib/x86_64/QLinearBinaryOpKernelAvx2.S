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
// MlasPackBytesMM256VpshufbControl:
//         .byte 0,4,8,12,        255,255,255,255, 255,255,255,255, 255,255,255,255
//         .byte 255,255,255,255, 0,4,8,12,        255,255,255,255, 255,255,255,255
// MlasPackBytesMM256VpermpsControl:
//         .int  0, 5, 2, 3, 4, 1, 6, 7

//
// Stack frame layout for the QLinearBinaryElementwise kernel.
//
        .equ    .LQLinearBinaryElementwiseFrame_ReturnAddress, 0
        .equ    .LQLinearBinaryElementwiseFrame_LengthA, 8
        .equ    .LQLinearBinaryElementwiseFrame_LengthB, 16

//
// Define the some constants for U8/S8.
//
        .equ    .MaxValue_S8, 127
        .equ    .MinValue_S8, -128
        .equ    .MaxValue_U8, 255
        .equ    .MinValue_U8, 0

//
// Define the some instruction alias for U8/S8.
//

#define TYPED_INSTRUCTION_ALIAS(TypedAlias, InstruncionName) \
        .macro TypedAlias Operand:vararg; InstruncionName \Operand\(); .endm;

TYPED_INSTRUCTION_ALIAS(ExtendBits_S8, movsx)
TYPED_INSTRUCTION_ALIAS(ExtendBits_U8, movzx)
TYPED_INSTRUCTION_ALIAS(UnpackBytesDWords_S8, vpmovsxbd)
TYPED_INSTRUCTION_ALIAS(UnpackBytesDWords_U8, vpmovzxbd)

        .text

/*++

Macro Description:

   This macro generates code for function QLinearOpName() on the specified
   signed/unsigned int8 DataType.

Arguments:

   DataType - S8 or U8.

   OpName - Name of the QLinearOp, like Add, Mul, etc.

   OpInstruction - the assembly code prefix which op() two ymm vector of floats,
                   like vaddps, vmulps, etc

--*/

        .macro QLinearBinaryOpAvx2 DataType, OpName, OpInstruction

/*++
Routine Description:

    This routine implements the kernels for the Quantize Linear \OpName\() for
    element type \DataType\(), vector on vector.

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

    LengthA - Supplies the number of elements of InputA.

    LengthB - Supplies the number of elements of InputB.
              LengthB == LengthA or LengthA == 1 or LengthB == 1

Return Value:

    None.
--*/
        .globl  C_UNDERSCORE(MlasQLinear\OpName\()\DataType\()KernelAvx2)
C_UNDERSCORE(MlasQLinear\OpName\()\DataType\()KernelAvx2):
        vbroadcastss ymm0,xmm0                  # Vector of ScaleA
        vbroadcastss ymm1,xmm1                  # Vector of ScaleB
        vbroadcastss ymm2,xmm2                  # Vector of ScaleC
        vmovd   xmm3,esi                        # ZeroPointA.
        vmovd   xmm4,ecx                        # ZeroPointB.
        vmovd   xmm5,r8d                        # ZeroPointC.
        vpbroadcastd ymm3,xmm3                  # Vector of ZeroPointA
        vpbroadcastd ymm4,xmm4                  # Vector of ZeroPointB
        vpbroadcastd ymm5,xmm5                  # Vector of ZeroPointC

        vmovaps ymm10,YMMWORD PTR C_UNDERSCORE(MlasPackBytesMM256VpshufbControl)[rip]
        vmovaps ymm11,YMMWORD PTR C_UNDERSCORE(MlasPackBytesMM256VpermpsControl)[rip]

        mov     r8d,.MaxValue_\DataType
        vmovd   xmm6,r8d
        vpbroadcastd ymm6,xmm6                  # Max U8/S8 Int Value Vector

.ifeq .MinValue_\DataType
        vxorps  ymm7,ymm7,ymm7                  # Min U8 Int Value Vector
.else
        mov     esi,.MinValue_\DataType
        vmovd   xmm7,esi
        vpbroadcastd ymm7,xmm7                  # Min S8 Int Value Vector
.endif

        vpsubd  ymm6,ymm6,ymm5
        vpsubd  ymm7,ymm7,ymm5
        vcvtdq2ps ymm6,ymm6                     # Float Max Value Vector
        vcvtdq2ps ymm7,ymm7                     # Float Min Value Vector

        # goto different loop according to broading cast semantic
        mov     rcx,.LQLinearBinaryElementwiseFrame_LengthA[rsp]
        dec     rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Process8EntranceScalarOnVector
        mov     rcx,.LQLinearBinaryElementwiseFrame_LengthB[rsp]
        dec     rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Process8EntranceVectorOnScalar
        inc     rcx
        jmp     .LQLinear\OpName\()\DataType\()Avx2Process8LoopVectorOnVector

.LQLinear\OpName\()\DataType\()Avx2Process8EntranceScalarOnVector:
        mov     rcx,.LQLinearBinaryElementwiseFrame_LengthB[rsp]
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit
        ExtendBits_\DataType eax,BYTE PTR [rdi] # ScalarValueA
        vmovd   xmm8,eax
        vpbroadcastd ymm8,xmm8                  # IntVectorA
        vpsubd  ymm8,ymm8,ymm3                  # - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     # FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  # * ScaleA

.LQLinear\OpName\()\DataType\()Avx2Process8LoopScalarOnVector:
        UnpackBytesDWords_\DataType ymm9,[rdx]  # IntVectorB
        vpsubd  ymm9,ymm9,ymm4                  # - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     # FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  # * ScaleB

        \OpInstruction  ymm9,ymm8,ymm9          # \OpName\() two float values

        vdivps  ymm9,ymm9,ymm2                  # Quantize 8 values, / ScaleC
        add     rdx,8                           # out-of-order instruction(s)
        sub     rcx,8                           # out-of-order instruction(s), set flag for below jb
        vminps  ymm9,ymm9,ymm6
        vmaxps  ymm9,ymm9,ymm7
        vcvtps2dq ymm9,ymm9                     # nearbyintf()
        vpaddd  ymm9,ymm9,ymm5                  # + ZeroPointC
        vpshufb ymm9,ymm9,ymm10                 # pack 32bits integers into 8bit integers
        vpermps ymm9,ymm11,ymm9

        jb      .LQLinear\OpName\()\DataType\()Avx2StoreLessThan8ScalarOnVector

        vmovq   QWORD PTR [r9],xmm9
        add     r9,8
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit
        jmp     .LQLinear\OpName\()\DataType\()Avx2Process8LoopScalarOnVector

.LQLinear\OpName\()\DataType\()Avx2StoreLessThan8ScalarOnVector:
        add     rcx,8
        vpextrq rax,xmm9,0
        jmp     .LQLinear\OpName\()\DataType\()Avx2StoreLoopGeneral

.LQLinear\OpName\()\DataType\()Avx2Process8EntranceVectorOnScalar:
        mov     rcx,.LQLinearBinaryElementwiseFrame_LengthA[rsp]
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit
        ExtendBits_\DataType eax,BYTE PTR [rdx] # ScalarValueB
        vmovd   xmm9,eax
        vpbroadcastd ymm9,xmm9                  # IntVectorB
        vpsubd  ymm9,ymm9,ymm4                  # - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     # FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  # * ScaleB

.LQLinear\OpName\()\DataType\()Avx2Process8LoopVectorOnScalar:
        UnpackBytesDWords_\DataType ymm8,[rdi]  # IntVectorA
        vpsubd  ymm8,ymm8,ymm3                  # - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     # FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  # * ScaleA

        \OpInstruction  ymm8,ymm8,ymm9          # \OpName\() two float values

        vdivps  ymm8,ymm8,ymm2                  # Quantize 8 values, / ScaleC
        add     rdi,8                           # out-of-order instruction(s)
        sub     rcx,8                           # out-of-order instruction(s), set flag for below jb
        vminps  ymm8,ymm8,ymm6
        vmaxps  ymm8,ymm8,ymm7
        vcvtps2dq ymm8,ymm8                     # nearbyintf()
        vpaddd  ymm8,ymm8,ymm5                  # + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 # pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        jb      .LQLinear\OpName\()\DataType\()Avx2StoreLessThan8VectorOnScalar

        vmovq   QWORD PTR [r9],xmm8
        add     r9,8
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit
        jmp     .LQLinear\OpName\()\DataType\()Avx2Process8LoopVectorOnScalar

.LQLinear\OpName\()\DataType\()Avx2StoreLessThan8VectorOnScalar:
        add     rcx,8
        vpextrq rax,xmm8,0
        jmp     .LQLinear\OpName\()\DataType\()Avx2StoreLoopGeneral

.LQLinear\OpName\()\DataType\()Avx2Process8LoopVectorOnVector:
        test    rcx,rcx
        jz      .LQLinear\OpName\()\DataType\()Avx2Exit

        UnpackBytesDWords_\DataType ymm8,[rdi]  # IntVectorA
        UnpackBytesDWords_\DataType ymm9,[rdx]  # IntVectorB
        vpsubd  ymm8,ymm8,ymm3                  # - ZeroPointA
        vpsubd  ymm9,ymm9,ymm4                  # - ZeroPointB
        vcvtdq2ps ymm8,ymm8                     # FloatVectorA
        vcvtdq2ps ymm9,ymm9                     # FloatVectorB
        vmulps  ymm8,ymm8,ymm0                  # * ScaleA
        vmulps  ymm9,ymm9,ymm1                  # * ScaleB

        \OpInstruction  ymm8,ymm8,ymm9          # \OpName\() two float values

        vdivps  ymm8,ymm8,ymm2                  # Quantize 8 values, / ScaleC
        add     rdi,8                           # out-of-order instruction(s)
        add     rdx,8                           # out-of-order instruction(s)
        sub     rcx,8                           # out-of-order instruction(s), set flag for below jb
        vminps  ymm8,ymm8,ymm6                  # When NaN in first vector, return value in second vector
        vmaxps  ymm8,ymm8,ymm7
        vcvtps2dq ymm8,ymm8                     # nearbyintf()
        vpaddd  ymm8,ymm8,ymm5                  # + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 # pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        jb      .LQLinear\OpName\()\DataType\()Avx2StoreLessThan8VectorOnVector

        vmovq   QWORD PTR [r9],xmm8
        add     r9,8
        jmp     .LQLinear\OpName\()\DataType\()Avx2Process8LoopVectorOnVector

.LQLinear\OpName\()\DataType\()Avx2StoreLessThan8VectorOnVector:
        add     rcx,8
        vpextrq rax,xmm8,0

.LQLinear\OpName\()\DataType\()Avx2StoreLoopGeneral:
        mov     BYTE PTR [r9],al
        shr     rax,8
        inc     r9
        dec     rcx
        jnz     .LQLinear\OpName\()\DataType\()Avx2StoreLoopGeneral

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
