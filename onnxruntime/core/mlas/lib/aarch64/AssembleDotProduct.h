/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    AssembleDotProduct.h

Abstract:

    This module contains macros to build Advanced SIMD dot product instructions
    for toolchains that do not natively support this newer instruction set
    extension.

    This implementation uses ARM v8.4 dot product instructions.

--*/

/*++

Macro Description:

    This macro builds a UDOT instruction of the form:

        UDOT DestReg.4s, Src1Reg.16b, Src2Reg.4b[Index]

Arguments:

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

    Index - Specifies the element index of the second source register.

--*/

        .macro  UdotByElement DestReg, Src1Reg, Src2Reg, Index

        .set    Instruction, 0x6F80E000
        .set    Instruction, Instruction + (\DestReg\() << 0)
        .set    Instruction, Instruction + (\Src1Reg\() << 5)
        .set    Instruction, Instruction + (\Src2Reg\() << 16)
        .set    Instruction, Instruction + ((\Index\() & 2) << 10)
        .set    Instruction, Instruction + ((\Index\() & 1) << 21)

        .inst   Instruction

        .endm
