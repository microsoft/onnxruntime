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

        MACRO
        UdotByElement $DestReg, $Src1Reg, $Src2Reg, $Index

        DCD     0x6F80E000:OR:($DestReg):OR:($Src1Reg:SHL:5):OR:($Src2Reg:SHL:16):OR:(($Index:AND:2):SHL:10):OR:(($Index:AND:1):SHL:21)

        MEND
