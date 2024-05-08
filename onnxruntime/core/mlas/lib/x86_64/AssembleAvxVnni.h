/*++

Copyright (c) 2020 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    AssembleAvxVnni.h

Abstract:

    This module contains macros to build VNNI instructions for toolchains that
    do not natively support this newer instruction set extension.

--*/

//
// Map friendly register names to the encoded register index.
//

        .equ    .LYmmIndex_ymm0, 0
        .equ    .LYmmIndex_ymm1, 1
        .equ    .LYmmIndex_ymm2, 2
        .equ    .LYmmIndex_ymm3, 3
        .equ    .LYmmIndex_ymm4, 4
        .equ    .LYmmIndex_ymm5, 5
        .equ    .LYmmIndex_ymm6, 6
        .equ    .LYmmIndex_ymm7, 7
        .equ    .LYmmIndex_ymm8, 8
        .equ    .LYmmIndex_ymm9, 9
        .equ    .LYmmIndex_ymm10, 10
        .equ    .LYmmIndex_ymm11, 11
        .equ    .LYmmIndex_ymm12, 12
        .equ    .LYmmIndex_ymm13, 13
        .equ    .LYmmIndex_ymm14, 14
        .equ    .LYmmIndex_ymm15, 15

        .equ    .LXmmIndex_xmm0, 0
        .equ    .LXmmIndex_xmm1, 1
        .equ    .LXmmIndex_xmm2, 2
        .equ    .LXmmIndex_xmm3, 3
        .equ    .LXmmIndex_xmm4, 4
        .equ    .LXmmIndex_xmm5, 5
        .equ    .LXmmIndex_xmm6, 6
        .equ    .LXmmIndex_xmm7, 7
        .equ    .LXmmIndex_xmm8, 8
        .equ    .LXmmIndex_xmm9, 9
        .equ    .LXmmIndex_xmm10, 10
        .equ    .LXmmIndex_xmm11, 11
        .equ    .LXmmIndex_xmm12, 12
        .equ    .LXmmIndex_xmm13, 13
        .equ    .LXmmIndex_xmm14, 14
        .equ    .LXmmIndex_xmm15, 15

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

        instr ymm1,ymm2,ymm3

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

--*/

        .macro VnniYmmYmmYmm Opcode, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LYmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + (1 << 6)
        .set    Payload0, Payload0 + ((((.LYmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)

        .set    Payload1, 0x05              # "66" prefix
        .set    Payload1, Payload1 + (((.LYmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LYmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LYmmIndex_\Src2Reg\() & 7)

        .byte   0xC4, Payload0, Payload1, \Opcode\(), ModRMByte

        .endm

        .macro VpdpbusdYmmYmmYmm DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 0x50, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbusdsYmmYmmYmm DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 0x51, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdYmmYmmYmm DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 0x52, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdsYmmYmmYmm DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 0x53, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

        instr xmm1,xmm2,xmm3

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

--*/

        .macro VnniXmmXmmXmm Opcode, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LXmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + (1 << 6)
        .set    Payload0, Payload0 + ((((.LXmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)

        .set    Payload1, 0x05              # "66" prefix
        .set    Payload1, Payload1 + (((.LXmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LXmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LXmmIndex_\Src2Reg\() & 7)

        .byte   0xC4, Payload0, Payload1, \Opcode\(), ModRMByte

        .endm

        .macro VpdpbusdXmmXmmXmm DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 0x50, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbusdsXmmXmmXmm DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 0x51, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdXmmXmmXmm DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 0x52, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdsXmmXmmXmm DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 0x53, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

        instr ymm1,ymm2,ymm3

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    Prefix - Specifies the opcode prefix for payload 1

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

--*/
        .macro Avx2VnniYmmYmmYmm Opcode, Prefix, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LYmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + (1 << 6)
        .set    Payload0, Payload0 + ((((.LYmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)

        .set    Payload1, 0x04 + \Prefix\()     # 256-bit length and opcode prefix
        .set    Payload1, Payload1 + (((.LYmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LYmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LYmmIndex_\Src2Reg\() & 7)

        .byte   0xC4, Payload0, Payload1, \Opcode\(), ModRMByte

        .endm

        .macro VpdpbssdYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x50, 0x03, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbssdsYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x51, 0x03, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbsudYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x50, 0x02, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbsudsYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x51, 0x02, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbuudYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x50, 0x00, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbuudsYmmYmmYmm DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 0x51, 0x00, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

        instr xmm1,xmm2,xmm3

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    Prefix - Specifies the opcode prefix for payload 1

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

--*/
        .macro Avx2VnniXmmXmmXmm Opcode, Prefix, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LYmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + (1 << 6)
        .set    Payload0, Payload0 + ((((.LYmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)

        .set    Payload1, 0x00 + \Prefix\()     # 128-bit length and opcode prefix
        .set    Payload1, Payload1 + (((.LYmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LYmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LYmmIndex_\Src2Reg\() & 7)

        .byte   0xC4, Payload0, Payload1, \Opcode\(), ModRMByte

        .endm

        .macro VpdpbssdXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x50, 0x03, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbssdsXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x51, 0x03, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbsudXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x50, 0x02, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbsudsXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x51, 0x02, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbuudXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x50, 0x00, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbuudsXmmXmmXmm DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 0x51, 0x00, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm
