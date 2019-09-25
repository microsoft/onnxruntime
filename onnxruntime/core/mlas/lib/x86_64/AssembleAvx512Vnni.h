/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    AssembleAvx512Vnni.h

Abstract:

    This module contains macros to build VNNI instructions for toolchains that
    do not natively support this newer instruction set extension.

--*/

//
// Map friendly register names to the encoded register index.
//

        .equ    .LZmmIndex_zmm0, 0
        .equ    .LZmmIndex_zmm1, 1
        .equ    .LZmmIndex_zmm2, 2
        .equ    .LZmmIndex_zmm3, 3
        .equ    .LZmmIndex_zmm4, 4
        .equ    .LZmmIndex_zmm5, 5
        .equ    .LZmmIndex_zmm6, 6
        .equ    .LZmmIndex_zmm7, 7
        .equ    .LZmmIndex_zmm8, 8
        .equ    .LZmmIndex_zmm9, 9
        .equ    .LZmmIndex_zmm10, 10
        .equ    .LZmmIndex_zmm11, 11
        .equ    .LZmmIndex_zmm12, 12
        .equ    .LZmmIndex_zmm13, 13
        .equ    .LZmmIndex_zmm14, 14
        .equ    .LZmmIndex_zmm15, 15
        .equ    .LZmmIndex_zmm16, 16
        .equ    .LZmmIndex_zmm17, 17
        .equ    .LZmmIndex_zmm18, 18
        .equ    .LZmmIndex_zmm19, 19
        .equ    .LZmmIndex_zmm20, 20
        .equ    .LZmmIndex_zmm21, 21
        .equ    .LZmmIndex_zmm22, 22
        .equ    .LZmmIndex_zmm23, 23
        .equ    .LZmmIndex_zmm24, 24
        .equ    .LZmmIndex_zmm25, 25
        .equ    .LZmmIndex_zmm26, 26
        .equ    .LZmmIndex_zmm27, 27
        .equ    .LZmmIndex_zmm28, 28
        .equ    .LZmmIndex_zmm29, 29
        .equ    .LZmmIndex_zmm30, 30
        .equ    .LZmmIndex_zmm31, 31

        .equ    .LGprIndex_rax, 0
        .equ    .LGprIndex_rcx, 1
        .equ    .LGprIndex_rdx, 2
        .equ    .LGprIndex_rbx, 3
        .equ    .LGprIndex_rbp, 5
        .equ    .LGprIndex_rsi, 6
        .equ    .LGprIndex_rdi, 7
        .equ    .LGprIndex_r8, 8
        .equ    .LGprIndex_r9, 9
        .equ    .LGprIndex_r10, 10
        .equ    .LGprIndex_r11, 11
        .equ    .LGprIndex_r12, 12
        .equ    .LGprIndex_r13, 13
        .equ    .LGprIndex_r14, 14
        .equ    .LGprIndex_r15, 15

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

        instr zmm1,zmm2,zmm3

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    Src2Reg - Specifies the second source register.

--*/

        .macro VnniZmmZmmZmm Opcode, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LZmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + ((((.LZmmIndex_\Src2Reg\() >> 4) & 1) ^ 1) << 6)
        .set    Payload0, Payload0 + ((((.LZmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)
        .set    Payload0, Payload0 + ((((.LZmmIndex_\DestReg\() >> 4) & 1) ^ 1) << 4)

        .set    Payload1, 0x05              # "66" prefix
        .set    Payload1, Payload1 + (((.LZmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    Payload2, 0x40              # 512-bit vector length
        .set    Payload2, Payload2 + ((((.LZmmIndex_\Src1Reg\() >> 4) & 1) ^ 1) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LZmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LZmmIndex_\Src2Reg\() & 7)

        .byte   0x62, Payload0, Payload1, Payload2, \Opcode\(), ModRMByte

        .endm

        .macro VpdpbusdZmmZmmZmm DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 0x50, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpbusdsZmmZmmZmm DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 0x51, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdZmmZmmZmm DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 0x52, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

        .macro VpdpwssdsZmmZmmZmm DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 0x53, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

/*++

Macro Description:

    This macro builds a VNNI instruction of the form:

         instr zmm1,zmm2,DWORD PTR [BaseReg+IndexReg*Scale+ByteOffset]{1to16}

Arguments:

    Opcode - Specifies the opcode for the VNNI instruction.

    DestReg - Specifies the destination register.

    Src1Reg - Specifies the first source register.

    BaseReg - Specifies the base register of the broadcast operand.

    ByteOffset - Specifies the DWORD aligned byte offset for the broadcast
        operand.

    IndexReg - Specifies the optional index register of the broadcast operand.

    Scale - Specifies the scaling factor of the optional index register.

--*/

        .macro VnniZmmZmmBroadcast Opcode, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LZmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
.ifnes "\IndexReg\()", ""
        .set    Payload0, Payload0 + ((((.LGprIndex_\IndexReg\() >> 3) & 1) ^ 1) << 6)
.else
        .set    Payload0, Payload0 + 0x40   # zero logical index register
.endif
        .set    Payload0, Payload0 + ((((.LGprIndex_\BaseReg\() >> 3) & 1) ^ 1) << 5)
        .set    Payload0, Payload0 + ((((.LZmmIndex_\DestReg\() >> 4) & 1) ^ 1) << 4)

        .set    Payload1, 0x05              # "66" prefix
        .set    Payload1, Payload1 + (((.LZmmIndex_\Src1Reg\() & 15) ^ 15) << 3)

        .set    Payload2, 0x50              # 512-bit vector length, broadcast
        .set    Payload2, Payload2 + ((((.LZmmIndex_\Src1Reg\() >> 4) & 1) ^ 1) << 3)

        .set    ModRMByte, 0x00             # memory form
        .set    ModRMByte, ModRMByte + ((.LZmmIndex_\DestReg\() & 7) << 3)
.ifnes "\IndexReg\()", ""
        .set    ModRMByte, ModRMByte + 0x04 # indicate SIB byte needed
.else
        .set    ModRMByte, ModRMByte + (.LGprIndex_\BaseReg\() & 7)
.endif
.if \ByteOffset\() != 0
        .set    ModRMByte, ModRMByte + 0x40 # indicate disp8 byte offset
.endif

.ifnes "\IndexReg\()", ""
        .set    SibByte, 0
.ifeqs "\Scale\()", "2"
        .set    SibByte, SibByte + (1 << 6)
.else
.ifeqs "\Scale\()", "4"
        .set    SibByte, SibByte + (2 << 6)
.else
.ifeqs "\Scale\()", "8"
        .set    SibByte, SibByte + (3 << 6)
.else
.ifnes "\Scale\()", "1"
        .err
.endif
.endif
.endif
.endif
        .set    SibByte, SibByte + ((.LGprIndex_\IndexReg\() & 7) << 3)
        .set    SibByte, SibByte + (.LGprIndex_\BaseReg\() & 7)
.endif

        .byte   0x62, Payload0, Payload1, Payload2, \Opcode\(), ModRMByte
.ifnes "\IndexReg\()", ""
        .byte   SibByte
.endif
.if \ByteOffset\() != 0
        .byte   (\ByteOffset\() >> 2)
.endif

        .endm

        .macro VpdpbusdZmmZmmBroadcast DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 0x50, \DestReg\(), \Src1Reg\(), \BaseReg\(), \ByteOffset\(), \IndexReg\(), \Scale\()

        .endm

        .macro VpdpbusdsZmmZmmBroadcast DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 0x51, \DestReg\(), \Src1Reg\(), \BaseReg\(), \ByteOffset\(), \IndexReg\(), \Scale\()

        .endm

        .macro VpdpwssdZmmZmmBroadcast DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 0x52, \DestReg\(), \Src1Reg\(), \BaseReg\(), \ByteOffset\(), \IndexReg\(), \Scale\()

        .endm

        .macro VpdpwssdsZmmZmmBroadcast DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 0x53, \DestReg\(), \Src1Reg\(), \BaseReg\(), \ByteOffset\(), \IndexReg\(), \Scale\()

        .endm
