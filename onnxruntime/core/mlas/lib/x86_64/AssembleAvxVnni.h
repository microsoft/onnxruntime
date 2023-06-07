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

	.equ    .LTmmIndex_tmm0, 0
        .equ    .LTmmIndex_tmm1, 1
        .equ    .LTmmIndex_tmm2, 2
        .equ    .LTmmIndex_tmm3, 3
        .equ    .LTmmIndex_tmm4, 4
        .equ    .LTmmIndex_tmm5, 5
        .equ    .LTmmIndex_tmm6, 6
        .equ    .LTmmIndex_tmm7, 7
        
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

/*--
 
    C4 E2 4B 5E CB  ; tdpbssd     tmm1,tmm3,tmm6
    C4 E2 7B 5E FA  ; tdpbssd     tmm7,tmm2,tmm0
    C4 E2 5A 5E DA  ; tdpbsud     tmm3,tmm2,tmm4
    C4 E2 4A 5E C7  ; tdpbsud     tmm0,tmm7,tmm6
    C4 E2 59 5E E8  ; tdpbusd     tmm5,tmm0,tmm4
    C4 E2 71 5E C3  ; tdpbusd     tmm0,tmm3,tmm1
    C4 E2 48 5E D0  ; tdpbuud     tmm2,tmm0,tmm6
    C4 E2 40 5E E3  ; tdpbuud     tmm4,tmm3,tmm7
--*/

        .macro DPTmmTmmTmm prefix, DestReg, Src1Reg, Src2Reg

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LTmmIndex_\DestReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + (1 << 6)
        .set    Payload0, Payload0 + ((((.LTmmIndex_\Src2Reg\() >> 3) & 1) ^ 1) << 5)

        .set    Payload1, \prefix\()
        .set    Payload1, Payload1 + (((.LTmmIndex_\Src2Reg\() & 15) ^ 15) << 3)

        .set    ModRMByte, 0xC0             # register form
        .set    ModRMByte, ModRMByte + ((.LTmmIndex_\DestReg\() & 7) << 3)
        .set    ModRMByte, ModRMByte + (.LTmmIndex_\Src1Reg\() & 7)

        .byte   0xC4, Payload0, Payload1, 0x5E, ModRMByte

        .endm


        .macro TdpbssdTmmTmmTmm DestReg, Src1Reg, Src2Reg

        DPTmmTmmTmm 0x03, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm


        .macro TdpbsudTmmTmmTmm DestReg, Src1Reg, Src2Reg

        DPTmmTmmTmm 0x02, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm


        .macro TdpbusdTmmTmmTmm DestReg, Src1Reg, Src2Reg

        DPTmmTmmTmm 0x01, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm


        .macro TdpbuudTmmTmmTmm DestReg, Src1Reg, Src2Reg

        DPTmmTmmTmm 0x00, \DestReg\(), \Src1Reg\(), \Src2Reg\()

        .endm

/*

    tilerelease ; C4 E2 78 49 C0

*/

        .macro TileReleaseMacro

        .byte 0xC4, 0xE2, 0x78, 0x49, 0xC0

        .endm

/*
    tilezero tmm5   ; C4 E2 7B 49 E8
    tilezero tmm3   ; C4 E2 7B 49 D8
*/

        .macro TileZeroMacro SrcReg

        .set ModRMByte, 0xC0     # register form
        .set    ModRMByte, ModRMByte + ((.LTmmIndex_\SrcReg\() & 7) << 3)
        .byte   0xC4, 0xE2, 0x7B, 0x49, ModRMByte

        .endm


/*
 tileloadd tmm1, [rcx+rdx]          ; C4 E2 7B 4B 0C 11    
 tileloaddt1 tmm0, [r11+rbx]        ; C4 C2 79 4B 04 1B
 tilestored [rsi+rdi], tmm6         ; C4 E2 7A 4B 34 3E
*/

        .macro TileLoadMacro instr, SrcReg, BaseReg, Stride

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LTmmIndex_\SrcReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + ((((.LGprIndex_\Stride\() >> 3) & 1) ^ 1) << 6)
        .set    Payload0, Payload0 + ((((.LGprIndex_\BaseReg\() >> 3) & 1) ^ 1) << 5)

        .set ModRMByte, 0x00     # memory form
        .set ModRMByte, ModRMByte + (1 << 2)   # SibBye required
        .set ModRMByte, ModRMByte + ((.LTmmIndex_\SrcReg\() & 7) << 3)

        .set SibByte, 0x00  # scale factor 1(SS)
        .set SibByte, SibByte + ((.LGprIndex_\Stride\() & 7) << 3)
        .set SibByte, SibByte + (.LGprIndex_\BaseReg\() & 7)

        .byte   0xC4, Payload0, \instr\(), 0x4B, ModRMByte, SibByte

        .endm


        .macro TileloaddTmmMem DstReg, BaseReg, Stride
        TileLoadMacro 0x7B, \DstReg\(), \BaseReg\(), \Stride\()
        .endm





        .macro TileloaddT1TmmMem DstReg, BaseReg, Stride
        TileLoadMacro 0x79, \DstReg\(), \BaseReg\(), \Stride\()
        .endm


        .macro TileStoredMemTmm SrcReg, BaseReg, Stride
        TileLoadMacro 0x7A, \SrcReg\(), \BaseReg\(), \Stride\()
        .endm

        .macro tilecfgMacro instr, BaseReg
	.set    Payload0, 0x02              # "0F 38" prefix
	.set    Payload0, Payload0 + (1 << 7)
	.set    Payload0, Payload0 + (1 << 6)
#.set    Payload0, Payload0 + ((((.LGprIndex_\BaseReg\() >> 3) & 1) ^ 1) << 5)
        .set    Payload0, Payload0 + ((((7 >> 3) & 1) ^ 1) << 5)

	.set ModRMByte, 0x00     # memory form & no reg
#.set ModRMByte, ModRMByte + (.LGprIndex_\BaseReg\() & 7)
	.set ModRMByte, ModRMByte + (7 & 7)

	.byte 0xC4, Payload0, \instr\(), 0x49, ModRMByte

        .endm


        .macro ldtilecfgMacro BaseReg

        tilecfgMacro 0x78, BaseReg

        .endm


        .macro sttilecfgMacro BaseReg
        
	tilecfgMacro 0x79, BaseReg

        .endm
	
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
