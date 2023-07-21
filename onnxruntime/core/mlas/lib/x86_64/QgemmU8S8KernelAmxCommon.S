/*++

Copyright (c) 2023 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    AssembleAmx.h

Abstract:

    This module contains macros to build AMX instructions for toolchains that
    do not natively support this newer instruction set extension.

--*/

//
// Map friendly register names to the encoded register index.
//

	.equ    .LTmmIndex_tmm0, 0
        .equ    .LTmmIndex_tmm1, 1
        .equ    .LTmmIndex_tmm2, 2
        .equ    .LTmmIndex_tmm3, 3
        .equ    .LTmmIndex_tmm4, 4
        .equ    .LTmmIndex_tmm5, 5
        .equ    .LTmmIndex_tmm6, 6
        .equ    .LTmmIndex_tmm7, 7

/*++

Macro Description:

    This macro builds a AMX instruction of the form:

        instr tmm1,tmm2,tmm3

Arguments:

    prefix - Specifies the opcode for the AMX instruction.

    DestReg - Specifies the destination AMX tile.

    Src1Reg - Specifies the first source AMX tile.

    Src2Reg - Specifies the second source AMX tile.

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

/*++

Macro Description:

    This macro builds a AMX tile release instruction.

Arguments:



--*/

//        .macro TileReleaseMacro

//        .byte 0xC4, 0xE2, 0x78, 0x49, 0xC0

//        .endm


/*++

Macro Description:

    This macro builds an AMX tile zero instruction of the form:

        instr tmm1

Arguments:

    SrcReg - Specifies the source AMX tile.

--*/

        .macro TileZeroMacro SrcReg

        .set 	ModRMByte, 0xC0     # register form
        .set    ModRMByte, ModRMByte + ((.LTmmIndex_\SrcReg\() & 7) << 3)
        .byte   0xC4, 0xE2, 0x7B, 0x49, ModRMByte

        .endm

/*++

Macro Description:

    This macro builds an AMX memory instruction of the form:

        instr tmm, base, stride

Arguments:

    instr - Specifies the opcode for the AMX instruction.

    SrcReg - Specifies the target AMX tile.

    BaseReg - Specifies the base address of memory location.

    Stride - Specifies the stride for the memory instruction

--*/

        .macro TileLoadMacro instr, SrcReg, BaseReg, Stride

        .set    Payload0, 0x02              # "0F 38" prefix
        .set    Payload0, Payload0 + ((((.LTmmIndex_\SrcReg\() >> 3) & 1) ^ 1) << 7)
        .set    Payload0, Payload0 + ((((3 >> 3) & 1) ^ 1) << 6)
        .set    Payload0, Payload0 + ((((0 >> 3) & 1) ^ 1) << 5)

        .set 	ModRMByte, 0x00     # memory form
        .set 	ModRMByte, ModRMByte + (1 << 2)   # SibBye required
        .set 	ModRMByte, ModRMByte + ((.LTmmIndex_\SrcReg\() & 7) << 3)

        .set 	SibByte, 0x00  # scale factor 1(SS)
        .set 	SibByte, SibByte + ((3 & 7) << 3)
        .set 	SibByte, SibByte + (0 & 7)

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


/*++

Macro Description:

    This macro builds an AMX tile configuration instruction of the form:

        instr base

Arguments:

    instr - Specifies the opcode for the AMX instruction.

    BaseReg - Specifies the memory address of the tile configuration.

--*/

        .macro tilecfgMacro instr, BaseReg
	.set    Payload0, 0x02              # "0F 38" prefix
	.set    Payload0, Payload0 + (1 << 7)
	.set    Payload0, Payload0 + (1 << 6)
	.set    Payload0, Payload0 + ((((0 >> 3) & 1) ^ 1) << 5)

	.set 	ModRMByte, 0x00     # memory form & no reg
	.set 	ModRMByte, ModRMByte + (0 & 7)

	.byte 0xC4, Payload0, \instr\(), 0x49, ModRMByte

        .endm


        .macro ldtilecfgMacro BaseReg

        tilecfgMacro 0x78, \BaseReg\()

        .endm


        .macro sttilecfgMacro BaseReg
        
	tilecfgMacro 0x79, \BaseReg\()

        .endm
	
