;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   AssembleAvx512Vnni.inc
;
; Abstract:
;
;   This module contains macros to build VNNI instructions for toolchains that
;   do not natively support this newer instruction set extension.
;
;--

;
; Map friendly register names to the encoded register index.
;

ZmmIndex_zmm0   EQU     0
ZmmIndex_zmm1   EQU     1
ZmmIndex_zmm2   EQU     2
ZmmIndex_zmm3   EQU     3
ZmmIndex_zmm4   EQU     4
ZmmIndex_zmm5   EQU     5
ZmmIndex_zmm6   EQU     6
ZmmIndex_zmm7   EQU     7
ZmmIndex_zmm8   EQU     8
ZmmIndex_zmm9   EQU     9
ZmmIndex_zmm10  EQU     10
ZmmIndex_zmm11  EQU     11
ZmmIndex_zmm12  EQU     12
ZmmIndex_zmm13  EQU     13
ZmmIndex_zmm14  EQU     14
ZmmIndex_zmm15  EQU     15
ZmmIndex_zmm16  EQU     16
ZmmIndex_zmm17  EQU     17
ZmmIndex_zmm18  EQU     18
ZmmIndex_zmm19  EQU     19
ZmmIndex_zmm20  EQU     20
ZmmIndex_zmm21  EQU     21
ZmmIndex_zmm22  EQU     22
ZmmIndex_zmm23  EQU     23
ZmmIndex_zmm24  EQU     24
ZmmIndex_zmm25  EQU     25
ZmmIndex_zmm26  EQU     26
ZmmIndex_zmm27  EQU     27
ZmmIndex_zmm28  EQU     28
ZmmIndex_zmm29  EQU     29
ZmmIndex_zmm30  EQU     30
ZmmIndex_zmm31  EQU     31

GprIndex_rax    EQU     0
GprIndex_rcx    EQU     1
GprIndex_rdx    EQU     2
GprIndex_rbx    EQU     3
GprIndex_rbp    EQU     5
GprIndex_rsi    EQU     6
GprIndex_rdi    EQU     7
GprIndex_r8     EQU     8
GprIndex_r9     EQU     9
GprIndex_r10    EQU     10
GprIndex_r11    EQU     11
GprIndex_r12    EQU     12
GprIndex_r13    EQU     13
GprIndex_r14    EQU     14
GprIndex_r15    EQU     15

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;       instr zmm1,zmm2,zmm3
;
; Arguments:
;
;   Opcode - Specifies the opcode for the VNNI instruction.
;
;   DestReg - Specifies the destination register.
;
;   Src1Reg - Specifies the first source register.
;
;   Src2Reg - Specifies the second source register.
;

VnniZmmZmmZmm MACRO Opcode, DestReg, Src1Reg, Src2Reg

        LOCAL   Payload0, Payload1, Payload2, ModRMByte

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((ZmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
        Payload0 = Payload0 + ((((ZmmIndex_&Src2Reg& SHR 4) AND 1) XOR 1) SHL 6)
        Payload0 = Payload0 + ((((ZmmIndex_&Src2Reg& SHR 3) AND 1) XOR 1) SHL 5)
        Payload0 = Payload0 + ((((ZmmIndex_&DestReg& SHR 4) AND 1) XOR 1) SHL 4)

        Payload1 = 005h                     ; "66" prefix
        Payload1 = Payload1 + (((ZmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        Payload2 = 040h                     ; 512-bit vector length
        Payload2 = Payload2 + ((((ZmmIndex_&Src1Reg& SHR 4) AND 1) XOR 1) SHL 3)

        ModRMByte = 0C0h                    ; register form
        ModRMByte = ModRMByte + ((ZmmIndex_&DestReg& AND 7) SHL 3)
        ModRMByte = ModRMByte + (ZmmIndex_&Src2Reg& AND 7)

        db      062h, Payload0, Payload1, Payload2, Opcode, ModRMByte

        ENDM

VpdpbusdZmmZmmZmm MACRO DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 050h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbusdsZmmZmmZmm MACRO DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 051h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdZmmZmmZmm MACRO DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 052h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdsZmmZmmZmm MACRO DestReg, Src1Reg, Src2Reg

        VnniZmmZmmZmm 053h, DestReg, Src1Reg, Src2Reg

        ENDM

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;        instr zmm1,zmm2,DWORD BCST [BaseReg+IndexReg*Scale+ByteOffset]
;
; Arguments:
;
;   Opcode - Specifies the opcode for the VNNI instruction.
;
;   DestReg - Specifies the destination register.
;
;   Src1Reg - Specifies the first source register.
;
;   BaseReg - Specifies the base register of the broadcast operand.
;
;   ByteOffset - Specifies the DWORD aligned byte offset for the broadcast
;       operand.
;
;   IndexReg - Specifies the optional index register of the broadcast operand.
;
;   Scale - Specifies the scaling factor of the optional index register.
;

VnniZmmZmmBroadcast MACRO Opcode, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        LOCAL   Payload0, Payload1, Payload2, ModRMByte, SibByte

.errnz (ByteOffset AND 3)

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((ZmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
IFNB <IndexReg>
        Payload0 = Payload0 + ((((GprIndex_&IndexReg& SHR 3) AND 1) XOR 1) SHL 6)
ELSE
        Payload0 = Payload0 + 040h          ; zero logical index register
ENDIF
        Payload0 = Payload0 + ((((GprIndex_&BaseReg& SHR 3) AND 1) XOR 1) SHL 5)
        Payload0 = Payload0 + ((((ZmmIndex_&DestReg& SHR 4) AND 1) XOR 1) SHL 4)

        Payload1 = 005h                     ; "66" prefix
        Payload1 = Payload1 + (((ZmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        Payload2 = 050h                     ; 512-bit vector length, broadcast
        Payload2 = Payload2 + ((((ZmmIndex_&Src1Reg& SHR 4) AND 1) XOR 1) SHL 3)

        ModRMByte = 000h                    ; memory form
        ModRMByte = ModRMByte + ((ZmmIndex_&DestReg& AND 7) SHL 3)
IFNB <IndexReg>
        ModRMByte = ModRMByte + 004h        ; indicate SIB byte needed
ELSE
        ModRMByte = ModRMByte + (GprIndex_&BaseReg& AND 7)
ENDIF
IF ByteOffset NE 0
        ModRMByte = ModRMByte + 040h        ; indicate disp8 byte offset
ENDIF

IFNB <IndexReg>
        SibByte = 0
IF Scale EQ 2
        SibByte = SibByte + (1 SHL 6)
ELSEIF Scale EQ 4
        SibByte = SibByte + (2 SHL 6)
ELSEIF Scale EQ 8
        SibByte = SibByte + (3 SHL 6)
ELSEIF Scale NE 1
        .err <invalid index factor>
ENDIF
        SibByte = SibByte + ((GprIndex_&IndexReg& AND 7) SHL 3)
        SibByte = SibByte + (GprIndex_&BaseReg& AND 7)
ENDIF

        db      062h, Payload0, Payload1, Payload2, Opcode, ModRMByte
IFNB <IndexReg>
        db      SibByte
ENDIF
IF ByteOffset NE 0
        db      ByteOffset SHR 2
ENDIF

        ENDM

VpdpbusdZmmZmmBroadcast MACRO DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 050h, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        ENDM

VpdpbusdsZmmZmmBroadcast MACRO DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 051h, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        ENDM

VpdpwssdZmmZmmBroadcast MACRO DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 052h, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        ENDM

VpdpwssdsZmmZmmBroadcast MACRO DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        VnniZmmZmmBroadcast 053h, DestReg, Src1Reg, BaseReg, ByteOffset, IndexReg, Scale

        ENDM
