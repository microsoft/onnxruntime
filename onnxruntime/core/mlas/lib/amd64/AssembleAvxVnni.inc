;++
;
; Copyright (c) 2020 Intel Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   AssembleAvxVnni.inc
;
; Abstract:
;
;   This module contains macros to build AVXVNNI instructions for toolchains that
;   do not natively support this newer instruction set extension.
;
;--

;
; Map friendly register names to the encoded register index.
;

YmmIndex_ymm0   EQU     0
YmmIndex_ymm1   EQU     1
YmmIndex_ymm2   EQU     2
YmmIndex_ymm3   EQU     3
YmmIndex_ymm4   EQU     4
YmmIndex_ymm5   EQU     5
YmmIndex_ymm6   EQU     6
YmmIndex_ymm7   EQU     7
YmmIndex_ymm8   EQU     8
YmmIndex_ymm9   EQU     9
YmmIndex_ymm10  EQU     10
YmmIndex_ymm11  EQU     11
YmmIndex_ymm12  EQU     12
YmmIndex_ymm13  EQU     13
YmmIndex_ymm14  EQU     14
YmmIndex_ymm15  EQU     15

XmmIndex_xmm0   EQU     0
XmmIndex_xmm1   EQU     1
XmmIndex_xmm2   EQU     2
XmmIndex_xmm3   EQU     3
XmmIndex_xmm4   EQU     4
XmmIndex_xmm5   EQU     5
XmmIndex_xmm6   EQU     6
XmmIndex_xmm7   EQU     7
XmmIndex_xmm8   EQU     8
XmmIndex_xmm9   EQU     9
XmmIndex_xmm10  EQU     10
XmmIndex_xmm11  EQU     11
XmmIndex_xmm12  EQU     12
XmmIndex_xmm13  EQU     13
XmmIndex_xmm14  EQU     14
XmmIndex_xmm15  EQU     15

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;       instr ymm1,ymm2,ymm3
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

VnniYmmYmmYmm MACRO Opcode, DestReg, Src1Reg, Src2Reg

        LOCAL   Payload0, Payload1, ModRMByte

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((YmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
        Payload0 = Payload0 + (1 SHL 6)
        Payload0 = Payload0 + ((((YmmIndex_&Src2Reg& SHR 3) AND 1) XOR 1) SHL 5)

        Payload1 = 005h                     ; "66" prefix
        Payload1 = Payload1 + (((YmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        ModRMByte = 0C0h                    ; register form
        ModRMByte = ModRMByte + ((YmmIndex_&DestReg& AND 7) SHL 3)
        ModRMByte = ModRMByte + (YmmIndex_&Src2Reg& AND 7)

        db      0C4h, Payload0, Payload1, Opcode, ModRMByte

        ENDM

VpdpbusdYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 050h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbusdsYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 051h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 052h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdsYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        VnniYmmYmmYmm 053h, DestReg, Src1Reg, Src2Reg

        ENDM

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;       instr xmm1,xmm2,xmm3
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

VnniXmmXmmXmm MACRO Opcode, DestReg, Src1Reg, Src2Reg

        LOCAL   Payload0, Payload1, ModRMByte

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((XmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
        Payload0 = Payload0 + (1 SHL 6)
        Payload0 = Payload0 + ((((XmmIndex_&Src2Reg& SHR 3) AND 1) XOR 1) SHL 5)

        Payload1 = 001h                     ; "66" prefix
        Payload1 = Payload1 + (((XmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        ModRMByte = 0C0h                    ; register form
        ModRMByte = ModRMByte + ((XmmIndex_&DestReg& AND 7) SHL 3)
        ModRMByte = ModRMByte + (XmmIndex_&Src2Reg& AND 7)

        db      0C4h, Payload0, Payload1, Opcode, ModRMByte

        ENDM

VpdpbusdXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 050h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbusdsXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 051h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 052h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpwssdsXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        VnniXmmXmmXmm 053h, DestReg, Src1Reg, Src2Reg

        ENDM

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;       instr ymm1,ymm2,ymm3
;
; Arguments:
;
;   Opcode - Specifies the opcode for the VNNI instruction.
;
;   Prefix - Specifies the opcode prefix for payload 1
;
;   DestReg - Specifies the destination register.
;
;   Src1Reg - Specifies the first source register.
;
;   Src2Reg - Specifies the second source register.
;

Avx2VnniYmmYmmYmm MACRO Opcode, Prefix, DestReg, Src1Reg, Src2Reg

        LOCAL   Payload0, Payload1, ModRMByte

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((YmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
        Payload0 = Payload0 + (1 SHL 6)
        Payload0 = Payload0 + ((((YmmIndex_&Src2Reg& SHR 3) AND 1) XOR 1) SHL 5)

        Payload1 = 004h + Prefix            ; 256-bit length and opcode prefix
        Payload1 = Payload1 + (((YmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        ModRMByte = 0C0h                    ; register form
        ModRMByte = ModRMByte + ((YmmIndex_&DestReg& AND 7) SHL 3)
        ModRMByte = ModRMByte + (YmmIndex_&Src2Reg& AND 7)

        db      0C4h, Payload0, Payload1, Opcode, ModRMByte

        ENDM

VpdpbssdYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 050h, 003h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbssdsYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 051h, 003h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbsudYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 050h, 002h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbsudsYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 051h, 002h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbuudYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 050h, 000h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbuudsYmmYmmYmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniYmmYmmYmm 051h, 000h, DestReg, Src1Reg, Src2Reg

        ENDM

;
; Macro Description:
;
;   This macro builds a VNNI instruction of the form:
;
;       instr xmm1,xmm2,xmm3
;
; Arguments:
;
;   Opcode - Specifies the opcode for the VNNI instruction.
;
;   Prefix - Specifies the opcode prefix for payload 1
;
;   DestReg - Specifies the destination register.
;
;   Src1Reg - Specifies the first source register.
;
;   Src2Reg - Specifies the second source register.
;

Avx2VnniXmmXmmXmm MACRO Opcode, Prefix, DestReg, Src1Reg, Src2Reg

        LOCAL   Payload0, Payload1, ModRMByte

        Payload0 = 002h                     ; "0F 38" prefix
        Payload0 = Payload0 + ((((XmmIndex_&DestReg& SHR 3) AND 1) XOR 1) SHL 7)
        Payload0 = Payload0 + (1 SHL 6)
        Payload0 = Payload0 + ((((XmmIndex_&Src2Reg& SHR 3) AND 1) XOR 1) SHL 5)

        Payload1 = 000h + Prefix            ; 128-bit length and opcode prefix
        Payload1 = Payload1 + (((XmmIndex_&Src1Reg& AND 15) XOR 15) SHL 3)

        ModRMByte = 0C0h                    ; register form
        ModRMByte = ModRMByte + ((XmmIndex_&DestReg& AND 7) SHL 3)
        ModRMByte = ModRMByte + (XmmIndex_&Src2Reg& AND 7)

        db      0C4h, Payload0, Payload1, Opcode, ModRMByte

        ENDM

VpdpbssdXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 050h, 003h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbssdsXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 051h, 003h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbsudXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 050h, 002h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbsudsXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 051h, 002h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbuudXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 050h, 000h, DestReg, Src1Reg, Src2Reg

        ENDM

VpdpbuudsXmmXmmXmm MACRO DestReg, Src1Reg, Src2Reg

        Avx2VnniXmmXmmXmm 051h, 000h, DestReg, Src1Reg, Src2Reg

        ENDM
