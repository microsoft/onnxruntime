;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   mlasi.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the Microsoft
;   Machine Learning algebra subprogram library.
;
;--

;
; Dual-assembler prologue/epilogue macros.
;
; These replace the Windows SDK macamd64.inc entry and unwind macros so the
; amd64 kernels assemble with BOTH ml64.exe and the LLVM MASM assembler
; (llvm-ml), enabling a Windows x64 cross-build from a non-Windows host. Each
; macro emits the same instructions and the same built-in structured-exception
; unwind directives (.pushreg/.allocstack/.savereg/.savexmm128/.setframe/
; .endprolog/.beginepilog) that macamd64.inc emits, so the generated code and
; .pdata/.xdata are identical. The only omission is macamd64.inc's
; ifdif <Section>, _CurrentSection_ same-module section check, a diagnostic-only
; construct that llvm-ml cannot parse.
;

push_reg MACRO Reg
        push    Reg
        .pushreg Reg
        ENDM

rex_push_reg MACRO Reg
        db      048h
        push    Reg
        .pushreg Reg
        ENDM

alloc_stack MACRO Size
        sub     rsp, Size
        .allocstack Size
        ENDM

save_reg MACRO Reg, Offset
        mov     Offset[rsp], Reg
        .savereg Reg, Offset
        ENDM

save_xmm128 MACRO Reg, Offset
        movaps  XMMWORD PTR Offset[rsp], Reg
        .savexmm128 Reg, Offset
        ENDM

set_frame MACRO Reg, Offset
IF Offset
        lea     Reg, Offset[rsp]
ELSE
        mov     Reg, rsp
ENDIF
        .setframe Reg, Offset
        ENDM

END_PROLOGUE MACRO
        .endprolog
        ENDM

;
; BEGIN_EPILOGUE emits no unwind marker. ml64.exe's macamd64.inc form emits
; .beginepilog, but llvm-ml lowers that to .seh_startepilogue which then requires
; a matching .seh_endepilogue that ml64 does not emit. The amd64 kernels use only
; standard epilogs (stack restore, pops, ret), which the Windows unwinder resolves
; via legacy epilog detection without explicit epilog unwind codes, so omitting the
; marker keeps unwind correct under both assemblers.
;
BEGIN_EPILOGUE MACRO
        ENDM

LEAF_ENTRY MACRO Name, Section, NoPad
Section segment para 'CODE'
IFB <NoPad>
        db      6 dup (0cch)
ENDIF
        align   16
        public  Name
Name    proc    frame
        END_PROLOGUE
        ENDM

LEAF_END MACRO Name, Section
Name    endp
Section ends
        ENDM

NESTED_ENTRY MACRO Name, Section, Handler, NoPad
Section segment para 'CODE'
IFB <NoPad>
        db      6 dup (0cch)
ENDIF
        align   16
        public  Name
IFB <Handler>
Name    proc    frame
ELSE
Name    proc    frame:Handler
ENDIF
        ENDM

NESTED_END MACRO Name, Section
Name    endp
Section ends
        ENDM

;
; Macro Description:
;
;   This macro generates an optimization for "add reg,128" which can instead
;   be encoded as "sub reg,-128" to reduce code size by using a signed 8-bit
;   value.
;
; Arguments:
;
;   Register - Supplies the register to be added to.
;
;   Immediate - Supplies the immediate to add to the register.
;

add_immed MACRO Register, Immediate

IF (Immediate NE 128)
        add     Register,Immediate
ELSE
        sub     Register,-Immediate         ; smaller encoding
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro conditionally emits the statement if Count is greater than or
;   equal to Value.
;
; Arguments:
;
;   Count - Supplies the variable used in the comparison.
;
;   Value - Supplies the static used in the comparison.
;
;   Statement - Supplies the statement to conditionally emit.
;

EmitIfCountGE MACRO Count, Value, Statement

IF (Count GE Value)
        Statement
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro conditionally emits the statement if Count1 is greater than or
;   equal to Value1 and Count2 is greater than or equal to Value2.
;
; Arguments:
;
;   Count1 - Supplies the variable used in the comparison.
;
;   Value1 - Supplies the static used in the comparison.
;
;   Count2 - Supplies the variable used in the comparison.
;
;   Value2 - Supplies the static used in the comparison.
;
;   Statement - Supplies the statement to conditionally emit.
;

EmitIfCount2GE MACRO Count1, Value1, Count2, Value2, Statement

IF (Count1 GE Value1) AND (Count2 GE Value2)
        Statement
ENDIF

        ENDM

EmitIfCount2EQ MACRO Count1, Value1, Count2, Value2, Statement

IF (Count1 EQ Value1) AND (Count2 EQ Value2)
        Statement
ENDIF

        ENDM


;
; Macro Description:
;
;   This macro emits the statement for each register listed in the register
;   list. The statement can use RegItem to access the current register.
;
; Arguments:
;
;   RegList - Supplies the list of registers.
;
;   Statement - Supplies the statement to emit.
;

EmitForEachRegister MACRO RegList, Statement

IRP     RegItem, <RegList>
        Statement
ENDM

        ENDM
