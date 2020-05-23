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

        .xlist
INCLUDE macamd64.inc
        .list

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
