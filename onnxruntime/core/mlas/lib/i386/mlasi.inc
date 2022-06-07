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
INCLUDE callconv.inc
        .list

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
