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
; save_xmm128_avx
;
; Macro Description:
;
;   This macro uses AVX instructions to save a vector register as part of a
;   function prologue as an alternative to save_xmm128.
;
; Arguments:
;
;   Register - Supplies the vector register to be saved.
;
;   Offset - Supplies the stack frame offset to store the contents of the
;       vector register.
;

save_xmm128_avx MACRO Register, Offset

        vmovaps Offset[rsp], Register
        .savexmm128 Register, Offset

        ENDM
