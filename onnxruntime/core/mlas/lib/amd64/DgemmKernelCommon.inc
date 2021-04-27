;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   DgemmKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the double
;   precision matrix/matrix multiply operation (DGEMM).
;
;--

;
; Define the double precision parameters.
;

FgemmElementShift       EQU     3
FgemmElementSize        EQU     (1 SHL FgemmElementShift)
FgemmElementPtr         EQU     QWORD PTR
FgemmElementBcst        EQU     QWORD BCST

;
; Define the typed instructions for double precision.
;

addpf                   EQU     addpd
movupf                  EQU     movupd

vaddpf                  EQU     vaddpd
vbroadcastsf            EQU     vbroadcastsd
vfmadd213pf             EQU     vfmadd213pd
vfmadd231pf             EQU     vfmadd231pd
vmaskmovpf              EQU     vmaskmovpd
vmovapf                 EQU     vmovapd
vmovsf                  EQU     vmovsd
vmovupf                 EQU     vmovupd
vmulpf                  EQU     vmulpd
vxorpf                  EQU     vxorpd

INCLUDE FgemmKernelCommon.inc
