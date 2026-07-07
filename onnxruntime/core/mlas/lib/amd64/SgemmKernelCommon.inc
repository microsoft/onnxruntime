;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision matrix/matrix multiply operation (SGEMM).
;
;--

;
; Define the single precision parameters.
;

FgemmElementShift       EQU     2
FgemmElementSize        EQU     (1 SHL FgemmElementShift)
FgemmElementPtr         TEXTEQU     <DWORD PTR>
FgemmElementBcst        TEXTEQU     <DWORD PTR>
FgemmElementBcstSuffix TEXTEQU <{1to16}>

;
; Define the typed instructions for single precision.
;

addpf                   EQU     addps
movupf                  EQU     movups

vaddpf                  EQU     vaddps
vbroadcastsf            EQU     vbroadcastss
vfmadd213pf             EQU     vfmadd213ps
vfmadd231pf             EQU     vfmadd231ps
vmaskmovpf              EQU     vmaskmovps
vmovapf                 EQU     vmovaps
vmovsf                  EQU     vmovss
vmovupf                 EQU     vmovups
vmulpf                  EQU     vmulps
vxorpf                  EQU     vxorps

INCLUDE FgemmKernelCommon.inc
