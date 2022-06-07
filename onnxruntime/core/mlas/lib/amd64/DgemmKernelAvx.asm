;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   DgemmKernelAvx.asm
;
; Abstract:
;
;   This module implements the kernels for the double precision matrix/matrix
;   multiply operation (DGEMM).
;
;   This implementation uses AVX instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE DgemmKernelCommon.inc
INCLUDE FgemmKernelAvxCommon.inc
        .list

;
; Generate the GEMM kernel.
;

FgemmKernelAvxFunction Double

        END
