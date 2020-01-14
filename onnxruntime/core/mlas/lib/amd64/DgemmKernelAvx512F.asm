;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   DgemmKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the double precision matrix/matrix
;   multiply operation (DGEMM).
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE DgemmKernelCommon.inc
INCLUDE FgemmKernelAvx512FCommon.inc
        .list

;
; Generate the GEMM kernel.
;

FgemmKernelAvx512FFunction Double

        END
