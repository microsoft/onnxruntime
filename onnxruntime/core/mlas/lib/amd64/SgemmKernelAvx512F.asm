;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX512F instructions.
;
;--

INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
INCLUDE FgemmKernelAvx512FCommon.inc

;
; Generate the GEMM kernel.
;

FgemmKernelAvx512FFunction Float

        END
