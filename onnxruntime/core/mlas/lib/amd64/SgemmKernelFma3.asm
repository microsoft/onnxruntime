;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelFma3.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
INCLUDE FgemmKernelFma3Common.inc

;
; Generate the GEMM kernel.
;

FgemmKernelFma3Function Float

        END
