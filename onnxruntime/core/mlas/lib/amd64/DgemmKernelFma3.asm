;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   DgemmKernelFma3.asm
;
; Abstract:
;
;   This module implements the kernels for the double precision matrix/matrix
;   multiply operation (DGEMM).
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE DgemmKernelCommon.inc
INCLUDE FgemmKernelFma3Common.inc
        .list

;
; Generate the GEMM kernel.
;

FgemmKernelFma3Function Double

        END
