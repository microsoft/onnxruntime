;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemvU8S8KernelAvx512Core.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/vector
;   multiply operation (QGEMV).
;
;   This implementation uses AVX512 core instructions (BW/DQ/VL).
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE QgemvU8S8KernelAvx512Common.inc
        .list

;
; Generate the GEMV kernel.
;

GemvU8S8KernelAvx512Function Avx512Core

        END
