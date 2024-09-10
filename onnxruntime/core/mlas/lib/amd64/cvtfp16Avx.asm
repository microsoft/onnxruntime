;++
;
; Copyright (c) Intel Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   cvtfp16Avx2.asm
;
; Abstract:
;
;   This module implements routines to convert between FP16 and FP32 formats using the AVX_NE_CONVERT ISA.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        .const

SINGLE_SIZE     equ 4
HALF_SIZE       equ 2
LOW_SELECTOR    equ 00100000b
HIGH_SELECTOR   equ 00110001b

        SUBTTL  "Convert buffer of half-precision floats to single-precision floats"
;++
;
; Routine Description:
;
;   This routine converts the source buffer of half-precision floats to the
;   destination buffer of single-precision floats.
;
;   This implementation uses AVX2 instructions.
;
; Arguments:
;
;   Source (rcx) - Supplies the address of the source buffer of half-precision
;       floats.
;
;   Destination (rdx) - Supplies the address of the destination buffer of
;       single-precision floats.
;
;   Count (r8) - Supplies the number of elements to convert.
;
; Return Value:
;
;   None.
;
;--


LEAF_ENTRY MlasCastF16ToF32KernelAvx, _TEXT

	test    r8, r8                  ; Check if we have any elements to convert
    jz      ExitRoutine
    cmp     r8, 8
    jb      ConvertMaskedVectors
    cmp     r8, 16
    jb      Convert128Vectors



Convert256Vectors:
        vcvtneeph2ps    ymm0, ymmword PTR [rcx]                 ; Load even indexes
        vcvtneoph2ps    ymm1, ymmword PTR [rcx]                 ; Load odd indexes
        vunpcklps       ymm2, ymm0, ymm1                        ; Interleave low part
        vunpckhps       ymm1, ymm0, ymm1                        ; Interleave high part
        vperm2f128      ymm0, ymm2, ymm1, LOW_SELECTOR   	    ; Fix the order
        vperm2f128      ymm1, ymm2, ymm1, HIGH_SELECTOR   	    ; Fix the order
        vmovups         ymmword PTR [rdx], ymm0                 ; Store the low part
        vmovups         ymmword PTR [rdx + 8*SINGLE_SIZE], ymm1 ; Store the high part

        add     rcx, 16*HALF_SIZE   ; Advance src ptr by 16 elements
        add     rdx, 16*SINGLE_SIZE ; Advance dest ptr by 16 elements
        sub     r8, 16              ; Reduce the counter by 16 elements

        jz      ExitRoutine ; If we are done, exit
        cmp     r8, 16      ; If the vector is big enough, we go again
        jae     Convert256Vectors



Convert128Vectors:
        vcvtneeph2ps    xmm2, xmmword PTR [rcx]                 ; Load even indexes
        vcvtneoph2ps    xmm1, xmmword PTR [rcx]                 ; Load odd indexes
        vunpcklps       xmm0, xmm2, xmm1                        ; Interleave low part to fix order
        vunpckhps       xmm1, xmm2, xmm1                        ; Interleave high part to fix order
        vmovups         xmmword PTR [rdx], xmm0                 ; Store the low part
        vmovups         xmmword PTR [rdx + 4*SINGLE_SIZE], xmm1 ; Store the high part

        add     rcx, 8*HALF_SIZE    ; Advance src ptr by 8 elements
        add     rdx, 8*SINGLE_SIZE  ; Advance dest ptr by 8 elements
        sub     r8, 8               ; Reduce the counter by 8 elements

        jz      ExitRoutine ; If we are done, exit



ConvertMaskedVectors:
        vcvtneeph2ps    xmm2, xmmword PTR [rcx]         ; Load even indexes
        vcvtneoph2ps    xmm1, xmmword PTR [rcx]         ; Load odd indexes
        vunpcklps       xmm0, xmm2, xmm1                ; Interleave low part to fix order
        vunpckhps       xmm1, xmm2, xmm1                ; Interleave high part to fix order

        cmp     r8, 4   ; Check if we can store the complete lower vector
        jae     ConvertLowerVector

        vpcmpeqw    xmm2, xmm2, xmm2                ; Initialize the mask full of ones
        cmp         r8, 2                           ; Check how many converts we need
        jb          ConvertLower1
        ja          ConvertLower3
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*2       ; Shift the memory store two values
        jmp         ConvertLowerMaskedVector
ConvertLower1:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*3       ; Shift the memory store only one value
        jmp         ConvertLowerMaskedVector
ConvertLower3:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE         ; Shift the memory store three values
ConvertLowerMaskedVector:
        vmaskmovps  xmmword PTR [rdx], xmm2, xmm0   ; Store the masked data, the shift is done in 8bit multiples
        jmp ExitRoutine                             ; If we ran into any of the cases above, means we are done after storing
ConvertLowerVector:
        vmovups xmmword PTR [rdx], xmm0             ; Store the low part
        sub     r8, 4                               ; Check if we still need to convert
        jz      ExitRoutine


        add         rdx, 4*SINGLE_SIZE              ; Advance dest ptr by 4 elements
        vpcmpeqw    xmm2, xmm2, xmm2                ; Initialize the mask full of ones
        cmp         r8, 2                           ; Check how many converts we need
        jb          ConvertUpper1
        ja          ConvertUpper3
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*2       ; Shift the memory store two values
        jmp         ConvertMaskedUpperVector
ConvertUpper1:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*3       ; Shift the memory store only one value
        jmp         ConvertMaskedUpperVector
ConvertUpper3:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE         ; Shift the memory store three values
ConvertMaskedUpperVector:
        vmaskmovps  xmmword PTR [rdx], xmm2, xmm1   ; Store the masked data, the shift is done in 8bit multiples

ExitRoutine:
        ret

        LEAF_END MlasCastF16ToF32KernelAvx, _TEXT

        END
