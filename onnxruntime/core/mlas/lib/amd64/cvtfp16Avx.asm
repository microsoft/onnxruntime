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

        ALIGN   16
; Legacy implementation constants
MlasFp16MaskSign                DD      4 DUP (00007FFFh)
MlasFp16CompareInfinity         DD      4 DUP (00007C00h)
MlasFp16CompareSmallest         DD      4 DUP (00000400h)
MlasFp16AdjustExponent          DD      4 DUP (38000000h)
MlasFp16MagicDenormal           DD      4 DUP (38800000h)
; AVX implementation constants
SINGLE_SIZE     equ 4
HALF_SIZE       equ 2
LOW_SELECTOR    equ 00100000b
HIGH_SELECTOR   equ 00110001b


        SUBTTL  "Convert buffer of half-precision floats to single-precision floats"
;++
;
; Routine Description:
;
;   This routine calls the implementation of the cast operator depending on the ISA flag.
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
;   ISA flag (r9) - Determines whether to use AVX_NE_CONVERT or not.
;
; Return Value:
;
;   None.
;
;--


LEAF_ENTRY MlasConvertHalfToFloatBuffer, _TEXT

	test    r8, r8      ; Check if we have any elements to convert
        jz      ExitRoutine
        test    r9, r9      ; Check if we need to use AVX_NE_CONVERT
        jz      SSE

AVX_NE_CONVERT:
        cmp     r8, 8
        jb      ConvertMaskedVectors
        cmp     r8, 16
        jb      Convert128Vectors



Convert256Vectors:
        vcvtneeph2ps    ymm0, ymmword PTR [rcx]                 ; Load even indexes
        vcvtneoph2ps    ymm1, ymmword PTR [rcx]                 ; Load odd indexes
        vunpcklps       ymm2, ymm0, ymm1                        ; Interleave low part
        vunpckhps       ymm1, ymm0, ymm1                        ; Interleave high part
        vperm2f128      ymm0, ymm2, ymm1, LOW_SELECTOR   	; Fix the order
        vperm2f128      ymm1, ymm2, ymm1, HIGH_SELECTOR   	; Fix the order
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
        jmp ExitRoutine ; If we ran into any of the cases above, means we are done after storing
ConvertLowerVector:
        vmovups xmmword PTR [rdx], xmm0     ; Store the low part
        sub     r8, 4   ; Check if we still need to convert
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

        jmp ExitRoutine



SSE:
        cmp     r8,4
        jb      LoadPartialVector

LoadFullVector:
        movq    xmm0,QWORD PTR [rcx]
        add     rcx,4*2                     ; advance S by 4 elements

ConvertHalfToFloat:
        punpcklwd xmm0,xmm0                 ; duplicate 4 WORDs to 4 DWORDs
        movaps  xmm1,xmm0                   ; isolate exponent/mantissa
        pand    xmm1,XMMWORD PTR [MlasFp16MaskSign]
        pxor    xmm0,xmm1                   ; isolate sign bit
        movaps  xmm2,XMMWORD PTR [MlasFp16CompareInfinity]
        pcmpgtd xmm2,xmm1                   ; test for infinity/NaNs
        movaps  xmm3,XMMWORD PTR [MlasFp16CompareSmallest]
        pcmpgtd xmm3,xmm1                   ; test for denormals
        pandn   xmm2,XMMWORD PTR [MlasFp16AdjustExponent]
        pslld   xmm1,13                     ; shift exponent/mask into place
        movaps  xmm4,xmm1
        paddd   xmm1,XMMWORD PTR [MlasFp16AdjustExponent]
        paddd   xmm1,xmm2                   ; adjust exponent again for infinity/NaNs
        paddd   xmm4,XMMWORD PTR [MlasFp16MagicDenormal]
        pslld   xmm0,16                     ; shift sign into place
        subps   xmm4,XMMWORD PTR [MlasFp16MagicDenormal]
        pand    xmm4,xmm3                   ; select elements that are denormals
        pandn   xmm3,xmm1                   ; select elements that are not denormals
        por     xmm3,xmm4                   ; blend the selected values together
        por     xmm0,xmm3                   ; merge sign into exponent/mantissa

        cmp     r8,4                        ; storing full vector?
        jb      StorePartialVector
        movups  XMMWORD PTR [rdx],xmm0
        add     rdx,4*4                     ; advance D by 4 elements
        sub     r8,4
        jz      ExitRoutine
        cmp     r8,4
        jae     LoadFullVector

LoadPartialVector:
        pxor    xmm0,xmm0
        pinsrw  xmm0,WORD PTR [rcx],0
        cmp     r8,2
        jb      ConvertHalfToFloat
        pinsrw  xmm0,WORD PTR [rcx+2],1
        je      ConvertHalfToFloat
        pinsrw  xmm0,WORD PTR [rcx+4],2
        jmp     ConvertHalfToFloat

StorePartialVector:
        cmp     r8,2
        jb      StoreLastElement
        movsd   QWORD PTR [rdx],xmm0
        je      ExitRoutine
        movhlps xmm0,xmm0                   ; shift third element down
        add     rdx,4*2                     ; advance D by 2 elements

StoreLastElement:
        movss   DWORD PTR [rdx],xmm0

ExitRoutine:
        ret

        LEAF_END MlasConvertHalfToFloatBuffer, _TEXT

        END
