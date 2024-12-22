;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   cvtfp16a.asm
;
; Abstract:
;
;   This module implements routines to convert between FP16 and FP32 formats.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        .const

        ALIGN   16
MlasFp16MaskSign                DD      4 DUP (00007FFFh)
MlasFp16CompareInfinity         DD      4 DUP (00007C00h)
MlasFp16CompareSmallest         DD      4 DUP (00000400h)
MlasFp16AdjustExponent          DD      4 DUP (38000000h)
MlasFp16MagicDenormal           DD      4 DUP (38800000h)

        SUBTTL  "Convert buffer of half-precision floats to single-precision floats"
;++
;
; Routine Description:
;
;   This routine converts the source buffer of half-precision floats to the
;   destination buffer of single-precision floats.
;
;   This implementation uses SSE2 instructions.
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

        LEAF_ENTRY MlasCastF16ToF32KernelSse, _TEXT

        test    r8,r8
        jz      ExitRoutine
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

        LEAF_END MlasCastF16ToF32KernelSse, _TEXT

        END
