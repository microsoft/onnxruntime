/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

        cvtfp16a.S

Abstract:

        This module implements routines to convert between FP16 and FP32 formats using SSE2 isntructions.

--*/

#include "asmmacro.h"

// We use RIP relative addressing to avoid reallication related errors
.section .rodata
MlasFp16MaskSign: .long 0x00007FFF
MlasFp16CompareInfinity: .long 0x00007C00
MlasFp16CompareSmallest: .long 0x00000400
MlasFp16AdjustExponent: .long 0x38000000
MlasFp16MagicDenormal: .long 0x38800000

.text
.intel_syntax noprefix

/*++ Routine Description:

   This routine converts the source buffer of half-precision floats to the
   destination buffer of single-precision floats.

   This implementation uses SSE2 instructions.

 Arguments:

   Source (rdi) - Supplies the address of the source buffer of half-precision
       floats.

   Destination (rsi) - Supplies the address of the destination buffer of
       single-precision floats.

   Count (rdx) - Supplies the number of elements to convert.

 Return Value:

   None.

--*/

FUNCTION_ENTRY MlasCastF16ToF32KernelSse

        test    rdx,rdx
        jz      ExitRoutine

        // Load xmm constants
        movd    xmm5, DWORD PTR [rip + MlasFp16MaskSign]
        pshufd  xmm5, xmm5, 0x00
        movd    xmm6, DWORD PTR [rip + MlasFp16AdjustExponent]
        pshufd  xmm6, xmm6, 0x00
        movd    xmm7, DWORD PTR [rip + MlasFp16MagicDenormal]
        pshufd  xmm7, xmm7, 0x00


        cmp     rdx,4
        jb      LoadPartialVector

LoadFullVector:
        movq    xmm0,QWORD PTR [rdi]
        add     rdi,4*2                     // advance S by 4 elements

ConvertHalfToFloat:
        punpcklwd xmm0,xmm0                 // duplicate 4 WORDs to 4 DWORDs
        movaps  xmm1,xmm0                   // isolate exponent/mantissa
        pand    xmm1,xmm5
        pxor    xmm0,xmm1                   // isolate sign bit
        movd    xmm2, DWORD PTR [rip + MlasFp16CompareInfinity]
        pshufd  xmm2, xmm2, 0x00
        pcmpgtd xmm2,xmm1                   // test for infinity/NaNs
        movd    xmm3, DWORD PTR [rip + MlasFp16CompareSmallest]
        pshufd  xmm3, xmm3, 0x00
        pcmpgtd xmm3,xmm1                   // test for denormals
        pandn   xmm2,xmm6
        pslld   xmm1,13                     // shift exponent/mask into place
        movaps  xmm4,xmm1
        paddd   xmm1,xmm6
        paddd   xmm1,xmm2                   // adjust exponent again for infinity/NaNs
        paddd   xmm4,xmm7
        pslld   xmm0,16                     // shift sign into place
        subps   xmm4,xmm7
        pand    xmm4,xmm3                   // select elements that are denormals
        pandn   xmm3,xmm1                   // select elements that are not denormals
        por     xmm3,xmm4                   // blend the selected values together
        por     xmm0,xmm3                   // merge sign into exponent/mantissa

        cmp     rdx,4                        // storing full vector?
        jb      StorePartialVector
        movups  XMMWORD PTR [rsi],xmm0
        add     rsi,4*4                     // advance D by 4 elements
        sub     rdx,4
        jz      ExitRoutine
        cmp     rdx,4
        jae     LoadFullVector

LoadPartialVector:
        pxor    xmm0,xmm0
        pinsrw  xmm0,WORD PTR [rdi],0
        cmp     rdx,2
        jb      ConvertHalfToFloat
        pinsrw  xmm0,WORD PTR [rdi+2],1
        je      ConvertHalfToFloat
        pinsrw  xmm0,WORD PTR [rdi+4],2
        jmp     ConvertHalfToFloat

StorePartialVector:
        cmp     rdx,2
        jb      StoreLastElement
        movsd   QWORD PTR [rsi],xmm0
        je      ExitRoutine
        movhlps xmm0,xmm0                   // shift third element down
        add     rsi,4*2                     // advance D by 2 elements

StoreLastElement:
        movss   DWORD PTR [rsi],xmm0

ExitRoutine:
        ret
