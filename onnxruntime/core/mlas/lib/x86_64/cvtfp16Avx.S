/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

        cvtfp16Avx2.asm

Abstract:

        This module implements routines to convert between FP16 and FP32 formats using the AVX_NE_CONVERT ISA.

--*/

#include "asmmacro.h"

.data
.equ SINGLE_SIZE, 4
.equ HALF_SIZE, 2
.equ LOW_SELECTOR, 0b00100000
.equ HIGH_SELECTOR, 0b00110001

.text
.intel_syntax noprefix

/*++ Routine Description:

   This routine converts the source buffer of half-precision floats to the
   destination buffer of single-precision floats.

   This implementation uses AVX2 instructions.

 Arguments:

   Source (rdi) - Supplies the address of the source buffer of half-precision
       floats.

   Destination (rsi) - Supplies the address of the destination buffer of
       single-precision floats.

   Count (rdx) - Supplies the number of elements to convert.

 Return Value:

   None.

--*/
FUNCTION_ENTRY MlasCastF16ToF32KernelAvx

        test    rdx, rdx      // Check if we have any elements to convert
        jz      ExitRoutine

AVX_NE_CONVERT:
        cmp     rdx, 8
        jb      ConvertMaskedVectors
        cmp     rdx, 16
        jb      Convert128Vectors

Convert256Vectors:
        vcvtneeph2ps    ymm0, ymmword PTR [rdi]                 // Load even indexes
        vcvtneoph2ps    ymm1, ymmword PTR [rdi]                 // Load odd indexes
        vunpcklps       ymm2, ymm0, ymm1                        // Interleave low part
        vunpckhps       ymm1, ymm0, ymm1                        // Interleave high part
        vperm2f128      ymm0, ymm2, ymm1, LOW_SELECTOR   	// Fix the order
        vperm2f128      ymm1, ymm2, ymm1, HIGH_SELECTOR   	// Fix the order
        vmovups         ymmword PTR [rsi], ymm0                 // Store the low part
        vmovups         ymmword PTR [rsi + 8*SINGLE_SIZE], ymm1 // Store the high part

        add     rdi, 16*HALF_SIZE       // Advance src ptr by 16 elements
        add     rsi, 16*SINGLE_SIZE     // Advance dest ptr by 16 elements
        sub     rdx, 16                 // Reduce the counter by 16 elements

        jz      ExitRoutine     // If we are done, exit
        cmp     rdx, 16         // If the vector is big enough, we go again
        jae     Convert256Vectors



Convert128Vectors:
        vcvtneeph2ps    xmm2, xmmword PTR [rdi]                 // Load even indexes
        vcvtneoph2ps    xmm1, xmmword PTR [rdi]                 // Load odd indexes
        vunpcklps       xmm0, xmm2, xmm1                        // Interleave low part to fix order
        vunpckhps       xmm1, xmm2, xmm1                        // Interleave high part to fix order
        vmovups         xmmword PTR [rsi], xmm0                 // Store the low part
        vmovups         xmmword PTR [rsi + 4*SINGLE_SIZE], xmm1 // Store the high part

        add     rdi, 8*HALF_SIZE    // Advance src ptr by 8 elements
        add     rsi, 8*SINGLE_SIZE  // Advance dest ptr by 8 elements
        sub     rdx, 8              // Reduce the counter by 8 elements

        jz      ExitRoutine // If we are done, exit



ConvertMaskedVectors:
        vcvtneeph2ps    xmm2, xmmword PTR [rdi]         // Load even indexes
        vcvtneoph2ps    xmm1, xmmword PTR [rdi]         // Load odd indexes
        vunpcklps       xmm0, xmm2, xmm1                // Interleave low part to fix order
        vunpckhps       xmm1, xmm2, xmm1                // Interleave high part to fix order

        cmp     rdx, 4   // Check if we can store the complete lower vector
        jae     ConvertLowerVector

        vpcmpeqw    xmm2, xmm2, xmm2                // Initialize the mask full of ones
        cmp         rdx, 2                          // Check how many converts we need
        jb          ConvertLower1
        ja          ConvertLower3
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*2       // Shift the memory store two values
        jmp         ConvertLowerMaskedVector
ConvertLower1:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*3       // Shift the memory store only one value
        jmp         ConvertLowerMaskedVector
ConvertLower3:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE         // Shift the memory store three values
ConvertLowerMaskedVector:
        vmaskmovps  xmmword PTR [rsi], xmm2, xmm0   // Store the masked data, the shift is done in 8bit multiples
        jmp ExitRoutine // If we ran into any of the cases above, means we are done after storing
ConvertLowerVector:
        vmovups xmmword PTR [rsi], xmm0     // Store the low part
        sub     rdx, 4                      // Check if we still need to convert
        jz      ExitRoutine


        add         rsi, 4*SINGLE_SIZE              // Advance dest ptr by 4 elements
        vpcmpeqw    xmm2, xmm2, xmm2                // Initialize the mask full of ones
        cmp         rdx, 2                          // Check how many converts we need
        jb          ConvertUpper1
        ja          ConvertUpper3
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*2       // Shift the memory store two values
        jmp         ConvertMaskedUpperVector
ConvertUpper1:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE*3       // Shift the memory store only one value
        jmp         ConvertMaskedUpperVector
ConvertUpper3:
        vpsrldq     xmm2, xmm2, SINGLE_SIZE         // Shift the memory store three values
ConvertMaskedUpperVector:
        vmaskmovps  xmmword PTR [rsi], xmm2, xmm1   // Store the masked data, the shift is done in 8bit multiples

        jmp ExitRoutine
ExitRoutine:
        ret
