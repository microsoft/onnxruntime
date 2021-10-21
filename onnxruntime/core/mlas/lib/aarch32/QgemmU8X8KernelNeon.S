/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelNeon.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

--*/

#include "asmmacro.h"

        .syntax unified
        .arch   armv7-a
        .thumb

//
// Stack frame layout for the U8X8 kernel.
//

        .equ    .LGemmU8X8KernelFrame_SavedGeneralRegisters, (7 * 4)
        .equ    .LGemmU8X8KernelFrame_SavedNeonRegisters, (8 * 8)
        .equ    .LGemmU8X8KernelFrame_SavedRegisters, .LGemmU8X8KernelFrame_SavedGeneralRegisters + .LGemmU8X8KernelFrame_SavedNeonRegisters
        .equ    .LGemmU8X8KernelFrame_CountM, 0 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_CountN, 4 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_ldc, 8 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_RowSumBuffer, 12 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_ColumnSumBuffer, 16 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_ZeroPointB, 20 + .LGemmU8X8KernelFrame_SavedRegisters
        .equ    .LGemmU8X8KernelFrame_ZeroMode, 24 + .LGemmU8X8KernelFrame_SavedRegisters

        .text

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (r0) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_NEON>.

    B (r1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_NEON>.

    C (r2) - Supplies the address of matrix C.

    PackedCountK (r3) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for matrix
        A and matrix C. The actual number of rows handled for this invocation
        depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to iterate
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumBuffer - Supplies the sum of each row from matrix A. These values have
        been pre-scaled by the zero point offset of matrix B if the offset is
        per-tensor (ZeroPointB is nullptr). Otherwise, these values must be
        scaled by the per-column zero point offsets of matrix B. These values are
        accumulated into every row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    ZeroPointB - Optionally supplies the per-column zero point offsets of matrix
        B, else nullptr if the matrix B is using per-tensor quantization.

    ZeroMode - Supplies true if the output matrix must be zero initialized, else
        false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY MlasGemmU8X8KernelNeon

//
// Register usage:
//
//      q0-q1 (d0-d3)       matrix B data
//      q2-q3 (d4-d7)       matrix A data
//      q4 (d8-d9)          packed matrix A data
//      q5 (d10-d11)        RowSumBuffer data
//      q6-q7 (d12-d15)     ColumnSumBuffer data
//      q8-q15              accumulators[4][2]
//

        push    {r4,r5,r6,r7,r8,r9,r10}
        vpush   {d8-d15}
        ldr     r4,[sp,#.LGemmU8X8KernelFrame_CountM]
        ldr     r5,[sp,#.LGemmU8X8KernelFrame_ZeroMode]
        ldr     r7,[sp,#.LGemmU8X8KernelFrame_ZeroPointB]
        ldr     r8,[sp,#.LGemmU8X8KernelFrame_ColumnSumBuffer]
        ldr     r9,[sp,#.LGemmU8X8KernelFrame_RowSumBuffer]
        ldr     r10,[sp,#.LGemmU8X8KernelFrame_ldc]
        ldr     r12,[sp,#.LGemmU8X8KernelFrame_CountN]
        vld1.32 {d10-d11},[r9]              // load RowSumBuffer
        mov     r6,r0
        mov     r9,r3
        cmp     r4,#1                       // CountM == 1?
        beq     .LGemmU8X8.M1.ProcessNextColumnLoop
        cmp     r4,#4                       // CountM < 4?
        blo     .LGemmU8X8.M2.ProcessNextColumnLoop

//
// Process 4 rows of the matrices.
//

.LGemmU8X8.M4.ProcessNextColumnLoop:
        vldr    d0,[r1]                     // load packed B0
        mov     r0,r6                       // reload matrix A
        vld1.32 {d12-d15},[r8]!             // load ColumnSumBuffer
        mov     r3,r9                       // reload PackedCountK
        vmovl.u8 q0,d0
        vdup.32 q9,d10[0]
        vdup.32 q11,d10[1]
        vdup.32 q13,d11[0]
        vdup.32 q15,d11[1]
        cbz     r7,.LGemmU8X8.M4.SkipScaleByZeroPointB
        vld1.32 {d8-d9},[r7]!               // load ZeroPointB0
        vmul.u32 q8,q9,q4
        vmul.u32 q10,q11,q4
        vmul.u32 q12,q13,q4
        vmul.u32 q14,q15,q4
        vld1.32 {d8-d9},[r7]!               // load ZeroPointB1
        vmul.u32 q9,q9,q4
        vmul.u32 q11,q11,q4
        vmul.u32 q13,q13,q4
        vmul.u32 q15,q15,q4
        vldr    d8,[r0]                     // load first packed A0
        vadd.u32 q8,q8,q6
        vadd.u32 q9,q9,q7
        vadd.u32 q10,q10,q6
        vadd.u32 q11,q11,q7
        vldr    d9,[r0,#8]                  // load first packed A1
        vadd.u32 q12,q12,q6
        vadd.u32 q13,q13,q7
        vadd.u32 q14,q14,q6
        vadd.u32 q15,q15,q7
        b       .LGemmU8X8.M4.ComputeBlockLoop

.LGemmU8X8.M4.SkipScaleByZeroPointB:
        vldr    d8,[r0]                     // load first packed A0
        vadd.u32 q8,q9,q6
        vadd.u32 q9,q9,q7
        vadd.u32 q10,q11,q6
        vadd.u32 q11,q11,q7
        vldr    d9,[r0,#8]                  // load first packed A1
        vadd.u32 q12,q13,q6
        vadd.u32 q13,q13,q7
        vadd.u32 q14,q15,q6
        vadd.u32 q15,q15,q7

.LGemmU8X8.M4.ComputeBlockLoop:
        vmovl.u8 q2,d8
        add     r0,#16
        vmovl.u8 q3,d9
        vldr    d2,[r1,#8]                  // load packed B1
        vmlal.u16 q8,d0,d4[0]
        vmlal.u16 q9,d1,d4[0]
        vmlal.u16 q10,d0,d5[0]
        vmlal.u16 q11,d1,d5[0]
        vmovl.u8 q1,d2
        vmlal.u16 q12,d0,d6[0]
        vmlal.u16 q13,d1,d6[0]
        vmlal.u16 q14,d0,d7[0]
        vmlal.u16 q15,d1,d7[0]
        vldr    d0,[r1,#16]                 // load packed B2
        vmlal.u16 q8,d2,d4[1]
        vmlal.u16 q9,d3,d4[1]
        vmlal.u16 q10,d2,d5[1]
        vmlal.u16 q11,d3,d5[1]
        vmovl.u8 q0,d0
        vmlal.u16 q12,d2,d6[1]
        vmlal.u16 q13,d3,d6[1]
        vmlal.u16 q14,d2,d7[1]
        vmlal.u16 q15,d3,d7[1]
        vldr    d2,[r1,#24]                 // load packed B3
        add     r1,#32
        subs    r3,#1
        beq     .LGemmU8X8.M4.ComputeBlockLoopFinish
        vmlal.u16 q8,d0,d4[2]
        vmlal.u16 q9,d1,d4[2]
        vmlal.u16 q10,d0,d5[2]
        vmlal.u16 q11,d1,d5[2]
        vmovl.u8 q1,d2
        vldr    d8,[r0]                     // load next packed A0
        vmlal.u16 q12,d0,d6[2]
        vmlal.u16 q13,d1,d6[2]
        vmlal.u16 q14,d0,d7[2]
        vmlal.u16 q15,d1,d7[2]
        vldr    d0,[r1]                     // load packed B0
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        vmlal.u16 q10,d2,d5[3]
        vmlal.u16 q11,d3,d5[3]
        vmovl.u8 q0,d0
        vldr    d9,[r0,#8]                  // load next packed A1
        vmlal.u16 q12,d2,d6[3]
        vmlal.u16 q13,d3,d6[3]
        vmlal.u16 q14,d2,d7[3]
        vmlal.u16 q15,d3,d7[3]
        b       .LGemmU8X8.M4.ComputeBlockLoop

.LGemmU8X8.M4.ComputeBlockLoopFinish:
        vmlal.u16 q8,d0,d4[2]               // finish computing tail vectors
        vmlal.u16 q9,d1,d4[2]
        add     r0,r2,r10,lsl #2            // compute output row 2
        vmlal.u16 q10,d0,d5[2]
        vmlal.u16 q11,d1,d5[2]
        vmovl.u8 q1,d2
        vmlal.u16 q12,d0,d6[2]
        vmlal.u16 q13,d1,d6[2]
        vmlal.u16 q14,d0,d7[2]
        vmlal.u16 q15,d1,d7[2]
        add     r3,r0,r10,lsl #2            // compute output row 3
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        vmlal.u16 q10,d2,d5[3]
        vmlal.u16 q11,d3,d5[3]
        vmlal.u16 q12,d2,d6[3]
        vmlal.u16 q13,d3,d6[3]
        add     r4,r3,r10,lsl #2            // compute output row 4
        vmlal.u16 q14,d2,d7[3]
        vmlal.u16 q15,d3,d7[3]
        subs    r12,#8                      // adjust CountN remaining
        blo     .LGemmU8X8.M4.StoreOutputPartial
        cbnz    r5,.LGemmU8X8.M4.SkipAccumulateOutput
        vld1.32 {d0-d3},[r2]
        vld1.32 {d4-d7},[r0]
        vadd.u32 q8,q8,q0
        vadd.u32 q9,q9,q1
        vld1.32 {d0-d3},[r3]
        vadd.u32 q10,q10,q2
        vadd.u32 q11,q11,q3
        vld1.32 {d4-d7},[r4]
        vadd.u32 q12,q12,q0
        vadd.u32 q13,q13,q1
        vadd.u32 q14,q14,q2
        vadd.u32 q15,q15,q3

.LGemmU8X8.M4.SkipAccumulateOutput:
        vst1.32 {d16-d19},[r2]!
        vst1.32 {d20-d23},[r0]
        vst1.32 {d24-d27},[r3]
        vst1.32 {d28-d31},[r4]
        cmp     r12,#0
        bne     .LGemmU8X8.M4.ProcessNextColumnLoop

.LGemmU8X8.M4.ExitKernel:
        mov     r0,#4                       // return number of rows handled
        vpop    {d8-d15}
        pop     {r4,r5,r6,r7,r8,r9,r10}
        bx      lr

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

.LGemmU8X8.M4.StoreOutputPartial:
        cbz     r5,.LGemmU8X8.M4.StoreOutputPartial.AddMode

.LGemmU8X8.M4.StoreOutputPartial.ZeroMode:
        tst     r12,#4
        beq     .LGemmU8X8.M4.StoreOutputPartial2.ZeroMode
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down
        vst1.32 {d20-d21},[r0]!
        vmov    q10,q11
        vst1.32 {d24-d25},[r3]!
        vmov    q12,q13
        vst1.32 {d28-d29},[r4]!
        vmov    q14,q15

.LGemmU8X8.M4.StoreOutputPartial2.ZeroMode:
        tst     r12,#2
        beq     .LGemmU8X8.M4.StoreOutputPartial1.ZeroMode
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down
        vst1.32 {d20},[r0]!
        vmov    d20,d21
        vst1.32 {d24},[r3]!
        vmov    d24,d25
        vst1.32 {d28},[r4]!
        vmov    d28,d29

.LGemmU8X8.M4.StoreOutputPartial1.ZeroMode:
        tst     r12,#1
        beq     .LGemmU8X8.M4.ExitKernel
        vst1.32 d16[0],[r2]
        vst1.32 d20[0],[r0]
        vst1.32 d24[0],[r3]
        vst1.32 d28[0],[r4]
        b       .LGemmU8X8.M4.ExitKernel

.LGemmU8X8.M4.StoreOutputPartial.AddMode:
        tst     r12,#4
        beq     .LGemmU8X8.M4.StoreOutputPartial2.AddMode
        vld1.32 {d0-d1},[r2]
        vld1.32 {d4-d5},[r0]
        vadd.u32 q8,q8,q0
        vld1.32 {d0-d1},[r3]
        vadd.u32 q10,q10,q2
        vld1.32 {d4-d5},[r4]
        vadd.u32 q12,q12,q0
        vadd.u32 q14,q14,q2
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down
        vst1.32 {d20-d21},[r0]!
        vmov    q10,q11
        vst1.32 {d24-d25},[r3]!
        vmov    q12,q13
        vst1.32 {d28-d29},[r4]!
        vmov    q14,q15

.LGemmU8X8.M4.StoreOutputPartial2.AddMode:
        tst     r12,#2
        beq     .LGemmU8X8.M4.StoreOutputPartial1.AddMode
        vld1.32 {d0},[r2]
        vld1.32 {d4},[r0]
        vadd.u32 d16,d16,d0
        vld1.32 {d0},[r3]
        vadd.u32 d20,d20,d4
        vld1.32 {d4},[r4]
        vadd.u32 d24,d24,d0
        vadd.u32 d28,d28,d4
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down
        vst1.32 {d20},[r0]!
        vmov    d20,d21
        vst1.32 {d24},[r3]!
        vmov    d24,d25
        vst1.32 {d28},[r4]!
        vmov    d28,d29

.LGemmU8X8.M4.StoreOutputPartial1.AddMode:
        tst     r12,#1
        beq     .LGemmU8X8.M4.ExitKernel
        vld1.32 d0[0],[r2]
        vld1.32 d4[0],[r0]
        vadd.u32 d16,d16,d0
        vld1.32 d0[0],[r3]
        vadd.u32 d20,d20,d4
        vld1.32 d4[0],[r4]
        vadd.u32 d24,d24,d0
        vadd.u32 d28,d28,d4
        vst1.32 d16[0],[r2]
        vst1.32 d20[0],[r0]
        vst1.32 d24[0],[r3]
        vst1.32 d28[0],[r4]
        b       .LGemmU8X8.M4.ExitKernel

//
// Process 2 rows of the matrices.
//

.LGemmU8X8.M2.ProcessNextColumnLoop:
        vldr    d0,[r1]                     // load packed B0
        mov     r0,r6                       // reload matrix A
        vld1.32 {d12-d15},[r8]!             // load ColumnSumBuffer
        mov     r3,r9                       // reload PackedCountK
        vmovl.u8 q0,d0
        vdup.32 q9,d10[0]
        vdup.32 q11,d10[1]
        cbz     r7,.LGemmU8X8.M2.SkipScaleByZeroPointB
        vld1.32 {d28-d31},[r7]!             // load ZeroPointB
        vmul.u32 q8,q9,q14
        vmul.u32 q9,q9,q15
        vmul.u32 q10,q11,q14
        vmul.u32 q11,q11,q15
        vld1.32 d8,[r0]!                    // load first packed A0
        vadd.u32 q8,q8,q6
        vadd.u32 q9,q9,q7
        vadd.u32 q10,q10,q6
        vadd.u32 q11,q11,q7
        b       .LGemmU8X8.M2.ComputeBlockLoop

.LGemmU8X8.M2.SkipScaleByZeroPointB:
        vld1.32 d8,[r0]!                    // load first packed A0
        vadd.u32 q8,q9,q6
        vadd.u32 q9,q9,q7
        vadd.u32 q10,q11,q6
        vadd.u32 q11,q11,q7

.LGemmU8X8.M2.ComputeBlockLoop:
        vmovl.u8 q2,d8
        vldr    d2,[r1,#8]                  // load packed B1
        vmlal.u16 q8,d0,d4[0]
        vmlal.u16 q9,d1,d4[0]
        vmlal.u16 q10,d0,d5[0]
        vmlal.u16 q11,d1,d5[0]
        vmovl.u8 q1,d2
        vldr    d0,[r1,#16]                 // load packed B2
        vmlal.u16 q8,d2,d4[1]
        vmlal.u16 q9,d3,d4[1]
        vmlal.u16 q10,d2,d5[1]
        vmlal.u16 q11,d3,d5[1]
        vmovl.u8 q0,d0
        vldr    d2,[r1,#24]                 // load packed B3
        add     r1,#32
        subs    r3,#1
        beq     .LGemmU8X8.M2.ComputeBlockLoopFinish
        vmlal.u16 q8,d0,d4[2]
        vmlal.u16 q9,d1,d4[2]
        vmlal.u16 q10,d0,d5[2]
        vmlal.u16 q11,d1,d5[2]
        vmovl.u8 q1,d2
        vld1.32 d8,[r0]!                    // load next packed A0
        vldr    d0,[r1]                     // load packed B0
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        vmlal.u16 q10,d2,d5[3]
        vmlal.u16 q11,d3,d5[3]
        vmovl.u8 q0,d0
        b       .LGemmU8X8.M2.ComputeBlockLoop

.LGemmU8X8.M2.ComputeBlockLoopFinish:
        vmlal.u16 q8,d0,d4[2]               // finish computing tail vectors
        vmlal.u16 q9,d1,d4[2]
        add     r0,r2,r10,lsl #2            // compute output row 2
        vmlal.u16 q10,d0,d5[2]
        vmlal.u16 q11,d1,d5[2]
        vmovl.u8 q1,d2
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        vmlal.u16 q10,d2,d5[3]
        vmlal.u16 q11,d3,d5[3]
        subs    r12,#8                      // adjust CountN remaining
        blo     .LGemmU8X8.M2.StoreOutputPartial
        cbnz    r5,.LGemmU8X8.M2.SkipAccumulateOutput
        vld1.32 {d0-d3},[r2]
        vld1.32 {d4-d7},[r0]
        vadd.u32 q8,q8,q0
        vadd.u32 q9,q9,q1
        vadd.u32 q10,q10,q2
        vadd.u32 q11,q11,q3

.LGemmU8X8.M2.SkipAccumulateOutput:
        vst1.32 {d16-d19},[r2]!
        vst1.32 {d20-d23},[r0]
        cmp     r12,#0
        bne     .LGemmU8X8.M2.ProcessNextColumnLoop

.LGemmU8X8.M2.ExitKernel:
        mov     r0,#2                       // return number of rows handled
        vpop    {d8-d15}
        pop     {r4,r5,r6,r7,r8,r9,r10}
        bx      lr

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

.LGemmU8X8.M2.StoreOutputPartial:
        cbz     r5,.LGemmU8X8.M2.StoreOutputPartial.AddMode

.LGemmU8X8.M2.StoreOutputPartial.ZeroMode:
        tst     r12,#4
        beq     .LGemmU8X8.M2.StoreOutputPartial2.ZeroMode
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down
        vst1.32 {d20-d21},[r0]!
        vmov    q10,q11

.LGemmU8X8.M2.StoreOutputPartial2.ZeroMode:
        tst     r12,#2
        beq     .LGemmU8X8.M2.StoreOutputPartial1.ZeroMode
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down
        vst1.32 {d20},[r0]!
        vmov    d20,d21

.LGemmU8X8.M2.StoreOutputPartial1.ZeroMode:
        tst     r12,#1
        beq     .LGemmU8X8.M2.ExitKernel
        vst1.32 d16[0],[r2]
        vst1.32 d20[0],[r0]
        b       .LGemmU8X8.M2.ExitKernel

.LGemmU8X8.M2.StoreOutputPartial.AddMode:
        tst     r12,#4
        beq     .LGemmU8X8.M2.StoreOutputPartial2.AddMode
        vld1.32 {d0-d1},[r2]
        vld1.32 {d4-d5},[r0]
        vadd.u32 q8,q8,q0
        vadd.u32 q10,q10,q2
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down
        vst1.32 {d20-d21},[r0]!
        vmov    q10,q11

.LGemmU8X8.M2.StoreOutputPartial2.AddMode:
        tst     r12,#2
        beq     .LGemmU8X8.M2.StoreOutputPartial1.AddMode
        vld1.32 {d0},[r2]
        vld1.32 {d4},[r0]
        vadd.u32 d16,d16,d0
        vadd.u32 d20,d20,d4
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down
        vst1.32 {d20},[r0]!
        vmov    d20,d21

.LGemmU8X8.M2.StoreOutputPartial1.AddMode:
        tst     r12,#1
        beq     .LGemmU8X8.M2.ExitKernel
        vld1.32 d0[0],[r2]
        vld1.32 d4[0],[r0]
        vadd.u32 d16,d16,d0
        vadd.u32 d20,d20,d4
        vst1.32 d16[0],[r2]
        vst1.32 d20[0],[r0]
        b       .LGemmU8X8.M2.ExitKernel

//
// Process 1 row of the matrices.
//

.LGemmU8X8.M1.ProcessNextColumnLoop:
        vldr    d0,[r1]                     // load packed B0
        mov     r0,r6                       // reload matrix A
        vld1.32 {d12-d15},[r8]!             // load ColumnSumBuffer
        mov     r3,r9                       // reload PackedCountK
        vmovl.u8 q0,d0
        vdup.32 q9,d10[0]
        cbz     r7,.LGemmU8X8.M1.SkipScaleByZeroPointB
        vld1.32 {d28-d31},[r7]!             // load ZeroPointB
        vmul.u32 q8,q9,q14
        vmul.u32 q9,q9,q15
        vld1.32 d8[0],[r0]!                 // load first packed A0
        vadd.u32 q8,q8,q6
        vadd.u32 q9,q9,q7
        b       .LGemmU8X8.M1.ComputeBlockLoop

.LGemmU8X8.M1.SkipScaleByZeroPointB:
        vld1.32 d8[0],[r0]!                 // load first packed A0
        vadd.u32 q8,q9,q6
        vadd.u32 q9,q9,q7

.LGemmU8X8.M1.ComputeBlockLoop:
        vmovl.u8 q2,d8
        vldr    d2,[r1,#8]                  // load packed B1
        vmlal.u16 q8,d0,d4[0]
        vmlal.u16 q9,d1,d4[0]
        vmovl.u8 q1,d2
        vldr    d0,[r1,#16]                 // load packed B2
        vmlal.u16 q8,d2,d4[1]
        vmlal.u16 q9,d3,d4[1]
        vmovl.u8 q0,d0
        vldr    d2,[r1,#24]                 // load packed B3
        add     r1,#32
        subs    r3,#1
        beq     .LGemmU8X8.M1.ComputeBlockLoopFinish
        vmlal.u16 q8,d0,d4[2]
        vmlal.u16 q9,d1,d4[2]
        vmovl.u8 q1,d2
        vld1.32 d8[0],[r0]!                 // load next packed A0
        vldr    d0,[r1]                     // load packed B0
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        vmovl.u8 q0,d0
        b       .LGemmU8X8.M1.ComputeBlockLoop

.LGemmU8X8.M1.ComputeBlockLoopFinish:
        vmlal.u16 q8,d0,d4[2]               // finish computing tail vectors
        vmlal.u16 q9,d1,d4[2]
        vmovl.u8 q1,d2
        vmlal.u16 q8,d2,d4[3]
        vmlal.u16 q9,d3,d4[3]
        subs    r12,#8                      // adjust CountN remaining
        blo     .LGemmU8X8.M1.StoreOutputPartial
        cbnz    r5,.LGemmU8X8.M1.SkipAccumulateOutput
        vld1.32 {d0-d3},[r2]
        vadd.u32 q8,q8,q0
        vadd.u32 q9,q9,q1

.LGemmU8X8.M1.SkipAccumulateOutput:
        vst1.32 {d16-d19},[r2]!
        cmp     r12,#0
        bne     .LGemmU8X8.M1.ProcessNextColumnLoop

.LGemmU8X8.M1.ExitKernel:
        mov     r0,#1                       // return number of rows handled
        vpop    {d8-d15}
        pop     {r4,r5,r6,r7,r8,r9,r10}
        bx      lr

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

.LGemmU8X8.M1.StoreOutputPartial:
        cbz     r5,.LGemmU8X8.M1.StoreOutputPartial.AddMode

.LGemmU8X8.M1.StoreOutputPartial.ZeroMode:
        tst     r12,#4
        beq     .LGemmU8X8.M1.StoreOutputPartial2.ZeroMode
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down

.LGemmU8X8.M1.StoreOutputPartial2.ZeroMode:
        tst     r12,#2
        beq     .LGemmU8X8.M1.StoreOutputPartial1.ZeroMode
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down

.LGemmU8X8.M1.StoreOutputPartial1.ZeroMode:
        tst     r12,#1
        beq     .LGemmU8X8.M1.ExitKernel
        vst1.32 d16[0],[r2]
        b       .LGemmU8X8.M1.ExitKernel

.LGemmU8X8.M1.StoreOutputPartial.AddMode:
        tst     r12,#4
        beq     .LGemmU8X8.M1.StoreOutputPartial2.AddMode
        vld1.32 {d0-d1},[r2]
        vadd.u32 q8,q8,q0
        vst1.32 {d16-d17},[r2]!
        vmov    q8,q9                       // shift remaining elements down

.LGemmU8X8.M1.StoreOutputPartial2.AddMode:
        tst     r12,#2
        beq     .LGemmU8X8.M1.StoreOutputPartial1.AddMode
        vld1.32 {d0},[r2]
        vadd.u32 d16,d16,d0
        vst1.32 {d16},[r2]!
        vmov    d16,d17                     // shift remaining elements down

.LGemmU8X8.M1.StoreOutputPartial1.AddMode:
        tst     r12,#1
        beq     .LGemmU8X8.M1.ExitKernel
        vld1.32 d0[0],[r2]
        vadd.u32 d16,d16,d0
        vst1.32 d16[0],[r2]
        b       .LGemmU8X8.M1.ExitKernel

        .end
