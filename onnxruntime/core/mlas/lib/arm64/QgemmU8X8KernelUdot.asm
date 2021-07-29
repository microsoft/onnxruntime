/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelUdot.asm

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses ARM v8.4 dot product instructions.

--*/

#include "kxarm64.h"
#include "AssembleDotProduct.h"

//
// Stack frame layout for the U8X8 kernel.
// Defining spaces for saving 8 vector registers, and pointers to parameters
// on the stack
//

#define GemmU8X8KernelFrame_SavedNeonRegisters       (8 * 8)
#define GemmU8X8KernelFrame_SavedRegisters           GemmU8X8KernelFrame_SavedNeonRegisters
#define GemmU8X8KernelFrame_ColumnSumBuffer          (0 + GemmU8X8KernelFrame_SavedRegisters)
#define GemmU8X8KernelFrame_ZeroPointB               (8 + GemmU8X8KernelFrame_SavedRegisters)
#define GemmU8X8KernelFrame_ZeroMode                 (16 + GemmU8X8KernelFrame_SavedRegisters)

        TEXTAREA

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_UDOT>.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_UDOT>.

    C (x2) - Supplies the address of matrix C.

    PackedCountK (x3) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.

    CountM (x4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (x5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc (x6) - Supplies the first dimension of matrix C.

    RowSumBuffer (x7) - Supplies the sum of each row from matrix A. These values
        have been pre-scaled by the zero point offset of matrix B if the offset
        is per-tensor (ZeroPointB is nullptr). Otherwise, these values must be
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

        NESTED_ENTRY MlasGemmU8X8KernelUdot

        stp     d8,d9,[sp,#-64]!
        ldr     x8,[sp,#GemmU8X8KernelFrame_ColumnSumBuffer]
        ldr     x9,[sp,#GemmU8X8KernelFrame_ZeroPointB]
        ldrb    w13,[sp,#GemmU8X8KernelFrame_ZeroMode]
        mov     x14,x0
        ld1     {v11.4s},[x7],#16
        mov     x15,x3
        dup     v8.4s,v11.s[0]              // broadcast row fixups
        cmp     x4,#1                       // CountM == 1?
        beq     ProcessNextColumnLoopM1
        dup     v9.4s,v11.s[1]
        cmp     x4,#4                       // CountM < 4?
        blo     ProcessNextColumnLoopM2
        stp     d10,d11,[sp,#16]
        dup     v10.4s,v11.s[2]
        dup     v11.4s,v11.s[3]

        cmp     x4,#8                       // CountM < 8?
        blo     ProcessNextColumnLoopM4
        stp     d14,d15,[sp,#48]
        ld1     {v15.4s},[x7]               // load row sum 5 ~ 8
        stp     d12,d13,[sp,#32]
        dup     v12.4s,v15.s[0]
        dup     v13.4s,v15.s[1]
        dup     v14.4s,v15.s[2]
        dup     v15.4s,v15.s[3]

//
// Process 8 rows of the matrices.
//
// v8 ~ v15 row sums 
//                                      int8 RHS 4x8 block
//                           /-----------------------------------------|
//                           |v0.b[0] ... v0.b[12] v1.b[0] ... v1.b[12]|
//                           |  ...                              ...   |
//                           |v0.b[3] ... v0.b[15] v1.b[3] ... v1.b[15]|
//                           \-----------------------------------------/
//    int8 LHS 8x4 block
//  /---------------------\  /-----------------------------------------|
//  |v4.b[0]  ... v4.b[3] |  |v16.s[0] .. v16.s[3] v17.s[0] .. v17.s[3]|
//  |v4.b[4]  ... v4.b[7] |  |v18.s[0] .. v18.s[3] v19.s[0] .. v19.s[3]|
//  |v4.b[8]  ... v4.b[11]|  |v20.s[0] .. v20.s[3] v21.s[0] .. v21.s[3]|
//  |v4.b[12] ... v4.b[15]|  |v22.s[0] .. v22.s[3] v23.s[0] .. v23.s[3]|
//  |v5.b[0]  ... v5.b[3] |  |v24.s[0] .. v24.s[3] v25.s[0] .. v25.s[3]|
//  |v5.b[4]  ... v5.b[7] |  |v26.s[0] .. v26.s[3] v27.s[0] .. v27.s[3]|
//  |v5.b[8]  ... v5.b[11]|  |v28.s[0] .. v28.s[3] v29.s[0] .. v29.s[3]|
//  |v5.b[12] ... v5.b[15]|  |v30.s[0] .. v30.s[3] v31.s[0] .. v31.s[3]|
//  \---------------------/  \-----------------------------------------/
//////////////////////////
//  unroll for the next 4 in k dimension
//                           /-----------------------------------------|
//                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
//                           |  ...                              ...   |
//                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
//                           \-----------------------------------------/
//  /---------------------\  /-----------------------------------------\
//  |v6.b[0]  ... v6.b[3] |  |v16.s[0] .. v16.s[3] v17.s[0] .. v17.s[3]|
//  |v6.b[4]  ... v6.b[7] |  |v18.s[0] .. v18.s[3] v19.s[0] .. v19.s[3]|
//  |v6.b[8]  ... v6.b[11]|  |v20.s[0] .. v20.s[3] v21.s[0] .. v21.s[3]|
//  |v6.b[12] ... v6.b[15]|  |v22.s[0] .. v22.s[3] v23.s[0] .. v23.s[3]|
//  |v7.b[0]  ... v7.b[3] |  |v24.s[0] .. v24.s[3] v25.s[0] .. v25.s[3]|
//  |v7.b[4]  ... v7.b[7] |  |v26.s[0] .. v26.s[3] v27.s[0] .. v27.s[3]|
//  |v7.b[8]  ... v7.b[11]|  |v28.s[0] .. v28.s[3] v29.s[0] .. v29.s[3]|
//  |v7.b[12] ... v7.b[15]|  |v30.s[0] .. v30.s[3] v31.s[0] .. v31.s[3]|
//  \---------------------/  \-----------------------------------------/
//                                  int32 accumulators 8x8 block


// Starting the loop: initialize accumulators with scaled combination
//                    of row and column sums
ProcessNextColumnLoopM8
        mov     x0,x14                      // reload matrix A
        ld1     {v3.4s},[x8],#16            // load ColumnSumBuffer[0]
        mov     x3,x15                      // reload PackedCountK
        ld1     {v7.4s},[x8],#16            // load ColumnSumBuffer[4]
        cbz     x9,SkipScaleByZeroPointBM8

        // accumulator = zero point B * row sum A + column sum B 
        ld1     {v30.4s},[x9],#16           // load ZeroPointB[0]
        mul     v16.4s,v30.4s,v8.4s
        mul     v18.4s,v30.4s,v9.4s
        ld1     {v31.4s},[x9],#16           // load ZeroPointB[4]
        mul     v20.4s,v30.4s,v10.4s
        mul     v22.4s,v30.4s,v11.4s
        mul     v24.4s,v30.4s,v12.4s
        mul     v26.4s,v30.4s,v13.4s
        mul     v28.4s,v30.4s,v14.4s
        mul     v30.4s,v30.4s,v15.4s

        mul     v17.4s,v31.4s,v8.4s
        mul     v19.4s,v31.4s,v9.4s
        mul     v21.4s,v31.4s,v10.4s
        mul     v23.4s,v31.4s,v11.4s
        mul     v25.4s,v31.4s,v12.4s
        mul     v27.4s,v31.4s,v13.4s
        mul     v29.4s,v31.4s,v14.4s
        mul     v31.4s,v31.4s,v15.4s

        ld1     {v0.16b},[x1],#16           // load packed B0
        add     v16.4s,v3.4s,v16.4s
        add     v18.4s,v3.4s,v18.4s
        ldr     q4,[x0],#16                 // load packed A0
        add     v20.4s,v3.4s,v20.4s
        add     v22.4s,v3.4s,v22.4s
        ldr     q5,[x0],#16                 // load packed A1
        add     v24.4s,v3.4s,v24.4s
        add     v26.4s,v3.4s,v26.4s
        ld1     {v1.16b},[x1],#16           // load packed B1
        add     v28.4s,v3.4s,v28.4s
        add     v30.4s,v3.4s,v30.4s
        ldr     q6,[x0],#16                 // load packed A2
        add     v17.4s,v7.4s,v17.4s
        add     v19.4s,v7.4s,v19.4s
        ld1     {v2.16b},[x1],#16           // load packed B0_next4k
        add     v21.4s,v7.4s,v21.4s
        add     v23.4s,v7.4s,v23.4s
        add     v25.4s,v7.4s,v25.4s
        add     v27.4s,v7.4s,v27.4s
        add     v29.4s,v7.4s,v29.4s
        add     v31.4s,v7.4s,v31.4s
        b       ComputeBlockLoopStartM8

SkipScaleByZeroPointBM8
        // accumulator = row sum A + column sum B 
        ld1     {v0.16b},[x1],#16           // load packed B0
        add     v16.4s,v3.4s,v8.4s
        add     v18.4s,v3.4s,v9.4s
        ldr     q4,[x0],#16                 // load packed A0
        add     v20.4s,v3.4s,v10.4s
        add     v22.4s,v3.4s,v11.4s
        ldr     q5,[x0],#16                 // load packed A1
        add     v24.4s,v3.4s,v12.4s
        add     v26.4s,v3.4s,v13.4s
        ld1     {v1.16b},[x1],#16           // load packed B1
        add     v28.4s,v3.4s,v14.4s
        add     v30.4s,v3.4s,v15.4s
        ldr     q6,[x0],#16                 // load packed A2
        add     v17.4s,v7.4s,v8.4s
        add     v19.4s,v7.4s,v9.4s
        ld1     {v2.16b},[x1],#16           // load packed B0_next4k
        add     v21.4s,v7.4s,v10.4s
        add     v23.4s,v7.4s,v11.4s
        add     v25.4s,v7.4s,v12.4s
        add     v27.4s,v7.4s,v13.4s
        add     v29.4s,v7.4s,v14.4s
        add     v31.4s,v7.4s,v15.4s


ComputeBlockLoopStartM8

ComputeBlockLoopM8
        sub     x3,x3,#1
        ld1     {v3.16b},[x1],#16           // load packed B1_next4k
        UdotByElement 16, 0, 4, 0
        UdotByElement 18, 0, 4, 1
        ldr     q7,[x0],#16                 // load packed A3
        UdotByElement 20, 0, 4, 2
        UdotByElement 22, 0, 4, 3
        cbz     x3,ComputeBlockLoopFinishM8
        UdotByElement 17, 1, 4, 0
        UdotByElement 19, 1, 4, 1
        UdotByElement 21, 1, 4, 2
        UdotByElement 23, 1, 4, 3
        ldr     q4,[x0],#16                 // load packed A0 for next iteration

        UdotByElement 24, 0, 5, 0
        UdotByElement 26, 0, 5, 1
        UdotByElement 28, 0, 5, 2
        UdotByElement 30, 0, 5, 3
        ld1     {v0.16b},[x1],#16           // load packed B0 for next iteration

        UdotByElement 25, 1, 5, 0
        UdotByElement 27, 1, 5, 1
        UdotByElement 29, 1, 5, 2
        UdotByElement 31, 1, 5, 3
        ld1     {v1.16b},[x1],#16           // load packed B1 for next iteration


        UdotByElement 16, 2, 6, 0
        UdotByElement 18, 2, 6, 1
        ldr     q5,[x0],#16                 // load packed A1 for next iteration
        UdotByElement 20, 2, 6, 2
        UdotByElement 22, 2, 6, 3
        UdotByElement 17, 3, 6, 0
        UdotByElement 19, 3, 6, 1
        UdotByElement 21, 3, 6, 2
        UdotByElement 23, 3, 6, 3
        ldr     q6,[x0],#16                 // load packed A2 for next iteration

        UdotByElement 24, 2, 7, 0
        UdotByElement 26, 2, 7, 1
        UdotByElement 28, 2, 7, 2
        UdotByElement 30, 2, 7, 3
        ld1     {v2.16b},[x1],#16           // load packed B0_next4k for next iteration
        UdotByElement 25, 3, 7, 0
        UdotByElement 27, 3, 7, 1
        UdotByElement 29, 3, 7, 2
        UdotByElement 31, 3, 7, 3
        b       ComputeBlockLoopM8

ComputeBlockLoopFinishM8
        // postfix, compute tail values and prepare to write results
        // We are either about to go to ProcessNextColumnLoopM8
        // where x0 and x3 are about to be restored, or exit
        // when x0 and x3 will not be used.
        // x4 x7 has finished their task
        // so we can use x0 x3 x4 x7 as output row pointers

        UdotByElement 17, 1, 4, 0
        UdotByElement 19, 1, 4, 1
        add     x10,x2,x6,lsl #2            // compute output row 2
        add     x11,x10,x6,lsl #2           // compute output row 3
        UdotByElement 21, 1, 4, 2
        UdotByElement 23, 1, 4, 3
        add     x12,x11,x6,lsl #2           // compute output row 4
        add     x0,x12,x6,lsl #2            // compute output row 5
        UdotByElement 24, 0, 5, 0
        UdotByElement 26, 0, 5, 1
        add     x3,x0,x6,lsl #2             // compute output row 6
        add     x4,x3,x6,lsl #2             // compute output row 7
        UdotByElement 28, 0, 5, 2
        UdotByElement 30, 0, 5, 3
        add     x7,x4,x6,lsl #2             // compute output row 8
        subs    x5,x5,#8                    // adjust CountN remaining
        UdotByElement 25, 1, 5, 0
        UdotByElement 27, 1, 5, 1
        UdotByElement 29, 1, 5, 2
        UdotByElement 31, 1, 5, 3
        UdotByElement 16, 2, 6, 0
        UdotByElement 18, 2, 6, 1
        UdotByElement 20, 2, 6, 2
        UdotByElement 22, 2, 6, 3
        UdotByElement 17, 3, 6, 0
        UdotByElement 19, 3, 6, 1
        UdotByElement 21, 3, 6, 2
        UdotByElement 23, 3, 6, 3
        UdotByElement 24, 2, 7, 0
        UdotByElement 26, 2, 7, 1
        UdotByElement 28, 2, 7, 2
        UdotByElement 30, 2, 7, 3
        UdotByElement 25, 3, 7, 0
        UdotByElement 27, 3, 7, 1
        UdotByElement 29, 3, 7, 2
        UdotByElement 31, 3, 7, 3
        blo     StoreOutputPartialM8
        cbnz    x13,SkipAccumulateOutputM8
        ldp     q0,q1,[x2]
        ldp     q2,q3,[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v17.4s,v17.4s,v1.4s
        ldp     q4,q5,[x11]
        add     v18.4s,v18.4s,v2.4s
        add     v19.4s,v19.4s,v3.4s
        ldp     q6,q7,[x12]
        add     v20.4s,v20.4s,v4.4s
        add     v21.4s,v21.4s,v5.4s
        ldp     q0, q1, [x0]
        add     v22.4s,v22.4s,v6.4s
        add     v23.4s,v23.4s,v7.4s
        ldp     q2, q3, [x3]
        add     v24.4s,v24.4s,v0.4s
        add     v25.4s,v25.4s,v1.4s
        ldp     q4, q5, [x4]
        add     v26.4s,v26.4s,v2.4s
        add     v27.4s,v27.4s,v3.4s
        ldp     q6, q7, [x7]
        add     v28.4s,v28.4s,v4.4s
        add     v29.4s,v29.4s,v5.4s
        add     v30.4s,v30.4s,v6.4s
        add     v31.4s,v31.4s,v7.4s


SkipAccumulateOutputM8
        stp     q16,q17,[x2],#32
        stp     q18,q19,[x10]
        stp     q20,q21,[x11]
        stp     q22,q23,[x12]
        stp     q24,q25,[x0]
        stp     q26,q27,[x3]
        stp     q28,q29,[x4]
        stp     q30,q31,[x7]

        cbnz    x5,ProcessNextColumnLoopM8

ExitKernelM8
        mov     x0,#8                       // return number of rows handled
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#64
        EPILOG_RETURN

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

StoreOutputPartialM8
        cbz     x13,StoreOutputPartialAddModeM8

StoreOutputPartialZeroModeM8
        tbz     x5,#2,StoreOutputPartial2ZeroModeM8
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b
        st1     {v20.4s},[x11],#16
        mov     v20.16b,v21.16b
        st1     {v22.4s},[x12],#16
        mov     v22.16b,v23.16b
        st1     {v24.4s},[x0],#16
        mov     v24.16b,v25.16b
        st1     {v26.4s},[x3],#16
        mov     v26.16b,v27.16b
        st1     {v28.4s},[x4],#16
        mov     v28.16b,v29.16b
        st1     {v30.4s},[x7],#16
        mov     v30.16b,v31.16b

StoreOutputPartial2ZeroModeM8
        tbz     x5,#1,StoreOutputPartial1ZeroModeM8
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]
        st1     {v20.2s},[x11],#8
        dup     v20.4s,v20.s[2]
        st1     {v22.2s},[x12],#8
        dup     v22.4s,v22.s[2]
        st1     {v24.2s},[x0],#8
        dup     v24.4s,v24.s[2]
        st1     {v26.2s},[x3],#8
        dup     v26.4s,v26.s[2]
        st1     {v28.2s},[x4],#8
        dup     v28.4s,v28.s[2]
        st1     {v30.2s},[x7],#8
        dup     v30.4s,v30.s[2]

StoreOutputPartial1ZeroModeM8
        tbz     x5,#0,ExitKernelM8
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        st1     {v20.s}[0],[x11]
        st1     {v22.s}[0],[x12]
        st1     {v24.s}[0],[x0]
        st1     {v26.s}[0],[x3]
        st1     {v28.s}[0],[x4]
        st1     {v30.s}[0],[x7]
        b       ExitKernelM8

StoreOutputPartialAddModeM8
        tbz     x5,#2,StoreOutputPartial2AddModeM8
        ld1     {v0.4s},[x2]
        ld1     {v1.4s},[x10]
        ld1     {v2.4s},[x11]
        ld1     {v3.4s},[x12]
        ld1     {v4.4s},[x0]
        ld1     {v5.4s},[x3]
        ld1     {v6.4s},[x4]
        ld1     {v7.4s},[x7]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b
        add     v20.4s,v20.4s,v2.4s
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.4s},[x11],#16
        mov     v20.16b,v21.16b
        st1     {v22.4s},[x12],#16
        mov     v22.16b,v23.16b

        add     v24.4s,v24.4s,v4.4s
        add     v26.4s,v26.4s,v5.4s
        st1     {v24.4s},[x0],#16
        mov     v24.16b,v25.16b
        st1     {v26.4s},[x3],#16
        mov     v26.16b,v27.16b
        add     v28.4s,v28.4s,v6.4s
        add     v30.4s,v30.4s,v7.4s
        st1     {v28.4s},[x4],#16
        mov     v28.16b,v29.16b
        st1     {v30.4s},[x7],#16
        mov     v30.16b,v31.16b

StoreOutputPartial2AddModeM8
        tbz     x5,#1,StoreOutputPartial1AddModeM8
        ld1     {v0.2s},[x2]
        ld1     {v1.2s},[x10]
        ld1     {v2.2s},[x11]
        ld1     {v3.2s},[x12]
        ld1     {v4.2s},[x0]
        ld1     {v5.2s},[x3]
        ld1     {v6.2s},[x4]
        ld1     {v7.2s},[x7]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]
        add     v20.4s,v20.4s,v2.4s
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.2s},[x11],#8
        dup     v20.4s,v20.s[2]
        st1     {v22.2s},[x12],#8
        dup     v22.4s,v22.s[2]

        add     v24.4s,v24.4s,v4.4s
        add     v26.4s,v26.4s,v5.4s
        st1     {v24.2s},[x0],#8
        dup     v24.4s,v24.s[2]    
        st1     {v26.2s},[x3],#8
        dup     v26.4s,v26.s[2]
        add     v28.4s,v28.4s,v6.4s
        add     v30.4s,v30.4s,v7.4s
        st1     {v28.2s},[x4],#8
        dup     v28.4s,v28.s[2]
        st1     {v30.2s},[x7],#8
        dup     v30.4s,v30.s[2]


StoreOutputPartial1AddModeM8
        tbz     x5,#0,ExitKernelM8
        ld1     {v0.s}[0],[x2]
        ld1     {v1.s}[0],[x10]
        add     v16.4s,v16.4s,v0.4s
        ld1     {v2.s}[0],[x11]
        add     v18.4s,v18.4s,v1.4s
        ld1     {v3.s}[0],[x12]
        add     v20.4s,v20.4s,v2.4s
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.s}[0],[x11]
        st1     {v22.s}[0],[x12]
        ld1     {v4.s}[0],[x0]
        ld1     {v5.s}[0],[x3]
        ld1     {v6.s}[0],[x4]
        ld1     {v7.s}[0],[x7]
        add     v24.4s,v24.4s,v4.4s
        st1     {v24.s}[0],[x0]
        add     v26.4s,v26.4s,v5.4s
        st1     {v26.s}[0],[x3]
        add     v28.4s,v28.4s,v6.4s
        st1     {v28.s}[0],[x4]
        add     v30.4s,v30.4s,v7.4s
        st1     {v30.s}[0],[x7]

        b       ExitKernelM8


//
// Process 4 rows of the matrices.
//
//
// The packing layout is setup to have a pair of four quad vectors from
// packed matrix A and a pair of eight quad vectors from packed matrix B.
// With this scheme, alternating loads from the packed matrices can be
// interleaved with the dot product instructions.
//
// One negative consequence of using four rows here is that the accumulator
// register tile is too small for processors with high out of order execution
// windows (such as the Apple M1). The dot product instructions for a given
// cell are too close to each other to avoid dependencies. To workaround this,
// the below loop uses a pair of accumulator registers that are then added
// together when the loop finishes.
//
// A55-based cores are optimized for 64-bit loads, so use 64-bit loads for
// packed matrix A. At the time of this implementation, using a wider 128-bit
// load didn't affect performance for higher end cores.
//
//                                      int8 RHS 4x8 block
//                           /-----------------------------------------|
//                           |v0.b[0] ... v0.b[12] v1.b[0] ... v1.b[12]|
//                           |  ...                              ...   |
//                           |v0.b[3] ... v0.b[15] v1.b[3] ... v1.b[15]|
//                           \-----------------------------------------/
//    int8 LHS 4x4 block
//  /---------------------\  /-----------------------------------------|
//  |d4.b[0]  ... d4.b[3] |  |v16.s[0] .. v16.s[3] v17.s[0] .. v17.s[3]|
//  |d4.b[4]  ... d4.b[7] |  |v18.s[0] .. v18.s[3] v19.s[0] .. v19.s[3]|
//  |d5.b[0]  ... d5.b[3] |  |v20.s[0] .. v20.s[3] v21.s[0] .. v21.s[3]|
//  |d5.b[4]  ... d5.b[7] |  |v22.s[0] .. v22.s[3] v23.s[0] .. v23.s[3]|
//  \---------------------/  \-----------------------------------------/

//  /---------------------\  /-----------------------------------------\
//  |d6.b[0]  ... d6.b[3] |  |v24.s[0] .. v24.s[3] v25.s[0] .. v25.s[3]|
//  |d6.b[4]  ... d6.b[7] |  |v26.s[0] .. v26.s[3] v27.s[0] .. v27.s[3]|
//  |d7.b[0]  ... d7.b[3] |  |v28.s[0] .. v24.s[3] v29.s[0] .. v29.s[3]|
//  |d7.b[4]  ... d7.b[7] |  |v30.s[0] .. v24.s[3] v31.s[0] .. v31.s[3]|
//  \---------------------/  \-----------------------------------------/
//                                  int32 accumulators 8x8 block

ProcessNextColumnLoopM4
        ld1     {v0.16b},[x1],#16           // load packed B0
        mov     x0,x14                      // reload matrix A
        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]
        mov     x3,x15                      // reload PackedCountK
        ld1     {v3.4s},[x8],#16            // load ColumnSumBuffer[4]
        cbz     x9,SkipScaleByZeroPointBM4
        ld1     {v30.4s},[x9],#16           // load ZeroPointB[0]
        mul     v16.4s,v30.4s,v8.4s
        mul     v18.4s,v30.4s,v9.4s
        ld1     {v31.4s},[x9],#16           // load ZeroPointB[4]
        mul     v20.4s,v30.4s,v10.4s
        mul     v22.4s,v30.4s,v11.4s
        mul     v17.4s,v31.4s,v8.4s
        mul     v19.4s,v31.4s,v9.4s
        mul     v21.4s,v31.4s,v10.4s
        mul     v23.4s,v31.4s,v11.4s
        add     v16.4s,v2.4s,v16.4s
        add     v18.4s,v2.4s,v18.4s
        add     v20.4s,v2.4s,v20.4s
        add     v22.4s,v2.4s,v22.4s
        add     v17.4s,v3.4s,v17.4s
        add     v19.4s,v3.4s,v19.4s
        add     v21.4s,v3.4s,v21.4s
        add     v23.4s,v3.4s,v23.4s
        b       ComputeBlockLoopStartM4

SkipScaleByZeroPointBM4
        add     v16.4s,v2.4s,v8.4s
        add     v18.4s,v2.4s,v9.4s
        add     v20.4s,v2.4s,v10.4s
        add     v22.4s,v2.4s,v11.4s
        add     v17.4s,v3.4s,v8.4s
        add     v19.4s,v3.4s,v9.4s
        add     v21.4s,v3.4s,v10.4s
        add     v23.4s,v3.4s,v11.4s

ComputeBlockLoopStartM4
        ldr     d4,[x0],#32                 // load packed A0.l
        movi    v24.4s,#0
        movi    v25.4s,#0
        ldur    d5,[x0,#-24]                // load packed A0.h
        movi    v26.4s,#0
        movi    v27.4s,#0
        ldur    d6,[x0,#-16]                // load packed A1.l
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0

ComputeBlockLoopM4
        ld1     {v1.16b},[x1],#16           // load packed B1
        UdotByElement 16, 0, 4, 0
        UdotByElement 18, 0, 4, 1
        ldur    d7,[x0,#-8]                 // load packed A1.h
        UdotByElement 20, 0, 5, 0
        UdotByElement 22, 0, 5, 1
        ld1     {v0.16b},[x1],#16           // load packed B0_next4k
        UdotByElement 17, 1, 4, 0
        UdotByElement 19, 1, 4, 1
        sub     x3,x3,#1
        cbz     x3,ComputeBlockLoopFinishM4
        ldr     d4,[x0],#32                 // load packed A0.l for next iteration
        UdotByElement 21, 1, 5, 0
        UdotByElement 23, 1, 5, 1
        ld1     {v1.16b},[x1],#16           // load packed B1_next4k
        UdotByElement 24, 0, 6, 0
        UdotByElement 26, 0, 6, 1
        ldur    d5,[x0,#-24]                // load packed A0.h for next iteration
        UdotByElement 28, 0, 7, 0
        UdotByElement 30, 0, 7, 1
        ld1     {v0.16b},[x1],#16           // load packed B0 for next iteration
        UdotByElement 25, 1, 6, 0
        UdotByElement 27, 1, 6, 1
        ldur    d6,[x0,#-16]                // load packed A1.l for next iteration
        UdotByElement 29, 1, 7, 0
        UdotByElement 31, 1, 7, 1
        b       ComputeBlockLoopM4

ComputeBlockLoopFinishM4
        UdotByElement 21, 1, 5, 0
        UdotByElement 23, 1, 5, 1
        ld1     {v1.16b},[x1],#16           // load packed B1_next4k
        UdotByElement 24, 0, 6, 0
        UdotByElement 26, 0, 6, 1
        UdotByElement 28, 0, 7, 0
        UdotByElement 30, 0, 7, 1
        UdotByElement 25, 1, 6, 0
        UdotByElement 27, 1, 6, 1
        UdotByElement 29, 1, 7, 0
        UdotByElement 31, 1, 7, 1
        add     x10,x2,x6,lsl #2            // compute output row 2
        add     v16.4s,v16.4s,v24.4s        // fold high results into low results
        add     v18.4s,v18.4s,v26.4s
        add     v20.4s,v20.4s,v28.4s
        add     v22.4s,v22.4s,v30.4s
        add     x11,x10,x6,lsl #2           // compute output row 3
        add     v17.4s,v17.4s,v25.4s
        add     v19.4s,v19.4s,v27.4s
        add     v21.4s,v21.4s,v29.4s
        add     v23.4s,v23.4s,v31.4s
        add     x12,x11,x6,lsl #2           // compute output row 4
        subs    x5,x5,#8                    // adjust CountN remaining
        blo     StoreOutputPartialM4
        cbnz    x13,SkipAccumulateOutputM4
        ldp     q0,q1,[x2]
        ldp     q2,q3,[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v17.4s,v17.4s,v1.4s
        ldp     q4,q5,[x11]
        add     v18.4s,v18.4s,v2.4s
        add     v19.4s,v19.4s,v3.4s
        ldp     q6,q7,[x12]
        add     v20.4s,v20.4s,v4.4s
        add     v21.4s,v21.4s,v5.4s
        add     v22.4s,v22.4s,v6.4s
        add     v23.4s,v23.4s,v7.4s

SkipAccumulateOutputM4
        stp     q16,q17,[x2],#32
        stp     q18,q19,[x10]
        stp     q20,q21,[x11]
        stp     q22,q23,[x12]
        cbnz    x5,ProcessNextColumnLoopM4

ExitKernelM4
        mov     x0,#4                       // return number of rows handled
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#64
        EPILOG_RETURN

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

StoreOutputPartialM4
        cbz     x13,StoreOutputPartialAddModeM4

StoreOutputPartialZeroModeM4
        tbz     x5,#2,StoreOutputPartial2ZeroModeM4
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b
        st1     {v20.4s},[x11],#16
        mov     v20.16b,v21.16b
        st1     {v22.4s},[x12],#16
        mov     v22.16b,v23.16b

StoreOutputPartial2ZeroModeM4
        tbz     x5,#1,StoreOutputPartial1ZeroModeM4
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]
        st1     {v20.2s},[x11],#8
        dup     v20.4s,v20.s[2]
        st1     {v22.2s},[x12],#8
        dup     v22.4s,v22.s[2]

StoreOutputPartial1ZeroModeM4
        tbz     x5,#0,ExitKernelM4
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        st1     {v20.s}[0],[x11]
        st1     {v22.s}[0],[x12]
        b       ExitKernelM4

StoreOutputPartialAddModeM4
        tbz     x5,#2,StoreOutputPartial2AddModeM4
        ld1     {v0.4s},[x2]
        ld1     {v1.4s},[x10]
        ld1     {v2.4s},[x11]
        ld1     {v3.4s},[x12]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b
        add     v20.4s,v20.4s,v2.4s
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.4s},[x11],#16
        mov     v20.16b,v21.16b
        st1     {v22.4s},[x12],#16
        mov     v22.16b,v23.16b

StoreOutputPartial2AddModeM4
        tbz     x5,#1,StoreOutputPartial1AddModeM4
        ld1     {v0.2s},[x2]
        ld1     {v1.2s},[x10]
        ld1     {v2.2s},[x11]
        ld1     {v3.2s},[x12]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]
        add     v20.4s,v20.4s,v2.4s
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.2s},[x11],#8
        dup     v20.4s,v20.s[2]
        st1     {v22.2s},[x12],#8
        dup     v22.4s,v22.s[2]

StoreOutputPartial1AddModeM4
        tbz     x5,#0,ExitKernelM4
        ld1     {v0.s}[0],[x2]
        ld1     {v1.s}[0],[x10]
        add     v16.4s,v16.4s,v0.4s
        ld1     {v2.s}[0],[x11]
        add     v18.4s,v18.4s,v1.4s
        ld1     {v3.s}[0],[x12]
        add     v20.4s,v20.4s,v2.4s
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        add     v22.4s,v22.4s,v3.4s
        st1     {v20.s}[0],[x11]
        st1     {v22.s}[0],[x12]
        b       ExitKernelM4


//
// Process 2 rows of the matrices.
//

ProcessNextColumnLoopM2
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        mov     x0,x14                      // reload matrix A
        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]
        mov     x3,x15                      // reload PackedCountK
        ld1     {v3.4s},[x8],#16            // load ColumnSumBuffer[4]
        cbz     x9,SkipScaleByZeroPointBM2
        ld1     {v30.4s},[x9],#16           // load ZeroPointB[0]
        ld1     {v31.4s},[x9],#16           // load ZeroPointB[4]
        mul     v16.4s,v30.4s,v8.4s
        mul     v18.4s,v30.4s,v9.4s
        mul     v17.4s,v31.4s,v8.4s
        mul     v19.4s,v31.4s,v9.4s
        ldr     d4,[x0],#8                  // load packed A0.l
        add     v16.4s,v2.4s,v16.4s
        add     v18.4s,v2.4s,v18.4s
        ldr     d5,[x0],#8                  // load packed A0.h
        add     v17.4s,v3.4s,v17.4s
        add     v19.4s,v3.4s,v19.4s
        b       ComputeBlockLoopM2

SkipScaleByZeroPointBM2
        ldr     d4,[x0],#8                  // load packed A0.l
        add     v16.4s,v2.4s,v8.4s
        add     v18.4s,v2.4s,v9.4s
        ldr     d5,[x0],#8                  // load packed A0.h
        add     v17.4s,v3.4s,v8.4s
        add     v19.4s,v3.4s,v9.4s

ComputeBlockLoopM2
        sub     x3,x3,#1
        ld1     {v6.16b},[x1],#16           // load packed B0 next 4 k
        ld1     {v7.16b},[x1],#16           // load packed B1 next 4 k
        UdotByElement 16, 0, 4, 0
        UdotByElement 17, 1, 4, 0
        UdotByElement 18, 0, 4, 1
        UdotByElement 19, 1, 4, 1
        cbz     x3,ComputeBlockLoopFinishM2
        ldr     d4,[x0],#8                  // load packed A0.l for next iter
        ld1     {v0.16b},[x1],#16           // load packed B0 for next iter
        ld1     {v1.16b},[x1],#16           // load packed B1 for next iter
        UdotByElement 16, 6, 5, 0
        UdotByElement 17, 7, 5, 0
        UdotByElement 18, 6, 5, 1
        UdotByElement 19, 7, 5, 1
        ldr     d5,[x0],#8                  // load packed A0.h for next iter
        b       ComputeBlockLoopM2

ComputeBlockLoopFinishM2
        add     x10,x2,x6,lsl #2            // compute output row 2
        subs    x5,x5,#8                    // adjust CountN remaining
        UdotByElement 16, 6, 5, 0
        UdotByElement 17, 7, 5, 0
        UdotByElement 18, 6, 5, 1
        UdotByElement 19, 7, 5, 1
        blo     StoreOutputPartialM2
        cbnz    x13,SkipAccumulateOutputM2
        ldp     q0,q1,[x2]
        ldp     q2,q3,[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v17.4s,v17.4s,v1.4s
        add     v18.4s,v18.4s,v2.4s
        add     v19.4s,v19.4s,v3.4s

SkipAccumulateOutputM2
        stp     q16,q17,[x2],#32
        stp     q18,q19,[x10]
        cbnz    x5,ProcessNextColumnLoopM2

ExitKernelM2
        mov     x0,#2                       // return number of rows handled
        ldp     d8,d9,[sp],#64
        EPILOG_RETURN

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

StoreOutputPartialM2
        cbz     x13,StoreOutputPartialAddModeM2

StoreOutputPartialZeroModeM2
        tbz     x5,#2,StoreOutputPartial2ZeroModeM2
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b

StoreOutputPartial2ZeroModeM2
        tbz     x5,#1,StoreOutputPartial1ZeroModeM2
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]

StoreOutputPartial1ZeroModeM2
        tbz     x5,#0,ExitKernelM2
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        b       ExitKernelM2

StoreOutputPartialAddModeM2
        tbz     x5,#2,StoreOutputPartial2AddModeM2
        ld1     {v0.4s},[x2]
        ld1     {v1.4s},[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down
        st1     {v18.4s},[x10],#16
        mov     v18.16b,v19.16b

StoreOutputPartial2AddModeM2
        tbz     x5,#1,StoreOutputPartial1AddModeM2
        ld1     {v0.2s},[x2]
        ld1     {v1.2s},[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v18.2s},[x10],#8
        dup     v18.4s,v18.s[2]

StoreOutputPartial1AddModeM2
        tbz     x5,#0,ExitKernelM2
        ld1     {v0.s}[0],[x2]
        ld1     {v1.s}[0],[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v18.4s,v18.4s,v1.4s
        st1     {v16.s}[0],[x2]
        st1     {v18.s}[0],[x10]
        b       ExitKernelM2

//
// Process 1 row of the matrices.
//

ProcessNextColumnLoopM1
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        mov     x0,x14                      // reload matrix A
        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer0
        mov     x3,x15                      // reload PackedCountK
        ld1     {v3.4s},[x8],#16            // load ColumnSumBuffer1
        cbz     x9,SkipScaleByZeroPointBM1
        ld1     {v30.4s},[x9],#16           // load ZeroPointB0
        ld1     {v31.4s},[x9],#16           // load ZeroPointB1
        mul     v16.4s,v30.4s,v8.4s
        mul     v17.4s,v31.4s,v8.4s
        ldr     d4,[x0],#8                  // load packed A0
        ld1     {v6.16b},[x1],#16           // load packed B0 next 4 k
        ld1     {v7.16b},[x1],#16           // load packed B1 next 4 k
        add     v16.4s,v2.4s,v16.4s
        add     v17.4s,v3.4s,v17.4s
        b       ComputeBlockLoopM1

SkipScaleByZeroPointBM1
        ldr     d4,[x0],#8                  // load packed A0
        ld1     {v6.16b},[x1],#16           // load packed B0 next 4 k
        ld1     {v7.16b},[x1],#16           // load packed B1 next 4 k
        add     v16.4s,v2.4s,v8.4s
        add     v17.4s,v3.4s,v8.4s

ComputeBlockLoopM1
        sub     x3,x3,#1
        UdotByElement 16, 0, 4, 0
        UdotByElement 17, 1, 4, 0
        cbz     x3,ComputeBlockLoopFinishM1
        ld1     {v0.16b},[x1],#16           // load packed B0 for next iter
        ld1     {v1.16b},[x1],#16           // load packed B1 for next iter
        UdotByElement 16, 6, 4, 1
        UdotByElement 17, 7, 4, 1
        ldr     d4,[x0],#8                  // load packed A0 for next iter
        ld1     {v6.16b},[x1],#16           // load packed B0 next 4 k for next iter
        ld1     {v7.16b},[x1],#16           // load packed B1 next 4 k for next iter
        b       ComputeBlockLoopM1

ComputeBlockLoopFinishM1
        subs    x5,x5,#8                    // adjust CountN remaining
        UdotByElement 16, 6, 4, 1
        UdotByElement 17, 7, 4, 1
        blo     StoreOutputPartialM1
        cbnz    x13,SkipAccumulateOutputM1
        ldp     q0,q1,[x2]
        add     v16.4s,v16.4s,v0.4s
        add     v17.4s,v17.4s,v1.4s

SkipAccumulateOutputM1
        stp     q16,q17,[x2],#32
        cbnz    x5,ProcessNextColumnLoopM1

ExitKernelM1
        mov     x0,#1                       // return number of rows handled
        ldp     d8,d9,[sp],#64
        EPILOG_RETURN

//
// Store the partial 1 to 7 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

StoreOutputPartialM1
        cbz     x13,StoreOutputPartialAddModeM1

StoreOutputPartialZeroModeM1
        tbz     x5,#2,StoreOutputPartial2ZeroModeM1
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down

StoreOutputPartial2ZeroModeM1
        tbz     x5,#1,StoreOutputPartial1ZeroModeM1
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down

StoreOutputPartial1ZeroModeM1
        tbz     x5,#0,ExitKernelM1
        st1     {v16.s}[0],[x2]
        b       ExitKernelM1

StoreOutputPartialAddModeM1
        tbz     x5,#2,StoreOutputPartial2AddModeM1
        ld1     {v0.4s},[x2]
        add     v16.4s,v16.4s,v0.4s
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v17.16b             // shift remaining elements down

StoreOutputPartial2AddModeM1
        tbz     x5,#1,StoreOutputPartial1AddModeM1
        ld1     {v0.2s},[x2]
        add     v16.4s,v16.4s,v0.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down

StoreOutputPartial1AddModeM1
        tbz     x5,#0,ExitKernelM1
        ld1     {v0.s}[0],[x2]
        add     v16.4s,v16.4s,v0.4s
        st1     {v16.s}[0],[x2]
        b       ExitKernelM1

        NESTED_END MlasGemmU8X8KernelUdot

        END
