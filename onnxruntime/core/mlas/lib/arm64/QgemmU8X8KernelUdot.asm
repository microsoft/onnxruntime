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
//

#define GemmU8XKernelFrame_SavedNeonRegisters       (4 * 8)
#define GemmU8XKernelFrame_SavedRegisters           GemmU8XKernelFrame_SavedNeonRegisters
#define GemmU8XKernelFrame_ColumnSumBuffer          (0 + GemmU8XKernelFrame_SavedRegisters)
#define GemmU8XKernelFrame_ZeroPointB               (8 + GemmU8XKernelFrame_SavedRegisters)
#define GemmU8XKernelFrame_ZeroMode                 (16 + GemmU8XKernelFrame_SavedRegisters)

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

        PROLOG_SAVE_REG_PAIR d8,d9,#-32!
        PROLOG_SAVE_REG_PAIR d10,d11,#16
        ldr     x8,[sp,#GemmU8XKernelFrame_ColumnSumBuffer]
        ldr     x9,[sp,#GemmU8XKernelFrame_ZeroPointB]
        ldrb    w13,[sp,#GemmU8XKernelFrame_ZeroMode]
        mov     x14,x0
        ld1     {v11.4s},[x7]
        mov     x15,x3
        dup     v8.4s,v11.s[0]              // broadcast row fixups
        cmp     x4,#1                       // CountM == 1?
        beq     ProcessNextColumnLoopM1
        dup     v9.4s,v11.s[1]
        cmp     x4,#4                       // CountM < 4?
        blo     ProcessNextColumnLoopM2
        dup     v10.4s,v11.s[2]
        dup     v11.4s,v11.s[3]

//
// Process 4 rows of the matrices.
//

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
        ld1     {v0.16b},[x1],#16           // load packed B0
        UdotByElement 17, 1, 4, 0
        UdotByElement 19, 1, 4, 1
        sub     x3,x3,#1
        cbz     x3,ComputeBlockLoopFinishM4
        ldr     d4,[x0],#32                 // load packed A0.l
        UdotByElement 21, 1, 5, 0
        UdotByElement 23, 1, 5, 1
        ld1     {v1.16b},[x1],#16           // load packed B1
        UdotByElement 24, 0, 6, 0
        UdotByElement 26, 0, 6, 1
        ldur    d5,[x0,#-24]                // load packed A0.h
        UdotByElement 28, 0, 7, 0
        UdotByElement 30, 0, 7, 1
        ld1     {v0.16b},[x1],#16           // load packed B0
        UdotByElement 25, 1, 6, 0
        UdotByElement 27, 1, 6, 1
        ldur    d6,[x0,#-16]                // load packed A1.l
        UdotByElement 29, 1, 7, 0
        UdotByElement 31, 1, 7, 1
        b       ComputeBlockLoopM4

ComputeBlockLoopFinishM4
        UdotByElement 21, 1, 5, 0
        UdotByElement 23, 1, 5, 1
        ld1     {v1.16b},[x1],#16           // load packed B1
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
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#32!
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
        ld1     {v4.16b},[x0],#16           // load packed A0
        add     v16.4s,v2.4s,v16.4s
        add     v18.4s,v2.4s,v18.4s
        add     v17.4s,v3.4s,v17.4s
        add     v19.4s,v3.4s,v19.4s
        b       ComputeBlockLoopM2

SkipScaleByZeroPointBM2
        ld1     {v4.16b},[x0],#16           // load packed A0
        add     v16.4s,v2.4s,v8.4s
        add     v18.4s,v2.4s,v9.4s
        add     v17.4s,v3.4s,v8.4s
        add     v19.4s,v3.4s,v9.4s

ComputeBlockLoopM2
        UdotByElement 16, 0, 4, 0
        UdotByElement 17, 1, 4, 0
        UdotByElement 18, 0, 4, 1
        UdotByElement 19, 1, 4, 1
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        UdotByElement 16, 0, 4, 2
        UdotByElement 17, 1, 4, 2
        UdotByElement 18, 0, 4, 3
        UdotByElement 19, 1, 4, 3
        sub     x3,x3,#1
        cbz     x3,ComputeBlockLoopFinishM2
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        ld1     {v4.16b},[x0],#16           // load packed A0
        b       ComputeBlockLoopM2

ComputeBlockLoopFinishM2
        add     x10,x2,x6,lsl #2            // compute output row 2
        subs    x5,x5,#8                    // adjust CountN remaining
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
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#32!
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
        add     v16.4s,v2.4s,v16.4s
        add     v17.4s,v3.4s,v17.4s
        b       ComputeBlockLoopM1

SkipScaleByZeroPointBM1
        ldr     d4,[x0],#8                  // load packed A0
        add     v16.4s,v2.4s,v8.4s
        add     v17.4s,v3.4s,v8.4s

ComputeBlockLoopM1
        UdotByElement 16, 0, 4, 0
        UdotByElement 17, 1, 4, 0
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        UdotByElement 16, 0, 4, 1
        UdotByElement 17, 1, 4, 1
        sub     x3,x3,#1
        cbz     x3,ComputeBlockLoopFinishM1
        ldr     d4,[x0],#8                  // load packed A0
        ld1     {v0.16b},[x1],#16           // load packed B0
        ld1     {v1.16b},[x1],#16           // load packed B1
        b       ComputeBlockLoopM1

ComputeBlockLoopFinishM1
        subs    x5,x5,#8                    // adjust CountN remaining
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
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#32!
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
