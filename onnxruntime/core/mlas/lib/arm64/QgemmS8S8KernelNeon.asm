/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmS8S8KernelNeon.asm

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

--*/

#include "kxarm64.h"

//
// Stack frame layout for the S8S8 kernel.
//

#define  GemmS8S8KernelFrame_SavedNeonRegisters    (4 * 8)
#define  GemmS8S8KernelFrame_SavedRegisters            GemmS8S8KernelFrame_SavedNeonRegisters
#define  GemmS8S8KernelFrame_ColumnSumBuffer       0 + GemmS8S8KernelFrame_SavedRegisters
#define  GemmS8S8KernelFrame_ZeroPointB            8 + GemmS8S8KernelFrame_SavedRegisters
#define  GemmS8S8KernelFrame_ZeroMode             16 + GemmS8S8KernelFrame_SavedRegisters

        TEXTAREA

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackA<MLAS_GEMM_S8S8_KERNEL_NEON>.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackB<MLAS_GEMM_S8S8_KERNEL_NEON>.

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

        LEAF_ENTRY MlasGemmS8S8KernelNeon

        stp     d8,d9,[sp,#-32]!
        stp     d10,d11,[sp,#16]
        ldr     x8,[sp,#GemmS8S8KernelFrame_ColumnSumBuffer]
        ldr     x9,[sp,#GemmS8S8KernelFrame_ZeroPointB]
        ldrb    w13,[sp,#GemmS8S8KernelFrame_ZeroMode]
        mov     x14,x0
        ld1     {v11.4s},[x7]
        mov     x15,x3
        dup     v8.4s,v11.s[0]             // broadcast row fixups
        cmp     x4,#1                       // CountM == 1?
        beq     GemmS8S8_M1_ProcessNextColumnLoop
        dup     v9.4s,v11.s[1]
        cmp     x4,#4                       // CountM < 4?
        blo     GemmS8S8_M2_ProcessNextColumnLoop
        dup     v10.4s,v11.s[2]
        dup     v11.4s,v11.s[3]

//
// Process 4 rows of the matrices.
//

GemmS8S8_M4_ProcessNextColumnLoop
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        ldr     q0,[x0],#64                 // A0
        movi    v16.4s,#0
        movi    v17.4s,#0
        ldr     q4,[x1],#64                 // B
        movi    v18.4s,#0
        movi    v19.4s,#0
        ldur    q5,[x1,#-48]
        movi    v20.4s,#0
        movi    v21.4s,#0
        ldur    q6,[x1,#-32]
        movi    v22.4s,#0
        movi    v23.4s,#0
        ldur    q7,[x1,#-16]
        movi    v24.4s,#0
        movi    v25.4s,#0
        movi    v26.4s,#0
        movi    v27.4s,#0
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0

GemmS8S8_M4_ComputeBlockLoop
        ldur    q1,[x0,#-48]
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal2  v12.8h,v0.16b,v4.16b
        smlal2  v13.8h,v0.16b,v5.16b
        smlal2  v14.8h,v0.16b,v6.16b
        smlal2  v15.8h,v0.16b,v7.16b
        sadalp  v16.4s,v12.8h
        sadalp  v17.4s,v13.8h
        sadalp  v18.4s,v14.8h
        sadalp  v19.4s,v15.8h
        ldur    q0,[x0,#-32]
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal2  v12.8h,v1.16b,v4.16b
        smlal2  v13.8h,v1.16b,v5.16b
        smlal2  v14.8h,v1.16b,v6.16b
        smlal2  v15.8h,v1.16b,v7.16b
        sadalp  v20.4s,v12.8h
        sadalp  v21.4s,v13.8h
        sadalp  v22.4s,v14.8h
        sadalp  v23.4s,v15.8h
        ldur    q1,[x0,#-16]
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal2  v12.8h,v0.16b,v4.16b
        smlal2  v13.8h,v0.16b,v5.16b
        smlal2  v14.8h,v0.16b,v6.16b
        smlal2  v15.8h,v0.16b,v7.16b
        sadalp  v24.4s,v12.8h
        sadalp  v25.4s,v13.8h
        sadalp  v26.4s,v14.8h
        sadalp  v27.4s,v15.8h

        sub     x3,x3,#1
        cbz     x3,GemmS8S8_M4_ComputeBlockLoopFinish

        ldr     q0,[x0],#64
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal2  v12.8h,v1.16b,v4.16b
        smlal2  v13.8h,v1.16b,v5.16b
        ldr     q4,[x1],#64                 // B
        smlal2  v14.8h,v1.16b,v6.16b
        smlal2  v15.8h,v1.16b,v7.16b
        ldur    q5,[x1,#-48]
        sadalp  v28.4s,v12.8h
        ldur    q6,[x1,#-32]
        sadalp  v29.4s,v13.8h
        ldur    q7,[x1,#-16]
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h
        b       GemmS8S8_M4_ComputeBlockLoop

GemmS8S8_M4_ComputeBlockLoopFinish

        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal2  v12.8h,v1.16b,v4.16b
        smlal2  v13.8h,v1.16b,v5.16b
        smlal2  v14.8h,v1.16b,v6.16b
        smlal2  v15.8h,v1.16b,v7.16b
        sadalp  v28.4s,v12.8h
        sadalp  v29.4s,v13.8h
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h

        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s
        addp    v20.4s,v20.4s,v21.4s
        addp    v22.4s,v22.4s,v23.4s
        addp    v24.4s,v24.4s,v25.4s
        addp    v26.4s,v26.4s,v27.4s
        addp    v28.4s,v28.4s,v29.4s
        addp    v30.4s,v30.4s,v31.4s

        addp    v16.4s,v16.4s,v18.4s
        addp    v20.4s,v20.4s,v22.4s
        addp    v24.4s,v24.4s,v26.4s
        addp    v28.4s,v28.4s,v30.4s

        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]

        add     v16.4s,v16.4s,v8.4s
        add     v20.4s,v20.4s,v9.4s
        add     v24.4s,v24.4s,v10.4s
        add     v28.4s,v28.4s,v11.4s
        add     v16.4s,v16.4s,v2.4s
        add     v20.4s,v20.4s,v2.4s
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v2.4s

        add     x10,x2,x6,lsl #2
        add     x11,x10,x6,lsl #2
        add     x12,x11,x6,lsl #2

        subs    x5,x5,#4                    // adjust CountN remaining
        blo     GemmS8S8_M4_StoreOutputPartial
        cbnz    x13,GemmS8S8_M4_SkipAccumulateOutput
        ld1     {v0.4s},[x2]
        ld1     {v1.4s},[x10]
        ld1     {v2.4s},[x11]
        ld1     {v3.4s},[x12]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v1.4s
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v3.4s

GemmS8S8_M4_SkipAccumulateOutput
        st1     {v16.4s},[x2],#16
        st1     {v20.4s},[x10]
        st1     {v24.4s},[x11]
        st1     {v28.4s},[x12]
        cbnz    x5,GemmS8S8_M4_ProcessNextColumnLoop

GemmS8S8_M4_ExitKernel
        mov     x0,#4                       // return number of rows handled
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#32
        ret

GemmS8S8_M4_StoreOutputPartial
        cbz     x13,GemmS8S8_M4_StoreOutputPartial_AddMode

GemmS8S8_M4_StoreOutputPartial_ZeroMode
        tbz     x5,#1,GemmS8S8_M4_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]
        st1     {v24.2s},[x11],#8
        dup     v24.4s,v24.s[2]
        st1     {v28.2s},[x12],#8
        dup     v28.4s,v28.s[2]

GemmS8S8_M4_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,GemmS8S8_M4_ExitKernel
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        st1     {v24.s}[0],[x11]
        st1     {v28.s}[0],[x12]
        b       GemmS8S8_M4_ExitKernel

GemmS8S8_M4_StoreOutputPartial_AddMode
        tbz     x5,#1,GemmS8S8_M4_StoreOutputPartial1_AddMode
        ld1     {v0.2s},[x2]
        ld1     {v1.2s},[x10]
        ld1     {v2.2s},[x11]
        ld1     {v3.2s},[x12]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v1.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v3.4s
        st1     {v24.2s},[x11],#8
        dup     v24.4s,v24.s[2]
        st1     {v28.2s},[x12],#8
        dup     v28.4s,v28.s[2]

GemmS8S8_M4_StoreOutputPartial1_AddMode
        tbz     x5,#0,GemmS8S8_M4_ExitKernel
        ld1     {v0.s}[0],[x2]
        ld1     {v1.s}[0],[x10]
        add     v16.4s,v16.4s,v0.4s
        ld1     {v2.s}[0],[x11]
        add     v20.4s,v20.4s,v1.4s
        ld1     {v3.s}[0],[x12]
        add     v24.4s,v24.4s,v2.4s
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        add     v28.4s,v28.4s,v3.4s
        st1     {v24.s}[0],[x11]
        st1     {v28.s}[0],[x12]
        b       GemmS8S8_M4_ExitKernel

//
// Process 2 rows of the matrices.
//

GemmS8S8_M2_ProcessNextColumnLoop
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        movi    v16.4s,#0
        movi    v17.4s,#0
        movi    v18.4s,#0
        movi    v19.4s,#0
        movi    v20.4s,#0
        movi    v21.4s,#0
        movi    v22.4s,#0
        movi    v23.4s,#0

GemmS8S8_M2_ComputeBlockLoop
        ld1     {v4.16b},[x1],#16           // B
        ld1     {v5.16b},[x1],#16
        ld1     {v6.16b},[x1],#16
        ld1     {v7.16b},[x1],#16

        ld1     {v0.16b},[x0],#16           // A0
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal2  v12.8h,v0.16b,v4.16b
        smlal2  v13.8h,v0.16b,v5.16b
        smlal2  v14.8h,v0.16b,v6.16b
        smlal2  v15.8h,v0.16b,v7.16b
        sadalp  v16.4s,v12.8h
        sadalp  v17.4s,v13.8h
        sadalp  v18.4s,v14.8h
        sadalp  v19.4s,v15.8h

        ld1     {v0.16b},[x0],#16           // A1
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal2  v12.8h,v0.16b,v4.16b
        smlal2  v13.8h,v0.16b,v5.16b
        smlal2  v14.8h,v0.16b,v6.16b
        smlal2  v15.8h,v0.16b,v7.16b
        sadalp  v20.4s,v12.8h
        sadalp  v21.4s,v13.8h
        sadalp  v22.4s,v14.8h
        sadalp  v23.4s,v15.8h

        sub     x3,x3,#1
        cbnz    x3,GemmS8S8_M2_ComputeBlockLoop

        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s
        addp    v20.4s,v20.4s,v21.4s
        addp    v22.4s,v22.4s,v23.4s

        addp    v16.4s,v16.4s,v18.4s
        addp    v20.4s,v20.4s,v22.4s

        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]

        add     v16.4s,v16.4s,v8.4s
        add     v20.4s,v20.4s,v9.4s
        add     v16.4s,v16.4s,v2.4s
        add     v20.4s,v20.4s,v2.4s

        add     x10,x2,x6,lsl #2

        subs    x5,x5,#4                    // adjust CountN remaining
        blo     GemmS8S8_M2_StoreOutputPartial
        cbnz    x13,GemmS8S8_M2_SkipAccumulateOutput
        ld1     {v0.4s},[x2]
        ld1     {v1.4s},[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v1.4s

GemmS8S8_M2_SkipAccumulateOutput
        st1     {v16.4s},[x2],#16
        st1     {v20.4s},[x10]
        cbnz    x5,GemmS8S8_M2_ProcessNextColumnLoop

GemmS8S8_M2_ExitKernel
        mov     x0,#2                       // return number of rows handled
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#32
        ret

GemmS8S8_M2_StoreOutputPartial
        cbz     x13,GemmS8S8_M2_StoreOutputPartial_AddMode

GemmS8S8_M2_StoreOutputPartial_ZeroMode
        tbz     x5,#1,GemmS8S8_M2_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]

GemmS8S8_M2_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,GemmS8S8_M2_ExitKernel
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        b       GemmS8S8_M2_ExitKernel

GemmS8S8_M2_StoreOutputPartial_AddMode
        tbz     x5,#1,GemmS8S8_M2_StoreOutputPartial1_AddMode
        ld1     {v0.2s},[x2]
        ld1     {v1.2s},[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v1.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]

GemmS8S8_M2_StoreOutputPartial1_AddMode
        tbz     x5,#0,GemmS8S8_M2_ExitKernel
        ld1     {v0.s}[0],[x2]
        ld1     {v1.s}[0],[x10]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v1.4s
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        b       GemmS8S8_M2_ExitKernel

//
// Process 1 row of the matrices.
//

GemmS8S8_M1_ProcessNextColumnLoop
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        movi    v16.4s,#0
        movi    v17.4s,#0
        movi    v18.4s,#0
        movi    v19.4s,#0

GemmS8S8_M1_ComputeBlockLoop
        ld1     {v4.16b},[x1],#16           // B
        ld1     {v5.16b},[x1],#16
        ld1     {v6.16b},[x1],#16
        ld1     {v7.16b},[x1],#16

        ld1     {v0.16b},[x0],#16           // A0
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal2  v12.8h,v0.16b,v4.16b
        smlal2  v13.8h,v0.16b,v5.16b
        smlal2  v14.8h,v0.16b,v6.16b
        smlal2  v15.8h,v0.16b,v7.16b
        sadalp  v16.4s,v12.8h
        sadalp  v17.4s,v13.8h
        sadalp  v18.4s,v14.8h
        sadalp  v19.4s,v15.8h

        sub     x3,x3,#1
        cbnz    x3,GemmS8S8_M1_ComputeBlockLoop

        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s

        addp    v16.4s,v16.4s,v18.4s

        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]

        add     v16.4s,v16.4s,v8.4s
        add     v16.4s,v16.4s,v2.4s

        subs    x5,x5,#4                    // adjust CountN remaining
        blo     GemmS8S8_M1_StoreOutputPartial
        cbnz    x13,GemmS8S8_M1_SkipAccumulateOutput
        ld1     {v0.4s},[x2]
        add     v16.4s,v16.4s,v0.4s

GemmS8S8_M1_SkipAccumulateOutput
        st1     {v16.4s},[x2],#16
        cbnz    x5,GemmS8S8_M1_ProcessNextColumnLoop

GemmS8S8_M1_ExitKernel
        mov     x0,#1                       // return number of rows handled
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#32
        ret

GemmS8S8_M1_StoreOutputPartial
        cbz     x13,GemmS8S8_M1_StoreOutputPartial_AddMode

GemmS8S8_M1_StoreOutputPartial_ZeroMode:
        tbz     x5,#1,GemmS8S8_M1_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down

GemmS8S8_M1_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,GemmS8S8_M1_ExitKernel
        st1     {v16.s}[0],[x2]
        b       GemmS8S8_M1_ExitKernel

GemmS8S8_M1_StoreOutputPartial_AddMode
        tbz     x5,#1,GemmS8S8_M1_StoreOutputPartial1_AddMode
        ld1     {v0.2s},[x2]
        add     v16.4s,v16.4s,v0.4s
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down

GemmS8S8_M1_StoreOutputPartial1_AddMode
        tbz     x5,#0,GemmS8S8_M1_ExitKernel
        ld1     {v0.s}[0],[x2]
        add     v16.4s,v16.4s,v0.4s
        st1     {v16.s}[0],[x2]
        b       GemmS8S8_M1_ExitKernel

        LEAF_END MlasGemmS8S8KernelNeon

        END
