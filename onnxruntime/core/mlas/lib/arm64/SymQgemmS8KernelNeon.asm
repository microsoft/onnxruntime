/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SymQgemmS8KernelNeon.asm

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM), where the right hand side is symmetrically quantized,
    i.e. zero point being zero.

    This kernel only requires prepacking of the right hand side, which is usually
    constant. When the packed right hand side is cached, we achieves higher performance
    by avoid packing all together.

--*/

#include "kxarm64.h"

//
// Stack frame layout for the S8S8 kernel.
//

#define  SQGemmS8Frame_SavedNeonRegisters     (8 * 8)
#define  SQGemmS8Frame_SavedRegisters         SQGemmS8Frame_SavedNeonRegisters
#define  SQGemmS8Frame_ColumnSumBuffer        0 + SQGemmS8Frame_SavedRegisters

        TEXTAREA

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmQuantCopyPackB<MLAS_GEMM_X8S8_KERNEL_NEON>.

    C (x2) - Supplies the address of matrix C.

    PackedCountK (x3) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.

    CountM (x4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (x5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc (x6) - Supplies the first dimension of matrix C.

    lda (x7) - Supplies the first dimension of matrix A.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.


Return Value:

    Returns the number of rows handled.

--*/

        NESTED_ENTRY MlasSymQgemmS8KernelNeon

        PROLOG_SAVE_REG_PAIR d8,d9,#-SQGemmS8Frame_SavedRegisters!
        PROLOG_SAVE_REG_PAIR d10,d11,#16
        PROLOG_SAVE_REG_PAIR d12,d13,#32
        PROLOG_SAVE_REG_PAIR d14,d15,#48
        ldr     x13,[sp,#SQGemmS8Frame_ColumnSumBuffer]
        mov     x14,x0
        mov     x15,x3
        cmp     x4,#1                       // CountM == 1?
        beq     M1_ProcessLoop
        cmp     x4,#4                       // CountM < 4?
        blo     M2_ProcessLoop

//
// Process 4 rows of the matrices.
//                                            B 16x4
//                                      ----------------------------------------
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v8.b[0]   v9.b[0]   v10.b[0]  v11.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v8.b[7]   v9.b[7]   v10.b[7]  v11.b[7]|
//            A 4x16                    ----------------------------------------
// -----------------------------------  ----------------------------------------
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v20.4s    v21.4s    v22.4s    v23.4s  |
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v24.4s    v25.4s    v26.4s    v27.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v28.4s    v29.4s    v30.4s    v31.4s  |
// -----------------------------------  ----------------------------------------
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)
//

M4_ProcessNextColumnLoop
        mov     x0,x14                      // reload matrix A0
        mov     x3,x15                      // reload PackedCountK
        ldr     d0,[x0],#8                  // Load A0
        add     x9,x14,x7                   // A1
        ldr     d2,[x0],#8                  // Load A0
        movi    v16.4s,#0
        movi    v17.4s,#0
        ldp     d4,d8,[x1],#64              // B
        movi    v18.4s,#0
        movi    v19.4s,#0
        ldp     d5,d9,[x1,#-48]
        movi    v20.4s,#0
        movi    v21.4s,#0
        ldp     d6,d10,[x1,#-32]
        movi    v22.4s,#0
        movi    v23.4s,#0
        ldp     d7,d11,[x1,#-16]
        movi    v24.4s,#0
        movi    v25.4s,#0
        add     x10,x9,x7                   // A2
        ldp     d1,d3,[x9],#16              // Load A1
        movi    v26.4s,#0
        movi    v27.4s,#0
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0
        add     x11,x10,x7                  // A3

M4_ComputeBlockLoop
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ldp     d0,d2,[x10],#16             // Load A2
        sadalp  v16.4s,v12.8h
        sadalp  v17.4s,v13.8h
        sadalp  v18.4s,v14.8h
        sadalp  v19.4s,v15.8h
        sub     x3,x3,#1
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal   v12.8h,v3.8b,v8.8b
        smlal   v13.8h,v3.8b,v9.8b
        smlal   v14.8h,v3.8b,v10.8b
        smlal   v15.8h,v3.8b,v11.8b
        ldp     d1,d3,[x11],#16             // Load A3
        sadalp  v20.4s,v12.8h
        sadalp  v21.4s,v13.8h
        sadalp  v22.4s,v14.8h
        sadalp  v23.4s,v15.8h
        cbz     x3,M4_ComputeBlockLoopFinish
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ldp     d0,d2,[x0],#16              // Load A0 next iter
        sadalp  v24.4s,v12.8h
        sadalp  v25.4s,v13.8h
        sadalp  v26.4s,v14.8h
        sadalp  v27.4s,v15.8h
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal   v12.8h,v3.8b,v8.8b
        ldp     d4,d8,[x1],#64              // B
        smlal   v13.8h,v3.8b,v9.8b
        ldp     d5,d9,[x1,#-48]
        smlal   v14.8h,v3.8b,v10.8b
        ldp     d6,d10,[x1,#-32]
        smlal   v15.8h,v3.8b,v11.8b
        ldp     d7,d11,[x1,#-16]
        sadalp  v28.4s,v12.8h
        ldp     d1,d3,[x9],#16              // Load A1 next iter
        sadalp  v29.4s,v13.8h
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h
        b       M4_ComputeBlockLoop

M4_ComputeBlockLoopFinish
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ld1     {v2.4s},[x13],#16           // load ColumnSumBuffer[0]
        sadalp  v24.4s,v12.8h
        sadalp  v25.4s,v13.8h
        sadalp  v26.4s,v14.8h
        sadalp  v27.4s,v15.8h
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal   v12.8h,v3.8b,v8.8b
        smlal   v13.8h,v3.8b,v9.8b
        smlal   v14.8h,v3.8b,v10.8b
        smlal   v15.8h,v3.8b,v11.8b
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

        // accumulator += column sum B 
        add     v16.4s,v16.4s,v2.4s
        add     v20.4s,v20.4s,v2.4s
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v2.4s

M4_StoreOutput
        add     x10,x2,x6,lsl #2
        add     x11,x10,x6,lsl #2
        add     x12,x11,x6,lsl #2
        subs    x5,x5,#4                    // adjust CountN remaining
        blo     M4_StoreOutputPartial
        st1     {v16.4s},[x2],#16
        st1     {v20.4s},[x10]
        st1     {v24.4s},[x11]
        st1     {v28.4s},[x12]
        cbnz    x5,M4_ProcessNextColumnLoop

M4_ExitKernel
        mov     x0,#4                       // return number of rows handled
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

M4_StoreOutputPartial

M4_StoreOutputPartial_ZeroMode
        tbz     x5,#1,M4_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]
        st1     {v24.2s},[x11],#8
        dup     v24.4s,v24.s[2]
        st1     {v28.2s},[x12],#8
        dup     v28.4s,v28.s[2]

M4_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,M4_ExitKernel
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        st1     {v24.s}[0],[x11]
        st1     {v28.s}[0],[x12]
        b       M4_ExitKernel

//
// Process 2 rows of the matrices.
//
// Column Sum v2.s[0] v2.s[4]
// Each row sum replicated to all 4 elements of a vector register 
// v30 v31
//                                            B 16x4
//                                      ----------------------------------------
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v24.b[0]  v25.b[0]  v26.b[0]  v27.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v24.b[7]  v25.b[7]  v26.b[7]  v27.b[7]|
//            A 2x16                    ----------------------------------------
// -----------------------------------  ----------------------------------------
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v20.4s    v21.4s    v22.4s    v23.4s  |
// -----------------------------------  ----------------------------------------
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)

M2_ProcessLoop

M2_ProcessNextColumnLoop
        ldp     d4,d24,[x1],#16             // B
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        ldp     d0,d2,[x0],#16              // Load A0
        add     x9,x14,x7                   // A1
        movi    v16.4s,#0
        movi    v17.4s,#0
        ldp     d5,d25,[x1],#16
        movi    v18.4s,#0
        movi    v19.4s,#0
        ldp     d6,d26,[x1],#16
        movi    v20.4s,#0
        movi    v21.4s,#0
        ldp     d7,d27,[x1],#16
        movi    v22.4s,#0
        movi    v23.4s,#0
        ldp     d1,d3,[x9],#16              // Load A1

M2_ComputeBlockLoop
        sub     x3,x3,#1
        smull   v28.8h,v0.8b,v4.8b
        smull   v29.8h,v0.8b,v5.8b
        smull   v30.8h,v0.8b,v6.8b
        smull   v31.8h,v0.8b,v7.8b
        cbz     x3,M2_ComputeBlockLoopFinish
        smlal   v28.8h,v2.8b,v24.8b
        smlal   v29.8h,v2.8b,v25.8b
        smlal   v30.8h,v2.8b,v26.8b
        smlal   v31.8h,v2.8b,v27.8b
        ldp     d0,d2,[x0],#16              // Load A0
        sadalp  v16.4s,v28.8h
        sadalp  v17.4s,v29.8h
        sadalp  v18.4s,v30.8h
        sadalp  v19.4s,v31.8h
        smull   v28.8h,v1.8b,v4.8b
        smull   v29.8h,v1.8b,v5.8b
        smull   v30.8h,v1.8b,v6.8b
        smull   v31.8h,v1.8b,v7.8b
        smlal   v28.8h,v3.8b,v24.8b
        ldp     d4,d24,[x1],#16             // B
        smlal   v29.8h,v3.8b,v25.8b
        ldp     d5,d25,[x1],#16
        smlal   v30.8h,v3.8b,v26.8b
        ldp     d6,d26,[x1],#16
        smlal   v31.8h,v3.8b,v27.8b
        ldp     d7,d27,[x1],#16
        sadalp  v20.4s,v28.8h
        ldp     d1,d3,[x9],#16              // Load A1
        sadalp  v21.4s,v29.8h
        sadalp  v22.4s,v30.8h
        sadalp  v23.4s,v31.8h
        b       M2_ComputeBlockLoop

M2_ComputeBlockLoopFinish
        ld1     {v0.4s},[x13],#16           // load ColumnSumBuffer[0]
        smlal   v28.8h,v2.8b,v24.8b
        smlal   v29.8h,v2.8b,v25.8b
        smlal   v30.8h,v2.8b,v26.8b
        smlal   v31.8h,v2.8b,v27.8b
        sadalp  v16.4s,v28.8h
        sadalp  v17.4s,v29.8h
        sadalp  v18.4s,v30.8h
        sadalp  v19.4s,v31.8h
        smull   v28.8h,v1.8b,v4.8b
        smull   v29.8h,v1.8b,v5.8b
        smull   v30.8h,v1.8b,v6.8b
        smull   v31.8h,v1.8b,v7.8b
        smlal   v28.8h,v3.8b,v24.8b
        smlal   v29.8h,v3.8b,v25.8b
        smlal   v30.8h,v3.8b,v26.8b
        smlal   v31.8h,v3.8b,v27.8b
        sadalp  v20.4s,v28.8h
        sadalp  v21.4s,v29.8h
        sadalp  v22.4s,v30.8h
        sadalp  v23.4s,v31.8h
        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s
        addp    v20.4s,v20.4s,v21.4s
        addp    v22.4s,v22.4s,v23.4s
        addp    v16.4s,v16.4s,v18.4s
        addp    v20.4s,v20.4s,v22.4s

        // accumulator = column sum B 
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v0.4s

M2_StoreOutput
        add     x10,x2,x6,lsl #2
        subs    x5,x5,#4                    // adjust CountN remaining
        blo     M2_StoreOutputPartial
        st1     {v16.4s},[x2],#16
        st1     {v20.4s},[x10]
        cbnz    x5,M2_ProcessNextColumnLoop

M2_ExitKernel
        mov     x0,#2                       // return number of rows handled
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

M2_StoreOutputPartial

M2_StoreOutputPartial_ZeroMode
        tbz     x5,#1,M2_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v20.2s},[x10],#8
        dup     v20.4s,v20.s[2]

M2_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,M2_ExitKernel
        st1     {v16.s}[0],[x2]
        st1     {v20.s}[0],[x10]
        b       M2_ExitKernel

//
// Process 1 row of the matrices.
//
// Column Sum v2.s[0] v2.s[4]
// row sum replicated to all 4 elements of a vector register 
// v31 
//                                            B 16x4
//                                      ----------------------------------------
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v24.b[0]  v25.b[0]  v26.b[0]  v27.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v24.b[7]  v25.b[7]  v26.b[7]  v27.b[7]|
//            A 1x16                    ----------------------------------------
// -----------------------------------  ----------------------------------------
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// -----------------------------------  ----------------------------------------
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)
//
M1_ProcessLoop

M1_ProcessNextColumnLoop
        ldp     d4,d24,[x1],#16             // B
        ldp     d5,d25,[x1],#16
        ldp     d6,d26,[x1],#16
        ldp     d7,d27,[x1],#16
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        ldp     d0,d2,[x0],#16              // A0
        movi    v16.4s,#0
        movi    v17.4s,#0
        movi    v18.4s,#0
        movi    v19.4s,#0

M1_ComputeBlockLoop
        sub     x3,x3,#1
        smull   v20.8h,v0.8b,v4.8b
        smull   v21.8h,v0.8b,v5.8b
        cbz    x3,M1_ComputeBlockLoopFinish
        smull   v22.8h,v0.8b,v6.8b
        smull   v23.8h,v0.8b,v7.8b
        smlal   v20.8h,v2.8b,v24.8b
        ldp     d4,d24,[x1],#16             // B
        smlal   v21.8h,v2.8b,v25.8b
        ldp     d5,d25,[x1],#16
        smlal   v22.8h,v2.8b,v26.8b
        ldp     d6,d26,[x1],#16
        smlal   v23.8h,v2.8b,v27.8b
        ldp     d0,d2,[x0],#16              // A0
        sadalp  v16.4s,v20.8h
        sadalp  v17.4s,v21.8h
        ldp     d7,d27,[x1],#16
        sadalp  v18.4s,v22.8h
        sadalp  v19.4s,v23.8h
        b       M1_ComputeBlockLoop

M1_ComputeBlockLoopFinish
        ld1     {v4.4s},[x13],#16           // load ColumnSumBuffer[0]
        smull   v22.8h,v0.8b,v6.8b
        smull   v23.8h,v0.8b,v7.8b
        smlal   v20.8h,v2.8b,v24.8b
        smlal   v21.8h,v2.8b,v25.8b
        smlal   v22.8h,v2.8b,v26.8b
        smlal   v23.8h,v2.8b,v27.8b
        sadalp  v16.4s,v20.8h
        sadalp  v17.4s,v21.8h
        sadalp  v18.4s,v22.8h
        sadalp  v19.4s,v23.8h
        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s
        addp    v16.4s,v16.4s,v18.4s

        // accumulator += column sum B
        add     v16.4s,v16.4s,v4.4s

M1_StoreOutput
        subs    x5,x5,#4                    // adjust CountN remaining
        blo     M1_StoreOutputPartial
        st1     {v16.4s},[x2],#16
        cbnz    x5,M1_ProcessNextColumnLoop

M1_ExitKernel
        mov     x0,#1                       // return number of rows handled
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

M1_StoreOutputPartial

M1_StoreOutputPartial_ZeroMode
        tbz     x5,#1,M1_StoreOutputPartial1_ZeroMode
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down

M1_StoreOutputPartial1_ZeroMode
        tbz     x5,#0,M1_ExitKernel
        st1     {v16.s}[0],[x2]
        b       M1_ExitKernel

        NESTED_END MlasSymQgemmS8KernelNeon

        END
