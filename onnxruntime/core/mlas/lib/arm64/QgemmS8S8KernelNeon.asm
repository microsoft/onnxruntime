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

#define  GemmS8S8KernelFrame_SavedNeonRegisters    (8 * 8)
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

        NESTED_ENTRY MlasGemmS8S8KernelNeon

        PROLOG_SAVE_REG_PAIR d8,d9,#-64!
        PROLOG_SAVE_REG_PAIR d10,d11,#16
        PROLOG_SAVE_REG_PAIR d12,d13,#32
        PROLOG_SAVE_REG_PAIR d14,d15,#48
        ldr     x8,[sp,#GemmS8S8KernelFrame_ColumnSumBuffer]
        ldr     x9,[sp,#GemmS8S8KernelFrame_ZeroPointB]
        ldrb    w13,[sp,#GemmS8S8KernelFrame_ZeroMode]
        mov     x14,x0
        mov     x15,x3
        cmp     x4,#1                       // CountM == 1?
        beq     GemmS8S8_M1_ProcessLoop
        cmp     x4,#4                       // CountM < 4?
        blo     GemmS8S8_M2_ProcessLoop

//
// Process 4 rows of the matrices.
//
//                                            B 16x4
//                                      /--------------------------------------\
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v8.b[0]   v9.b[0]   v10.b[0]  v11.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v8.b[7]   v9.b[7]   v10.b[7]  v11.b[7]|
//            A 4x16                    \--------------------------------------/
// /---------------------------------\  /--------------------------------------\
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v20.4s    v21.4s    v22.4s    v23.4s  |
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v24.4s    v25.4s    v26.4s    v27.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v28.4s    v29.4s    v30.4s    v31.4s  |
// \---------------------------------/  \--------------------------------------/
//
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)

GemmS8S8_M4_ProcessNextColumnLoop
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        ldp     d0,d2,[x0],#64              // A0
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
        ldp     d1,d3,[x0,#-48]
        movi    v26.4s,#0
        movi    v27.4s,#0
        movi    v28.4s,#0
        movi    v29.4s,#0
        movi    v30.4s,#0
        movi    v31.4s,#0

GemmS8S8_M4_ComputeBlockLoop
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ldp     d0,d2,[x0,#-32]
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
        ldp     d1,d3,[x0,#-16]
        sadalp  v20.4s,v12.8h
        sadalp  v21.4s,v13.8h
        sadalp  v22.4s,v14.8h
        sadalp  v23.4s,v15.8h
        cbz     x3,GemmS8S8_M4_ComputeBlockLoopFinish
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ldp     d0,d2,[x0],#64
        sadalp  v24.4s,v12.8h
        sadalp  v25.4s,v13.8h
        sadalp  v26.4s,v14.8h
        sadalp  v27.4s,v15.8h
        smull   v12.8h,v1.8b,v4.8b
        smull   v13.8h,v1.8b,v5.8b
        smull   v14.8h,v1.8b,v6.8b
        smull   v15.8h,v1.8b,v7.8b
        smlal   v12.8h,v3.8b,v8.8b
        ldp     d4,d8,[x1],#64                 // B
        smlal   v13.8h,v3.8b,v9.8b
        ldp     d5,d9,[x1,#-48]
        smlal   v14.8h,v3.8b,v10.8b
        ldp     d6,d10,[x1,#-32]
        smlal   v15.8h,v3.8b,v11.8b
        ldp     d7,d11,[x1,#-16]
        sadalp  v28.4s,v12.8h
        ldp     d1,d3,[x0,#-48]
        sadalp  v29.4s,v13.8h
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h
        b       GemmS8S8_M4_ComputeBlockLoop

GemmS8S8_M4_ComputeBlockLoopFinish
        smull   v12.8h,v0.8b,v4.8b
        smull   v13.8h,v0.8b,v5.8b
        smull   v14.8h,v0.8b,v6.8b
        smull   v15.8h,v0.8b,v7.8b
        ld1     {v0.4s},[x7]
        smlal   v12.8h,v2.8b,v8.8b
        smlal   v13.8h,v2.8b,v9.8b
        smlal   v14.8h,v2.8b,v10.8b
        smlal   v15.8h,v2.8b,v11.8b
        ld1     {v2.4s},[x8],#16            // load ColumnSumBuffer[0]
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
        dup     v8.4s,v0.s[0]              // broadcast row fixups
        dup     v9.4s,v0.s[1]
        dup     v10.4s,v0.s[2]
        dup     v11.4s,v0.s[3]
        cbz     x9,GemmS8S8_M4_SkipScaleByZeroPointB

        // accumulator = zero point B * row sum A + column sum B
        ld1     {v30.4s},[x9],#16           // load ZeroPointB
        mul     v17.4s,v30.4s,v8.4s
        mul     v21.4s,v30.4s,v9.4s
        mul     v25.4s,v30.4s,v10.4s
        mul     v29.4s,v30.4s,v11.4s
        add     v16.4s,v16.4s,v17.4s
        add     v20.4s,v20.4s,v21.4s
        add     v24.4s,v24.4s,v25.4s
        add     v28.4s,v28.4s,v29.4s
        add     v16.4s,v16.4s,v2.4s
        add     v20.4s,v20.4s,v2.4s
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v2.4s
        b GemmS8S8_M4_StoreOutput

GemmS8S8_M4_SkipScaleByZeroPointB
        // accumulator = row sum A + column sum B 
        add     v16.4s,v16.4s,v8.4s
        add     v20.4s,v20.4s,v9.4s
        add     v24.4s,v24.4s,v10.4s
        add     v28.4s,v28.4s,v11.4s
        add     v16.4s,v16.4s,v2.4s
        add     v20.4s,v20.4s,v2.4s
        add     v24.4s,v24.4s,v2.4s
        add     v28.4s,v28.4s,v2.4s

GemmS8S8_M4_StoreOutput
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
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN


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
// Column Sum v2.s[0] v2.s[4]
// Each row sum replicated to all 4 elements of a vector register 
// v30 v31
//                                            B 16x4
//                                      /--------------------------------------\
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v24.b[0]  v25.b[0]  v26.b[0]  v27.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v24.b[7]  v25.b[7]  v26.b[7]  v27.b[7]|
//            A 2x16                    \--------------------------------------/
// /---------------------------------\  /--------------------------------------\
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// |v1.b[0]..v1.b[7] v3.b[0]..v3.b[7]|  |v20.4s    v21.4s    v22.4s    v23.4s  |
// \---------------------------------/  \--------------------------------------/
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)
//
GemmS8S8_M2_ProcessLoop

GemmS8S8_M2_ProcessNextColumnLoop
        ldp     d4,d24,[x1],#16             // B
        mov     x0,x14                      // reload matrix A
        mov     x3,x15                      // reload PackedCountK
        ldp     d0,d2,[x0],#16              // A0
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
        ldp     d1,d3,[x0],#16              // A1

GemmS8S8_M2_ComputeBlockLoop

        sub     x3,x3,#1
        smull   v28.8h,v0.8b,v4.8b
        smull   v29.8h,v0.8b,v5.8b
        smull   v30.8h,v0.8b,v6.8b
        smull   v31.8h,v0.8b,v7.8b
        cbz     x3,GemmS8S8_M2_ComputeBlockLoopFinish
        smlal   v28.8h,v2.8b,v24.8b
        smlal   v29.8h,v2.8b,v25.8b
        smlal   v30.8h,v2.8b,v26.8b
        smlal   v31.8h,v2.8b,v27.8b
        ldp     d0,d2,[x0],#16              // A0
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
        ldp     d1,d3,[x0],#16              // A1
        sadalp  v21.4s,v29.8h
        sadalp  v22.4s,v30.8h
        sadalp  v23.4s,v31.8h
        b       GemmS8S8_M2_ComputeBlockLoop

GemmS8S8_M2_ComputeBlockLoopFinish
        ld1     {v0.4s},[x8],#16            // load ColumnSumBuffer[0]
        smlal   v28.8h,v2.8b,v24.8b
        smlal   v29.8h,v2.8b,v25.8b
        smlal   v30.8h,v2.8b,v26.8b
        smlal   v31.8h,v2.8b,v27.8b
        ldr     d2,[x7]                     // load row sums
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
        dup     v30.4s,v2.s[0]              // broadcast row fixups
        dup     v31.4s,v2.s[1]              // broadcast row fixups
        addp    v16.4s,v16.4s,v18.4s
        addp    v20.4s,v20.4s,v22.4s
        cbz     x9,GemmS8S8_M2_SkipScaleByZeroPointB

        // accumulator = zero point B * row sum A + column sum B
        ld1     {v18.4s},[x9],#16           // load ZeroPointB[0]
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v0.4s
        mul     v17.4s,v18.4s,v30.4s
        mul     v21.4s,v18.4s,v31.4s
        add     v16.4s,v16.4s,v17.4s
        add     v20.4s,v20.4s,v21.4s
        b       GemmS8S8_M2_StoreOutput

GemmS8S8_M2_SkipScaleByZeroPointB
        // accumulator = row sum A + column sum B 
        add     v16.4s,v16.4s,v0.4s
        add     v20.4s,v20.4s,v0.4s
        add     v16.4s,v16.4s,v30.4s
        add     v20.4s,v20.4s,v31.4s

GemmS8S8_M2_StoreOutput
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
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

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
// Column Sum v2.s[0] v2.s[4]
// row sum replicated to all 4 elements of a vector register 
// v31 
//                                            B 16x4
//                                      /--------------------------------------\
//                                      |v4.b[0]   v5.b[0]   v6.b[0]   v7.b[0] |
//                                      |  ...      ...        ...      ...    |
//                                      |v4.b[7]   v5.b[7]   v6.b[7]   v7.b[7] |
//                                      |v24.b[0]  v25.b[0]  v26.b[0]  v27.b[0]|
//                                      |  ...      ...       ...       ...    |
//                                      |v24.b[7]  v25.b[7]  v26.b[7]  v27.b[7]|
//            A 1x16                    \--------------------------------------/
// /---------------------------------\  /--------------------------------------\
// |v0.b[0]..v0.b[7] v2.b[0]..v2.b[7]|  |v16.4s    v17.4s    v18.4s    v19.4s  |
// \---------------------------------/  \--------------------------------------/
//
// Accumulators are horizontally aggregated to the left most register
// for each row. e.g. (v16.s[0], v16.s[1], v16.s[2], v16.s[3]) <- (v16, v17, v18, v19)
//
GemmS8S8_M1_ProcessLoop
        ldr     d31,[x7]
        dup     v31.4s,v31.s[0]              // broadcast row fixups

GemmS8S8_M1_ProcessNextColumnLoop
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

GemmS8S8_M1_ComputeBlockLoop
        sub     x3,x3,#1
        smull   v20.8h,v0.8b,v4.8b
        smull   v21.8h,v0.8b,v5.8b
        cbz     x3,GemmS8S8_M1_ComputeBlockLoopFinish
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
        b       GemmS8S8_M1_ComputeBlockLoop

GemmS8S8_M1_ComputeBlockLoopFinish
        ld1     {v4.4s},[x8],#16            // load ColumnSumBuffer[0]
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
        cbz     x9,GemmS8S8_M1_SkipScaleByZeroPointB

        // accumulator = zero point B * row sum A + column sum B
        ld1     {v30.4s},[x9],#16           // load ZeroPointB[0]
        mul     v17.4s,v30.4s,v31.4s
        add     v16.4s,v16.4s,v17.4s
        add     v16.4s,v16.4s,v4.4s
        b       GemmS8S8_M1_StoreOutput
GemmS8S8_M1_SkipScaleByZeroPointB
        // accumulator = row sum A + column sum B
        add     v16.4s,v16.4s,v31.4s
        add     v16.4s,v16.4s,v4.4s

GemmS8S8_M1_StoreOutput
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
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

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

        NESTED_END MlasGemmS8S8KernelNeon

        END
