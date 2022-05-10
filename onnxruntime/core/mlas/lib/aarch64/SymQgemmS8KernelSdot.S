/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SymQgemmS8KernelSdot.S

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM), where the right hand side is symmetrically quantized,
    i.e. zero point being zero.

    This kernel only requires prepacking of the right hand side, which is usually
    constant. When the packed right hand side is cached, we achieves higher performance
    by avoid packing all together.

    This version utilizes dot product instructions, and uses 128b loads

--*/

#include "asmmacro.h"
#include "AssembleDotProduct.h"

//
// Stack frame layout d8-d15, x19-x30 need to be preserved if used
//
        .equ    .LGemmS8S8KernelFrame_SavedRegisters,   (4 * 8)
        .equ    .LGemmS8S8KernelFrame_ColumnSumBuffer,  (0 + .LGemmS8S8KernelFrame_SavedRegisters)

        .text

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmQuantCopyPackB<MLAS_SYMM_GEMM_S8S8_KERNEL_SDOT>.

    C (x2) - Supplies the address of matrix C.

    PackedCountK (x3) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.
        Packed K should be 16x

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

        FUNCTION_ENTRY MlasSymQgemmS8KernelSdot

        stp     d8,d9,[sp,#-.LGemmS8S8KernelFrame_SavedRegisters]!
        ldr     x8,[sp,#.LGemmS8S8KernelFrame_ColumnSumBuffer]
        stp     d10,d11,[sp,#16]

        // compute C pointers: x2, x16, x17, x6
        cmp     x4,#2                   // M < 2 ?
        add     x16,x2,x6,lsl #2        // x16 -> C1
        add     x17,x2,x6,lsl #3        // x17 -> C2
        csel    x16,x2,x16,lo           // if M < 2  x16/C1 -> C0
        csel    x17,x16,x17,ls          // if M <= 2  x17/C2 -> C1
        cmp     x4,#4                   // M < 4 ?
        mov     x12,#4                  // set max M to 4
        add     x6,x16,x6,lsl #3        // x6 -> C3
        mov     x9,x0                   // save A0
        mov     x10,x3                  // save K
        csel    x6,x17,x6,lo            // if M < 4  x6/C3 -> C2
        csel    x4,x12,x4,hi            // if M > 4  M = 4;

// Register Usage
//                                            B (x1) -> 4x16
//                        ----------------------------------------------------------------------------
//                        |v4.b[0]..v4.b[12] v5.b[0]..v5.b[12]  v6.b[0]..v6.b[12]   v7.b[0]..v7.b[12]|
//                        |  ...      ...     ...       ...       ...       ...       ...     ...    |
//                        |v4.b[3]..v4.b[15] v5.b[3]..v5.b[15]  v6.b[3]..v6.b[15]   v7.b[3]..v7.b[15]|
//            A 4x4       ----------------------------------------------------------------------------
//     ------------------ ----------------------------------------------------------------------------
// x0  |v0.b[0]..v0.b[3]| |v16.s[0]_v16.s[3] v20.s[0]_v20.s[3]  v24.s[0]_v24.s[3]   v28.s[0]_v28.s[3]| x2
// x12 |v1.b[0]..v1.b[3]| |v17.s[0]_v17.s[3] v21.s[0]_v21.s[3]  v25.s[0]_v25.s[3]   v29.s[0]_v29.s[3]| x16
// x13 |v2.b[0]..v2.b[3]| |v18.s[0]_v18.s[3] v22.s[0]_v22.s[3]  v26.s[0]_v26.s[3]   v30.s[0]_v30.s[3]| x17
// x14 |v3.b[0]..v3.b[3]| |v19.s[0]_v19.s[3] v23.s[0]_v23.s[3]  v27.s[0]_v27.s[3]   v31.s[0]_v31.s[3]| x6
//     ------------------ ----------------------------------------------------------------------------

ProcessNextColumnLoop:
        ldr     q16,[x8],#16            // Init accumulators with column sums
        ldr     q20,[x8],#16
        ldr     q24,[x8],#16
        ldr     q28,[x8],#16
        mov     x0,x9                   // reload A0
        cmp     x4,#2                   // M < 2 ?
        add     x12,x9,x7               // x12 -> A1
        add     x13,x0,x7,lsl #1        // x13 -> A2
        ldr     q4,[x1],#16             // Load B
        csel    x12,x0,x12,lo           // if M < 2  A1 -> A0
        csel    x13,x12,x13,ls          // if M <= 2  A2 -> A1
        cmp     x4,4                    // M < 4 ?
        add     x14,x12,x7,lsl #1       // x14 -> A3
        ldr     q5,[x1],#16
        csel    x14,x13,x14,lo          // if M < 4  A3 -> A2
        ldr     d0,[x0],#8              // Load A0  1st/2nd block of 4
        mov     v17.16b,v16.16b
        mov     v18.16b,v16.16b
        ldr     d1,[x12],#8             // Load A1
        mov     v19.16b,v16.16b
        mov     v21.16b,v20.16b
        ldr     d2,[x13],#8             // Load A2
        mov     v22.16b,v20.16b
        mov     v23.16b,v20.16b
        ldr     d3,[x14],#8             // Load A3
        mov     v25.16b,v24.16b
        mov     v26.16b,v24.16b
        ldr     q6,[x1],#16
        mov     v27.16b,v24.16b
        mov     v29.16b,v28.16b
        ldr     q7,[x1],#16
        subs    x3,x10,#2               // one loop iteration and epilogue consume k = 32
        mov     v30.16b,v28.16b
        mov     v31.16b,v28.16b
        b.lo    BlockLoopEpilogue       // Need 32 k for main loop

BlockLoop:
        SdotByElement    16, 4, 0,0
        SdotByElement    17, 4, 1,0
        ldr     d8,[x0],#8              // Load A0  3rd/4th block of 4
        SdotByElement    18, 4, 2,0
        SdotByElement    19, 4, 3,0
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 0,0
        SdotByElement    21, 5, 1,0
        ldr     d9,[x12],#8
        SdotByElement    22, 5, 2,0
        SdotByElement    23, 5, 3,0
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 0,0
        SdotByElement    25, 6, 1,0
        ldr     d10,[x13],#8
        SdotByElement    26, 6, 2,0
        SdotByElement    27, 6, 3,0
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 0,0
        SdotByElement    29, 7, 1,0
        ldr     d11,[x14],#8
        SdotByElement    30, 7, 2,0
        SdotByElement    31, 7, 3,0
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 0,1
        SdotByElement    17, 4, 1,1
        SdotByElement    18, 4, 2,1
        SdotByElement    19, 4, 3,1
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 0,1
        SdotByElement    21, 5, 1,1
        SdotByElement    22, 5, 2,1
        SdotByElement    23, 5, 3,1
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 0,1
        SdotByElement    25, 6, 1,1
        SdotByElement    26, 6, 2,1
        SdotByElement    27, 6, 3,1
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 0,1
        SdotByElement    29, 7, 1,1
        SdotByElement    30, 7, 2,1
        SdotByElement    31, 7, 3,1
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 8,0
        SdotByElement    17, 4, 9,0
        ldr     d0,[x0],#8
        SdotByElement    18, 4,10,0
        SdotByElement    19, 4,11,0
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 8,0
        SdotByElement    21, 5, 9,0
        ldr     d1,[x12],#8
        SdotByElement    22, 5,10,0
        SdotByElement    23, 5,11,0
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 8,0
        SdotByElement    25, 6, 9,0
        ldr     d2,[x13],#8
        SdotByElement    26, 6,10,0
        SdotByElement    27, 6,11,0
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 8,0
        SdotByElement    29, 7, 9,0
        ldr     d3,[x14],#8
        SdotByElement    30, 7,10,0
        SdotByElement    31, 7,11,0
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 8,1
        SdotByElement    17, 4, 9,1
        SdotByElement    18, 4,10,1
        SdotByElement    19, 4,11,1
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 8,1
        SdotByElement    21, 5, 9,1
        SdotByElement    22, 5,10,1
        SdotByElement    23, 5,11,1
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 8,1
        SdotByElement    25, 6, 9,1
        SdotByElement    26, 6,10,1
        SdotByElement    27, 6,11,1
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 8,1
        SdotByElement    29, 7, 9,1
        subs    x3,x3,#1                // k -= 16
        SdotByElement    30, 7,10,1
        SdotByElement    31, 7,11,1
        ldr     q7,[x1],#16
        b.hs    BlockLoop

BlockLoopEpilogue:
        SdotByElement    16, 4, 0,0
        SdotByElement    17, 4, 1,0
        ldr     d8,[x0],#8
        SdotByElement    18, 4, 2,0
        SdotByElement    19, 4, 3,0
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 0,0
        SdotByElement    21, 5, 1,0
        ldr     d9,[x12],#8
        SdotByElement    22, 5, 2,0
        SdotByElement    23, 5, 3,0
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 0,0
        SdotByElement    25, 6, 1,0
        ldr     d10,[x13],#8
        SdotByElement    26, 6, 2,0
        SdotByElement    27, 6, 3,0
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 0,0
        SdotByElement    29, 7, 1,0
        ldr     d11,[x14],#8
        SdotByElement    30, 7, 2,0
        SdotByElement    31, 7, 3,0
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 0,1
        SdotByElement    17, 4, 1,1
        SdotByElement    18, 4, 2,1
        SdotByElement    19, 4, 3,1
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 0,1
        SdotByElement    21, 5, 1,1
        SdotByElement    22, 5, 2,1
        SdotByElement    23, 5, 3,1
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 0,1
        SdotByElement    25, 6, 1,1
        SdotByElement    26, 6, 2,1
        SdotByElement    27, 6, 3,1
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 0,1
        SdotByElement    29, 7, 1,1
        SdotByElement    30, 7, 2,1
        SdotByElement    31, 7, 3,1
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 8,0
        SdotByElement    17, 4, 9,0
        SdotByElement    18, 4,10,0
        SdotByElement    19, 4,11,0
        ldr     q4,[x1],#16
        SdotByElement    20, 5, 8,0
        SdotByElement    21, 5, 9,0
        SdotByElement    22, 5,10,0
        SdotByElement    23, 5,11,0
        ldr     q5,[x1],#16
        SdotByElement    24, 6, 8,0
        SdotByElement    25, 6, 9,0
        SdotByElement    26, 6,10,0
        SdotByElement    27, 6,11,0
        ldr     q6,[x1],#16
        SdotByElement    28, 7, 8,0
        SdotByElement    29, 7, 9,0
        SdotByElement    30, 7,10,0
        SdotByElement    31, 7,11,0
        ldr     q7,[x1],#16
        SdotByElement    16, 4, 8,1
        SdotByElement    17, 4, 9,1
        SdotByElement    18, 4,10,1
        SdotByElement    19, 4,11,1
        SdotByElement    20, 5, 8,1
        SdotByElement    21, 5, 9,1
        SdotByElement    22, 5,10,1
        SdotByElement    23, 5,11,1
        SdotByElement    24, 6, 8,1
        SdotByElement    25, 6, 9,1
        SdotByElement    26, 6,10,1
        SdotByElement    27, 6,11,1
        SdotByElement    28, 7, 8,1
        SdotByElement    29, 7, 9,1
        subs    x5,x5,#16               // adjust CountN remaining
        SdotByElement    30, 7,10,1
        SdotByElement    31, 7,11,1
        blo     StoreOutputPartial
        stp     q16,q20,[x2],#32
        stp     q24,q28,[x2],#32
        stp     q17,q21,[x16],#32
        stp     q25,q29,[x16],#32
        stp     q18,q22,[x17],#32
        stp     q26,q30,[x17],#32
        stp     q19,q23,[x6],#32
        stp     q27,q31,[x6],#32
        cbnz    x5,ProcessNextColumnLoop

ExitKernel:
        mov     x0,x4                   // return number of rows handled
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#.LGemmS8S8KernelFrame_SavedRegisters
        ret

//
// Store the partial 1 to 15 columns either overwriting the output matrix or
// accumulating into the existing contents of the output matrix.
//

StoreOutputPartial:
        tbz     x5,#3,StoreOutputPartial4
        stp     q16,q20,[x2],#32
        mov     v16.16b,v24.16b             // shift remaining elements down
        mov     v20.16b,v28.16b
        stp     q17,q21,[x16],#32
        mov     v17.16b,v25.16b
        mov     v21.16b,v29.16b
        stp     q18,q22,[x17],#32
        mov     v18.16b,v26.16b
        mov     v22.16b,v30.16b
        stp     q19,q23,[x6],#32
        mov     v19.16b,v27.16b
        mov     v23.16b,v31.16b

StoreOutputPartial4:
        tbz     x5,#2,StoreOutputPartial2
        st1     {v16.4s},[x2],#16
        mov     v16.16b,v20.16b             // shift remaining elements down
        st1     {v17.4s},[x16],#16
        mov     v17.16b,v21.16b
        st1     {v18.4s},[x17],#16
        mov     v18.16b,v22.16b
        st1     {v19.4s},[x6],#16
        mov     v19.16b,v23.16b

StoreOutputPartial2:
        tbz     x5,#1,StoreOutputPartial1
        st1     {v16.2s},[x2],#8
        dup     v16.4s,v16.s[2]             // shift remaining elements down
        st1     {v17.2s},[x16],#8
        dup     v17.4s,v17.s[2]
        st1     {v18.2s},[x17],#8
        dup     v18.4s,v18.s[2]
        st1     {v19.2s},[x6],#8
        dup     v19.4s,v19.s[2]

StoreOutputPartial1:
        tbz     x5,#0,ExitKernel
        st1     {v16.s}[0],[x2]
        st1     {v17.s}[0],[x16]
        st1     {v18.s}[0],[x17]
        st1     {v19.s}[0],[x6]
        b       ExitKernel

        .end
