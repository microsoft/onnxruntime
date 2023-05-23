/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    HalfGemmKernelNeon.asm

Abstract:

    This module implements the kernels for the half precision matrix/matrix
    multiply operation (HALF GEMM).

--*/

#include "kxarm64.h"

//
// Stack frame layout for the half gemm kernel.
// Callee save registers: d8-d15, x19-x30. x18 is reserved by the OS.
//

#define HGemmKernelFrame_SavedRegs  (2 * 8)
#define HGemmKernelFrame_B          0  + HGemmKernelFrame_SavedRegs
#define HGemmKernelFrame_ldb        8  + HGemmKernelFrame_SavedRegs
#define HGemmKernelFrame_ZeroMode   16 + HGemmKernelFrame_SavedRegs


/*++

Routine Description:

    This routine is an inner kernel to compute 6 rows of GEMM

Arguments:

    CountM - (x0) the number of rows for matrix A and matrix C.
            only process 6 rows

    CountN - (x1) the number of columns from matrix B and matrix C

    CountK - (x2/x0) the number of columns from matrix A and the
        number of rows from matrix B.

    C      - (x3) the address of matrix C.

    ldc    - (x4) - the first dimension of matrix C.

    Bias   - (x5) - the address of the Bias vector (optional)

    A      - (x6) - the address of matrix A

    lda    - (x7) - the first dimension of matrix A

    B      - the address of matrix B

    ldb    - the first dimension of matrix B

    ZeroMode - true if the output matrix must be zero initialized, else
        if the output matrix is accumulated into

--*/

        LEAF_ENTRY MlasHalfGemmKernelNeon

        PROLOG_SAVE_REG     x19,#-HGemmKernelFrame_SavedRegs!
        ldr     x9,[sp,#HGemmKernelFrame_ldb]
        lsl     x2,x2,#1                // k *= sizeof(fp16)
        cmp     x0,2
        add     x14,x6,x7,lsl #1        // a1 = a0 + lda
        add     x10,x3,x4,lsl #1        // c1 = c0 + ldc
        ldr     x8,[sp,#HGemmKernelFrame_B]
        csel    x14,x6,x14,LO           // M < 2 ? a1 = a0
        csel    x10,x3,x10,LO           //         c1 = c0
        add     x15,x14,x7,lsl #1       // a2 = a1 + lda
        add     x11,x10,x4,lsl #1       // c2 = c1 + ldc
        csel    x15,x14,x15,LS          // M <= 2 ? a2 = a1
        csel    x11,x10,x11,LS          //          c2 = c1
        cmp     x0,4
        add     x16,x15,x7,lsl #1       // a3 = a2 + lda
        add     x12,x11,x4,lsl #1       // c3 = c2 + ldc
        csel    x16,x15,x16,LO          // M < 4 ? a3 = a2
        csel    x12,x11,x12,LO          //         c3 = c2
        add     x17,x16,x7,lsl #1       // a4 = a3 + lda
        add     x13,x12,x4,lsl #1       // c4 = c3 + ldc
        csel    x17,x16,x17,LS          // M <= 4 ? a4 = a3
        csel    x13,x12,x13,LS          //          c4 = c3
        cmp     x0,6
        add     x7,x17,x7,lsl #1        // a5 = a4 + lda
        add     x4,x13,x4,lsl #1        // c5 = c4 + ldc
        csel    x7,x17,x7,LO            // M < 6 ? a5 = a4
        csel    x4,x13,x4,LO            //         c5 = c4
        lsl     x9,x9,#1                // ldb *= sizeof(fp16)
        ldrb    w19,[sp,#HGemmKernelFrame_ZeroMode]
        sub     x9,x9,16                // ldb -= 16

/****
Main loop processes 6x16 tile, depth 4.
                                       B 4x16
                        ---------------------------------------
                        |v16.h[0]..v16.h[7] v17.h[0]..v17.h[7]|  x8
                        |v18.h[0]..v18.h[7] v19.h[0]..v19.h[7]|  x8
                        |v16.h[0]..v16.h[7] v17.h[0]..v17.h[7]|  x8
                        |v18.h[0]..v18.h[7] v19.h[0]..v19.h[7]|  x8
          A 6x4         ---------------------------------------
    ------------------  ---------------------------------------
x6  |v0.h[0]..v0.h[3]|  |v20.h[0]..v20.h[7] v21.h[0]..v21.h[7]|  x3
x14 |v1.h[0]..v1.h[3]|  |v22.h[0]..v22.h[7] v23.h[0]..v23.h[7]|  x10
x15 |v2.h[0]..v2.h[3]|  |v24.h[0]..v24.h[7] v25.h[0]..v25.h[7]|  x11
x16 |v3.h[0]..v3.h[3]|  |v26.h[0]..v26.h[7] v27.h[0]..v27.h[7]|  x12
x17 |v4.h[0]..v4.h[3]|  |v28.h[0]..v28.h[7] v29.h[0]..v29.h[7]|  x13
x7  |v5.h[0]..v5.h[3]|  |v30.h[0]..v30.h[7] v31.h[0]..v31.h[7]|  x4
    ------------------  ---------------------------------------
****/

M6N16OutterLoopN
        cbz     x5,M6N16SkipBias
        ldp     q20,q21,[x5],32         // Load 16 Bias values
        b       M6N16PopulateAccumulators

M6N16SkipBias
        eor     q20.16b,q20.16b,q20.16b // No bias, reset regs
        eor     q21.16b,q21.16b,q21.16b

M6N16PopulateAccumulators
        mov     v22.16b,v20.16b
        mov     v23.16b,v21.16b
        mov     v24.16b,v20.16b
        mov     v25.16b,v21.16b
        mov     v26.16b,v20.16b
        mov     v27.16b,v21.16b
        mov     v28.16b,v20.16b
        subs    x0,x2,8                 // k -= 4 (8 bytes)
        mov     v29.16b,v21.16b
        mov     v30.16b,v20.16b
        mov     v31.16b,v21.16b
        b.LO    M6N16RemainderK123      // remaining k 1~3

        ldr     d0,[x6],8               // A0
        ldr     q16,[x8],16             // B0.l
        ld1     {v17.16b},[x8],x9       // B0.high  x8 <- next row
        subs    x0,x0,8                 // over decement k -= 4 (8 bytes)
        ldr     d1,[x14],8              // A1
        ldr     d2,[x15],8              // A2
        ldr     d3,[x16],8              // A3
        b.LO    M6N16LoopK_Epilogue     // need k>=8 for main loop

M6N16InnerLoopK
        fmla    v20.8h,v16.8h,v0.h[0]
        fmla    v21.8h,v17.8h,v0.h[0]
        ldr     d4,[x17],8              // A4
        fmla    v22.8h,v16.8h,v1.h[0]
        fmla    v23.8h,v17.8h,v1.h[0]
        ldr     d5,[x7],8               // A5
        fmla    v24.8h,v16.8h,v2.h[0]
        fmla    v25.8h,v17.8h,v2.h[0]
        ldr     q18,[x8],16             // B1.low
        fmla    v26.8h,v16.8h,v3.h[0]
        fmla    v27.8h,v17.8h,v3.h[0]
        ld1     {v19.16b},[x8],x9       // B1.high  x8 <- next row
        fmla    v28.8h,v16.8h,v4.h[0]
        fmla    v29.8h,v17.8h,v4.h[0]
        fmla    v30.8h,v16.8h,v5.h[0]
        fmla    v31.8h,v17.8h,v5.h[0]
        subs    x0,x0,8                 // k -= 4

        fmla    v20.8h,v18.8h,v0.h[1]
        fmla    v21.8h,v19.8h,v0.h[1]
        ldr     q16,[x8],16             // B2.low
        fmla    v22.8h,v18.8h,v1.h[1]
        fmla    v23.8h,v19.8h,v1.h[1]
        ld1     {v17.16b},[x8],x9       // B2.high  x8 <- next row
        fmla    v24.8h,v18.8h,v2.h[1]
        fmla    v25.8h,v19.8h,v2.h[1]
        fmla    v26.8h,v18.8h,v3.h[1]
        fmla    v27.8h,v19.8h,v3.h[1]
        fmla    v28.8h,v18.8h,v4.h[1]
        fmla    v29.8h,v19.8h,v4.h[1]
        fmla    v30.8h,v18.8h,v5.h[1]
        fmla    v31.8h,v19.8h,v5.h[1]

        fmla    v20.8h,v16.8h,v0.h[2]
        fmla    v21.8h,v17.8h,v0.h[2]
        ldr     q18,[x8],16             // B3.low
        fmla    v22.8h,v16.8h,v1.h[2]
        fmla    v23.8h,v17.8h,v1.h[2]
        ld1     {v19.16b},[x8],x9       // B3.high  x8 <- next row
        fmla    v24.8h,v16.8h,v2.h[2]
        fmla    v25.8h,v17.8h,v2.h[2]
        fmla    v26.8h,v16.8h,v3.h[2]
        fmla    v27.8h,v17.8h,v3.h[2]
        fmla    v28.8h,v16.8h,v4.h[2]
        fmla    v29.8h,v17.8h,v4.h[2]
        fmla    v30.8h,v16.8h,v5.h[2]
        fmla    v31.8h,v17.8h,v5.h[2]

        ldr     q16,[x8],16             // Load B0.low for next iter
        fmla    v20.8h,v18.8h,v0.h[3]
        fmla    v21.8h,v19.8h,v0.h[3]
        ld1     {v17.16b},[x8],x9       // Load B0.high for next iter
        fmla    v22.8h,v18.8h,v1.h[3]
        fmla    v23.8h,v19.8h,v1.h[3]
        ldr     d0,[x6],8               // Load A0 for next iter
        fmla    v24.8h,v18.8h,v2.h[3]
        fmla    v25.8h,v19.8h,v2.h[3]
        ldr     d1,[x14],8              // Load A1 for next iter
        fmla    v26.8h,v18.8h,v3.h[3]
        fmla    v27.8h,v19.8h,v3.h[3]
        ldr     d2,[x15],8              // Load A2 for next iter
        fmla    v28.8h,v18.8h,v4.h[3]
        fmla    v29.8h,v19.8h,v4.h[3]
        ldr     d3,[x16],8              // Load A3 for next iter
        fmla    v30.8h,v18.8h,v5.h[3]
        fmla    v31.8h,v19.8h,v5.h[3]
        b.hs    M6N16InnerLoopK         // k >= 8 for main loop

M6N16LoopK_Epilogue
        // last block of k >= 4, no pre-load for next iter
        fmla    v20.8h,v16.8h,v0.h[0]
        fmla    v21.8h,v17.8h,v0.h[0]
        ldr     d4,[x17],8              // A4
        fmla    v22.8h,v16.8h,v1.h[0]
        fmla    v23.8h,v17.8h,v1.h[0]
        ldr     d5,[x7],8               // A5
        fmla    v24.8h,v16.8h,v2.h[0]
        fmla    v25.8h,v17.8h,v2.h[0]
        ldr     q18,[x8],16             // B1.low
        fmla    v26.8h,v16.8h,v3.h[0]
        fmla    v27.8h,v17.8h,v3.h[0]
        ld1     {v19.16b},[x8],x9       // B1.high  x8 <- next row
        fmla    v28.8h,v16.8h,v4.h[0]
        fmla    v29.8h,v17.8h,v4.h[0]
        fmla    v30.8h,v16.8h,v5.h[0]
        fmla    v31.8h,v17.8h,v5.h[0]
        adds    x0,x0,8                 // revert k over-decrement

        fmla    v20.8h,v18.8h,v0.h[1]
        fmla    v21.8h,v19.8h,v0.h[1]
        ldr     q16,[x8],16             // B2.low
        fmla    v22.8h,v18.8h,v1.h[1]
        fmla    v23.8h,v19.8h,v1.h[1]
        ld1     {v17.16b},[x8],x9       // B2.high  x8 <- next row
        fmla    v24.8h,v18.8h,v2.h[1]
        fmla    v25.8h,v19.8h,v2.h[1]
        fmla    v26.8h,v18.8h,v3.h[1]
        fmla    v27.8h,v19.8h,v3.h[1]
        fmla    v28.8h,v18.8h,v4.h[1]
        fmla    v29.8h,v19.8h,v4.h[1]
        fmla    v30.8h,v18.8h,v5.h[1]
        fmla    v31.8h,v19.8h,v5.h[1]

        fmla    v20.8h,v16.8h,v0.h[2]
        fmla    v21.8h,v17.8h,v0.h[2]
        ldr     q18,[x8],16             // B3.low
        fmla    v22.8h,v16.8h,v1.h[2]
        fmla    v23.8h,v17.8h,v1.h[2]
        ld1     {v19.16b},[x8],x9       // B3.high  x8 <- next row
        fmla    v24.8h,v16.8h,v2.h[2]
        fmla    v25.8h,v17.8h,v2.h[2]
        fmla    v26.8h,v16.8h,v3.h[2]
        fmla    v27.8h,v17.8h,v3.h[2]
        fmla    v28.8h,v16.8h,v4.h[2]
        fmla    v29.8h,v17.8h,v4.h[2]
        fmla    v30.8h,v16.8h,v5.h[2]
        fmla    v31.8h,v17.8h,v5.h[2]

        fmla    v20.8h,v18.8h,v0.h[3]
        fmla    v21.8h,v19.8h,v0.h[3]
        fmla    v22.8h,v18.8h,v1.h[3]
        fmla    v23.8h,v19.8h,v1.h[3]
        fmla    v24.8h,v18.8h,v2.h[3]
        fmla    v25.8h,v19.8h,v2.h[3]
        fmla    v26.8h,v18.8h,v3.h[3]
        fmla    v27.8h,v19.8h,v3.h[3]
        fmla    v28.8h,v18.8h,v4.h[3]
        fmla    v29.8h,v19.8h,v4.h[3]
        fmla    v30.8h,v18.8h,v5.h[3]
        fmla    v31.8h,v19.8h,v5.h[3]
        b.NE    M6N16RemainderK123      // remaining k 1~3

M6N16OutterLoopNTail
        subs    x1,x1,16                // N -= 16
        ldr     x8,[sp,#HGemmKernelFrame_B]
        b.LO    M6StoreRemainderN       // remaining N < 16

        cbnz    x19,M6N16SkipAccumulateOutput
        ldp     q0,q1,[x3]
        ldp     q2,q3,[x10]
        ldp     q4,q5,[x11]
        ldp     q6,q7,[x12]
        ldp     q16,q17,[x13]
        ldp     q18,q19,[x4]
        fadd    v20.8h,v20.8h,v0.8h     // !ZeroMode
        fadd    v21.8h,v21.8h,v1.8h     // accumulate into C
        fadd    v22.8h,v22.8h,v2.8h
        fadd    v23.8h,v23.8h,v3.8h
        fadd    v24.8h,v24.8h,v4.8h
        fadd    v25.8h,v25.8h,v5.8h
        fadd    v26.8h,v26.8h,v6.8h
        fadd    v27.8h,v27.8h,v7.8h
        fadd    v28.8h,v28.8h,v16.8h
        fadd    v29.8h,v29.8h,v17.8h
        fadd    v30.8h,v30.8h,v18.8h
        fadd    v31.8h,v31.8h,v19.8h

M6N16SkipAccumulateOutput
        st1     {v20.16b,v21.16b},[x3],32
        sub     x6,x6,x2                // restore a0
        st1     {v22.16b,v23.16b},[x10],32
        sub     x14,x14,x2              // restore a1
        st1     {v24.16b,v25.16b},[x11],32
        sub     x15,x15,x2              // restore a2
        st1     {v26.16b,v27.16b},[x12],32
        sub     x16,x16,x2              // restore a3
        st1     {v28.16b,v29.16b},[x13],32
        sub     x17,x17,x2              // restore a4
        add     x8,x8,32                // B <- next 16 columns
        st1     {v30.16b,v31.16b},[x4],32
        sub     x7,x7,x2                // restore a5
        str     x8,[sp,#HGemmKernelFrame_B]
        b.HI    M6N16OutterLoopN

ExitKernel
        EPILOG_RESTORE_REG       x19,#HGemmKernelFrame_SavedRegs!
        EPILOG_RETURN

M6N16RemainderK123
        tbz     x0,2,M6N16RemainderK1
        ldr     s0,[x6],4               // A0
        ldr     q16,[x8],16             // B0.low
        ld1     {v17.16b},[x8],x9       // B0.high
        ldr     s1,[x14],4              // A1
        ldr     s2,[x15],4              // A2
        ldr     s3,[x16],4              // A3
        ldr     s4,[x17],4              // A4
        ldr     s5,[x7],4               // A5
        ldr     q18,[x8],16             // B1.low
        ld1     {v19.16b},[x8],x9       // B2.high
        fmla    v20.8h,v16.8h,v0.h[0]
        fmla    v22.8h,v16.8h,v1.h[0]
        fmla    v24.8h,v16.8h,v2.h[0]
        fmla    v26.8h,v16.8h,v3.h[0]
        fmla    v28.8h,v16.8h,v4.h[0]
        fmla    v30.8h,v16.8h,v5.h[0]
        fmla    v21.8h,v17.8h,v0.h[0]
        fmla    v23.8h,v17.8h,v1.h[0]
        fmla    v25.8h,v17.8h,v2.h[0]
        fmla    v27.8h,v17.8h,v3.h[0]
        fmla    v29.8h,v17.8h,v4.h[0]
        fmla    v31.8h,v17.8h,v5.h[0]

        fmla    v20.8h,v18.8h,v0.h[1]
        fmla    v22.8h,v18.8h,v1.h[1]
        fmla    v24.8h,v18.8h,v2.h[1]
        fmla    v26.8h,v18.8h,v3.h[1]
        fmla    v28.8h,v18.8h,v4.h[1]
        fmla    v30.8h,v18.8h,v5.h[1]
        fmla    v21.8h,v19.8h,v0.h[1]
        fmla    v23.8h,v19.8h,v1.h[1]
        fmla    v25.8h,v19.8h,v2.h[1]
        fmla    v27.8h,v19.8h,v3.h[1]
        fmla    v29.8h,v19.8h,v4.h[1]
        fmla    v31.8h,v19.8h,v5.h[1]
        tbz     x0,1,M6N16OutterLoopNTail

M6N16RemainderK1
        ldr     h0,[x6],2               // A0
        ldr     q16,[x8],16             // B0.low
        ld1     {v17.16b},[x8],x9       // B0.high
        ldr     h1,[x14],2              // A1
        ldr     h2,[x15],2              // A2
        ldr     h3,[x16],2              // A3
        ldr     h4,[x17],2              // A4
        ldr     h5,[x7],2               // A5
        fmla    v20.8h,v16.8h,v0.h[0]
        fmla    v22.8h,v16.8h,v1.h[0]
        fmla    v24.8h,v16.8h,v2.h[0]
        fmla    v26.8h,v16.8h,v3.h[0]
        fmla    v28.8h,v16.8h,v4.h[0]
        fmla    v30.8h,v16.8h,v5.h[0]
        fmla    v21.8h,v17.8h,v0.h[0]
        fmla    v23.8h,v17.8h,v1.h[0]
        fmla    v25.8h,v17.8h,v2.h[0]
        fmla    v27.8h,v17.8h,v3.h[0]
        fmla    v29.8h,v17.8h,v4.h[0]
        fmla    v31.8h,v17.8h,v5.h[0]
        b       M6N16OutterLoopNTail

M6StoreRemainderN
        cbnz    x19,M6StoreRemainderNZeroMode
        tbz     x1,3,M6StoreRemainderN4
        ldr     q0,[x3]
        ldr     q1,[x10]
        ldr     q2,[x11]
        ldr     q3,[x12]
        ldr     q4,[x13]
        ldr     q5,[x4]
        fadd    v20.8h,v20.8h,v0.8h
        fadd    v22.8h,v22.8h,v1.8h
        fadd    v24.8h,v24.8h,v2.8h
        str     q20,[x3],16
        mov     v20.16b,v21.16b
        str     q22,[x10],16
        mov     v22.16b,v23.16b
        str     q24,[x11],16
        mov     v24.16b,v25.16b
        fadd    v26.8h,v26.8h,v3.8h
        fadd    v28.8h,v28.8h,v4.8h
        fadd    v30.8h,v30.8h,v5.8h
        str     q26,[x12],16
        mov     v26.16b,v27.16b
        str     q28,[x13],16
        mov     v28.16b,v29.16b
        str     q30,[x4],16
        mov     v30.16b,v31.16b

M6StoreRemainderN4
        tbz     x1,2,M6StoreRemainderN2
        ldr     d0,[x3]
        ldr     d1,[x10]
        ldr     d2,[x11]
        ldr     d3,[x12]
        ldr     d4,[x13]
        ldr     d5,[x4]
        fadd    v21.4h,v20.4h,v0.4h
        dup     d20,v20.d[1]
        fadd    v23.4h,v22.4h,v1.4h
        dup     d22,v22.d[1]
        fadd    v25.4h,v24.4h,v2.4h
        dup     d24,v24.d[1]
        fadd    v27.4h,v26.4h,v3.4h
        dup     d26,v26.d[1]
        fadd    v29.4h,v28.4h,v4.4h
        dup     d28,v28.d[1]
        fadd    v31.4h,v30.4h,v5.4h
        dup     d30,v30.d[1]
        str     d21,[x3],8
        str     d23,[x10],8
        str     d25,[x11],8
        str     d27,[x12],8
        str     d29,[x13],8
        str     d31,[x4],8

M6StoreRemainderN2
        tbz     x1,1,M6StoreRemainderN1
        ldr     s0,[x3]
        ldr     s1,[x10]
        ldr     s2,[x11]
        ldr     s3,[x12]
        ldr     s4,[x13]
        ldr     s5,[x4]
        fadd    v21.4h,v20.4h,v0.4h
        fadd    v23.4h,v22.4h,v1.4h
        fadd    v25.4h,v24.4h,v2.4h
        fadd    v27.4h,v26.4h,v3.4h
        fadd    v29.4h,v28.4h,v4.4h
        fadd    v31.4h,v30.4h,v5.4h
        str     s21,[x3],4
        str     s23,[x10],4
        dup     s20,v20.s[1]
        dup     s22,v22.s[1]
        str     s25,[x11],4
        str     s27,[x12],4
        dup     s24,v24.s[1]
        dup     s26,v26.s[1]
        str     s29,[x13],4
        str     s31,[x4],4
        dup     s28,v28.s[1]
        dup     s30,v30.s[1]

M6StoreRemainderN1
        tbz     x1,0,ExitKernel
        ldr     h0,[x3]
        ldr     h1,[x10]
        ldr     h2,[x11]
        ldr     h3,[x12]
        ldr     h4,[x13]
        ldr     h5,[x4]
        fadd    v20.4h,v20.4h,v0.4h
        fadd    v22.4h,v22.4h,v1.4h
        fadd    v24.4h,v24.4h,v2.4h
        fadd    v26.4h,v26.4h,v3.4h
        fadd    v28.4h,v28.4h,v4.4h
        fadd    v30.4h,v30.4h,v5.4h
        str     h20,[x3]
        str     h22,[x10]
        str     h24,[x11]
        str     h26,[x12]
        str     h28,[x13]
        str     h30,[x4]
        b       ExitKernel

M6StoreRemainderNZeroMode
        tbz     x1,3,M6StoreRemainderN4ZeroMode
        str     q20,[x3],16
        mov     v20.16b,v21.16b
        str     q22,[x10],16
        mov     v22.16b,v23.16b
        str     q24,[x11],16
        mov     v24.16b,v25.16b
        str     q26,[x12],16
        mov     v26.16b,v27.16b
        str     q28,[x13],16
        mov     v28.16b,v29.16b
        str     q30,[x4],16
        mov     v30.16b,v31.16b

M6StoreRemainderN4ZeroMode
        tbz     x1,2,M6StoreRemainderN2ZeroMode
        str     d20,[x3],8
        str     d22,[x10],8
        dup     d20,v20.d[1]
        dup     d22,v22.d[1]
        str     d24,[x11],8
        str     d26,[x12],8
        dup     d24,v24.d[1]
        dup     d26,v26.d[1]
        str     d28,[x13],8
        str     d30,[x4],8
        dup     d28,v28.d[1]
        dup     d30,v30.d[1]

M6StoreRemainderN2ZeroMode
        tbz     x1,1,M6StoreRemainderN1ZeroMode
        str     s20,[x3],4
        str     s22,[x10],4
        dup     s20,v20.s[1]
        dup     s22,v22.s[1]
        str     s24,[x11],4
        str     s26,[x12],4
        dup     s24,v24.s[1]
        dup     s26,v26.s[1]
        str     s28,[x13],4
        str     s30,[x4],4
        dup     s28,v28.s[1]
        dup     s30,v30.s[1]

M6StoreRemainderN1ZeroMode
        tbz     x1,0,ExitKernel
        str     h20,[x3]
        str     h22,[x10]
        str     h24,[x11]
        str     h26,[x12]
        str     h28,[x13]
        str     h30,[x4]
        b       ExitKernel

        LEAF_END MlasHalfGemmKernelNeon

        END
