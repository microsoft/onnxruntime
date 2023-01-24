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
        lsl     x2,x2,#1                // k *= sizeof(fp16)
        ldr     x8,[sp,#HGemmKernelFrame_B]
        ldr     x15,[sp,#HGemmKernelFrame_ldb]
        CMP     x0, 2                   // if M < 2
        add     x9,x6,x7,lsl #1         // a1 = a0 + lda
        add     x16,x3,x4,lsl #1        // c1 = c0 + ldc
        CSEL    x9, x6, x9, LO          //     a1 = a0
        CSEL    x16, x3, x16, LO        //     c1 = c0
        add     x10,x9,x7,lsl #1        // a2 = a1 + lda
        add     x17,x16,x4,lsl #1       // c2 = c1 + ldc
        CSEL    x10, x9, x10, LS        // if M <= 2  a2 = a1
        CSEL    x17, x16, x17, LS       //            c2 = c1
        CMP     x0, 4                   // if M < 4
        add     x11,x10,x7,lsl #1       // a3 = a2 + lda
        add     x14,x17,x4,lsl #1       // c3 = c2 + ldc
        CSEL    x11, x10, x11, LO       //     a3 = a2
        CSEL    x14, x17, x14, LO       //     c3 = c2
        add     x12,x11,x7,lsl #1       // a4 = a3 + lda
        add     x13,x14,x4,lsl #1       // c4 = c3 + ldc
        CSEL    x12, x11, x12, LS       // if M <= 4  a4 = a3
        CSEL    x13, x14, x13, LS       //            c4 = c3
        CMP     x0, 6                   // if M < 6
        add     x7,x12,x7,lsl #1        // a5 = a4 + lda
        add     x4,x13,x4,lsl #1        // c5 = c4 + ldc
        CSEL    x7, x12, x7, LO         //     a5 = a4
        CSEL    x4, x13, x4, LO         //     c5 = c4
        lsl     x15,x15,#1              // ldb *= sizeof(fp16)
        sub     x15,x15,16              // ldb -= 16
        ldrb    w19,[sp,#HGemmKernelFrame_ZeroMode]

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
x9  |v1.h[0]..v1.h[3]|  |v22.h[0]..v22.h[7] v23.h[0]..v23.h[7]|  x16
x10 |v2.h[0]..v2.h[3]|  |v24.h[0]..v24.h[7] v25.h[0]..v25.h[7]|  x17
x11 |v3.h[0]..v3.h[3]|  |v26.h[0]..v26.h[7] v27.h[0]..v27.h[7]|  x14
x12 |v4.h[0]..v4.h[3]|  |v28.h[0]..v28.h[7] v29.h[0]..v29.h[7]|  x13
x7  |v5.h[0]..v5.h[3]|  |v30.h[0]..v30.h[7] v31.h[0]..v31.h[7]|  x4
    ------------------  ---------------------------------------
****/

M6N16OutterLoopN
        cbz     x5, M6N16SkipBias
        ldp     q20,q21,[x5],32         // Load 16 Bias values
        b       M6N16PopulateAccumulators

M6N16SkipBias
        eor     q20.16b,q20.16b,q20.16b // No bias, reset regs
        eor     q21.16b,q21.16b,q21.16b

M6N16PopulateAccumulators
        MOV     v22.16b, v20.16b
        MOV     v23.16b, v21.16b
        MOV     v24.16b, v20.16b
        MOV     v25.16b, v21.16b
        MOV     v26.16b, v20.16b
        MOV     v27.16b, v21.16b
        MOV     v28.16b, v20.16b
        subs    x0,x2,8                 // k -= 4 (8 bytes)
        MOV     v29.16b, v21.16b
        MOV     v30.16b, v20.16b
        MOV     v31.16b, v21.16b
        b.lo    M6N16RemainderK123      // remaining k 1~3

        ldr     d0,[x6],8               // A0
        ldr     q16,[x8],16             // B0.l
        ld1     {v17.16b},[x8],x15      // B0.high  x8 <- next row
        subs    x0,x0,8                 // over decement k -= 4 (8 bytes)
        ldr     d1,[x9],8               // A1
        ldr     d2,[x10],8              // A2
        ldr     d3,[x11],8              // A3
        b.lo    M6N16LoopK_Epilogue     // need k>=8 for main loop

M6N16InnerLoopK
        FMLA    v20.8h, v16.8h,  v0.h[0]
        FMLA    v21.8h, v17.8h,  v0.h[0]
        LDR     d4, [x12], 8              // A4
        FMLA    v22.8h, v16.8h,  v1.h[0]
        FMLA    v23.8h, v17.8h,  v1.h[0]
        LDR     d5,  [x7], 8              // A5
        FMLA    v24.8h, v16.8h,  v2.h[0]
        FMLA    v25.8h, v17.8h,  v2.h[0]
        ldr     q18,[x8],16             // B1.low
        FMLA    v26.8h, v16.8h,  v3.h[0]
        FMLA    v27.8h, v17.8h,  v3.h[0]
        ld1     {v19.16b},[x8],x15      // B1.high  x8 <- next row
        FMLA    v28.8h, v16.8h,  v4.h[0]
        FMLA    v29.8h, v17.8h,  v4.h[0]
        FMLA    v30.8h, v16.8h,  v5.h[0]
        FMLA    v31.8h, v17.8h,  v5.h[0]
        subs    x0,x0,8                     // k -= 4

        FMLA    v20.8h, v18.8h,  v0.h[1]
        FMLA    v21.8h, v19.8h,  v0.h[1]
        ldr     q16,[x8],16             // B2.low
        FMLA    v22.8h, v18.8h,  v1.h[1]
        FMLA    v23.8h, v19.8h,  v1.h[1]
        ld1     {v17.16b},[x8],x15      // B2.high  x8 <- next row
        FMLA    v24.8h, v18.8h,  v2.h[1]
        FMLA    v25.8h, v19.8h,  v2.h[1]
        FMLA    v26.8h, v18.8h,  v3.h[1]
        FMLA    v27.8h, v19.8h,  v3.h[1]
        FMLA    v28.8h, v18.8h,  v4.h[1]
        FMLA    v29.8h, v19.8h,  v4.h[1]
        FMLA    v30.8h, v18.8h,  v5.h[1]
        FMLA    v31.8h, v19.8h,  v5.h[1]

        FMLA    v20.8h, v16.8h,  v0.h[2]
        FMLA    v21.8h, v17.8h,  v0.h[2]
        ldr     q18,[x8],16             // B3.low
        FMLA    v22.8h, v16.8h,  v1.h[2]
        FMLA    v23.8h, v17.8h,  v1.h[2]
        ld1     {v19.16b},[x8],x15      // B3.high  x8 <- next row
        FMLA    v24.8h, v16.8h,  v2.h[2]
        FMLA    v25.8h, v17.8h,  v2.h[2]
        FMLA    v26.8h, v16.8h,  v3.h[2]
        FMLA    v27.8h, v17.8h,  v3.h[2]
        FMLA    v28.8h, v16.8h,  v4.h[2]
        FMLA    v29.8h, v17.8h,  v4.h[2]
        FMLA    v30.8h, v16.8h,  v5.h[2]
        FMLA    v31.8h, v17.8h,  v5.h[2]

        ldr     q16,[x8],16             // B0.low  next iter
        FMLA    v20.8h, v18.8h,  v0.h[3]
        FMLA    v21.8h, v19.8h,  v0.h[3]
        ld1     {v17.16b},[x8],x15      // B0.high  x8 <- next row
        FMLA    v22.8h, v18.8h,  v1.h[3]
        FMLA    v23.8h, v19.8h,  v1.h[3]
        LDR     d0,  [x6], 8              // A0
        FMLA    v24.8h, v18.8h,  v2.h[3]
        FMLA    v25.8h, v19.8h,  v2.h[3]
        LDR     d1,  [x9], 8              // A1
        FMLA    v26.8h, v18.8h,  v3.h[3]
        FMLA    v27.8h, v19.8h,  v3.h[3]
        LDR     d2, [x10], 8              // A2
        FMLA    v28.8h, v18.8h,  v4.h[3]
        FMLA    v29.8h, v19.8h,  v4.h[3]
        LDR     d3, [x11], 8              // A3
        FMLA    v30.8h, v18.8h,  v5.h[3]
        FMLA    v31.8h, v19.8h,  v5.h[3]
        b.hs    M6N16InnerLoopK             // k >= 8 for main loop

M6N16LoopK_Epilogue
        // last block of k >= 4, no pre-load for next iter
        FMLA    v20.8h, v16.8h,  v0.h[0]
        FMLA    v21.8h, v17.8h,  v0.h[0]
        LDR     d4, [x12], 8              // A4
        FMLA    v22.8h, v16.8h,  v1.h[0]
        FMLA    v23.8h, v17.8h,  v1.h[0]
        LDR     d5,  [x7], 8              // A5
        FMLA    v24.8h, v16.8h,  v2.h[0]
        FMLA    v25.8h, v17.8h,  v2.h[0]
        ldr     q18,[x8],16             // B1.low
        FMLA    v26.8h, v16.8h,  v3.h[0]
        FMLA    v27.8h, v17.8h,  v3.h[0]
        ld1     {v19.16b},[x8],x15      // B1.high  x8 <- next row
        FMLA    v28.8h, v16.8h,  v4.h[0]
        FMLA    v29.8h, v17.8h,  v4.h[0]
        FMLA    v30.8h, v16.8h,  v5.h[0]
        FMLA    v31.8h, v17.8h,  v5.h[0]
        adds    x0,x0,8                     // revert k over-decrement

        FMLA    v20.8h, v18.8h,  v0.h[1]
        FMLA    v21.8h, v19.8h,  v0.h[1]
        ldr     q16,[x8],16             // B2.low
        FMLA    v22.8h, v18.8h,  v1.h[1]
        FMLA    v23.8h, v19.8h,  v1.h[1]
        ld1     {v17.16b},[x8],x15      // B2.high  x8 <- next row
        FMLA    v24.8h, v18.8h,  v2.h[1]
        FMLA    v25.8h, v19.8h,  v2.h[1]
        FMLA    v26.8h, v18.8h,  v3.h[1]
        FMLA    v27.8h, v19.8h,  v3.h[1]
        FMLA    v28.8h, v18.8h,  v4.h[1]
        FMLA    v29.8h, v19.8h,  v4.h[1]
        FMLA    v30.8h, v18.8h,  v5.h[1]
        FMLA    v31.8h, v19.8h,  v5.h[1]

        FMLA    v20.8h, v16.8h,  v0.h[2]
        FMLA    v21.8h, v17.8h,  v0.h[2]
        ldr     q18,[x8],16             // B3.low
        FMLA    v22.8h, v16.8h,  v1.h[2]
        FMLA    v23.8h, v17.8h,  v1.h[2]
        ld1     {v19.16b},[x8],x15      // B3.high  x8 <- next row
        FMLA    v24.8h, v16.8h,  v2.h[2]
        FMLA    v25.8h, v17.8h,  v2.h[2]
        FMLA    v26.8h, v16.8h,  v3.h[2]
        FMLA    v27.8h, v17.8h,  v3.h[2]
        FMLA    v28.8h, v16.8h,  v4.h[2]
        FMLA    v29.8h, v17.8h,  v4.h[2]
        FMLA    v30.8h, v16.8h,  v5.h[2]
        FMLA    v31.8h, v17.8h,  v5.h[2]

        FMLA    v20.8h, v18.8h,  v0.h[3]
        FMLA    v21.8h, v19.8h,  v0.h[3]
        FMLA    v22.8h, v18.8h,  v1.h[3]
        FMLA    v23.8h, v19.8h,  v1.h[3]
        FMLA    v24.8h, v18.8h,  v2.h[3]
        FMLA    v25.8h, v19.8h,  v2.h[3]
        FMLA    v26.8h, v18.8h,  v3.h[3]
        FMLA    v27.8h, v19.8h,  v3.h[3]
        FMLA    v28.8h, v18.8h,  v4.h[3]
        FMLA    v29.8h, v19.8h,  v4.h[3]
        FMLA    v30.8h, v18.8h,  v5.h[3]
        FMLA    v31.8h, v19.8h,  v5.h[3]
        B.NE    M6N16RemainderK123         // remaining k 1~3

M6N16NextIterN
        SUBS    x1, x1, 16
        B.LO    M6StoreRemainderN

        ldr     x8,[sp,#HGemmKernelFrame_B]
        add     x8,x8,32                // B <- next 16 columns
        str     x8,[sp,#HGemmKernelFrame_B]

        cbnz    x19,M6N16SkipAccumulateOutput
        ldp     q0,q1,[x3]
        ldp     q2,q3,[x16]
        ldp     q4,q5,[x17]
        ldp     q6,q7,[x14]
        ldp     q16,q17,[x13]
        ldp     q18,q19,[x4]
        fadd    v20.8h,v20.8h,v0.8h
        fadd    v21.8h,v21.8h,v1.8h
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
        ST1     {v20.16b, v21.16b},  [x3], 32
        SUB     x6,  x6, x2             // a0 -= k
        ST1     {v22.16b, v23.16b}, [x16], 32
        SUB     x9,  x9, x2             // a1 -= k
        ST1     {v24.16b, v25.16b}, [x17], 32
        SUB     x10, x10, x2            // a2 -= k
        ST1     {v26.16b, v27.16b}, [x14], 32
        SUB     x11, x11, x2            // a3 -= k
        ST1     {v28.16b, v29.16b}, [x13], 32
        SUB     x12, x12, x2            // a4 -= k
        ST1     {v30.16b, v31.16b},  [x4], 32
        SUB     x7,  x7, x2             // a5 -= k
        B.HI    M6N16OutterLoopN

ExitKernel
        EPILOG_RESTORE_REG       x19,#HGemmKernelFrame_SavedRegs!
        EPILOG_RETURN

M6N16RemainderK123
        TBZ     x0, 2, M6N16RemainderK1
        LDR     s0,  [x6], 4
        LDR     q16, [x8], 16
        ld1     {v17.16b},[x8],x15
        LDR     s1,  [x9], 4
        LDR     s2, [x10], 4
        LDR     s3, [x11], 4
        LDR     s4, [x12], 4
        LDR     s5,  [x7], 4
        LDR     q18, [x8], 16
        ld1     {v19.16b},[x8],x15
        FMLA    v20.8h, v16.8h,  v0.h[0]
        FMLA    v22.8h, v16.8h,  v1.h[0]
        FMLA    v24.8h, v16.8h,  v2.h[0]
        FMLA    v26.8h, v16.8h,  v3.h[0]
        FMLA    v28.8h, v16.8h,  v4.h[0]
        FMLA    v30.8h, v16.8h,  v5.h[0]
        FMLA    v21.8h, v17.8h,  v0.h[0]
        FMLA    v23.8h, v17.8h,  v1.h[0]
        FMLA    v25.8h, v17.8h,  v2.h[0]
        FMLA    v27.8h, v17.8h,  v3.h[0]
        FMLA    v29.8h, v17.8h,  v4.h[0]
        FMLA    v31.8h, v17.8h,  v5.h[0]

        FMLA    v20.8h, v18.8h,  v0.h[1]
        FMLA    v22.8h, v18.8h,  v1.h[1]
        FMLA    v24.8h, v18.8h,  v2.h[1]
        FMLA    v26.8h, v18.8h,  v3.h[1]
        FMLA    v28.8h, v18.8h,  v4.h[1]
        FMLA    v30.8h, v18.8h,  v5.h[1]
        FMLA    v21.8h, v19.8h,  v0.h[1]
        FMLA    v23.8h, v19.8h,  v1.h[1]
        FMLA    v25.8h, v19.8h,  v2.h[1]
        FMLA    v27.8h, v19.8h,  v3.h[1]
        FMLA    v29.8h, v19.8h,  v4.h[1]
        FMLA    v31.8h, v19.8h,  v5.h[1]
        TBZ     x0, 1, M6N16NextIterN

M6N16RemainderK1
        LDR     h0,  [x6], 2
        LDR     q16, [x8], 16
        ld1     {v17.16b},[x8],x15
        LDR     h1,  [x9], 2
        LDR     h2, [x10], 2
        LDR     h3, [x11], 2
        LDR     h4, [x12], 2
        LDR     h5,  [x7], 2
        FMLA    v20.8h, v16.8h,  v0.h[0]
        FMLA    v22.8h, v16.8h,  v1.h[0]
        FMLA    v24.8h, v16.8h,  v2.h[0]
        FMLA    v26.8h, v16.8h,  v3.h[0]
        FMLA    v28.8h, v16.8h,  v4.h[0]
        FMLA    v30.8h, v16.8h,  v5.h[0]
        FMLA    v21.8h, v17.8h,  v0.h[0]
        FMLA    v23.8h, v17.8h,  v1.h[0]
        FMLA    v25.8h, v17.8h,  v2.h[0]
        FMLA    v27.8h, v17.8h,  v3.h[0]
        FMLA    v29.8h, v17.8h,  v4.h[0]
        FMLA    v31.8h, v17.8h,  v5.h[0]
        B       M6N16NextIterN

M6StoreRemainderN
        TBZ     x1, 3, M6StoreRemainderN
        cbnz    x19,M6N8SkipAccumulateOutput
        ldr     q0,[x3]
        ldr     q1,[x16]
        ldr     q2,[x17]
        ldr     q3,[x14]
        ldr     q4,[x13]
        ldr     q5,[x4]
        fadd    v20.8h,v20.8h,v0.8h
        fadd    v22.8h,v22.8h,v1.8h
        fadd    v24.8h,v24.8h,v2.8h
        fadd    v26.8h,v26.8h,v3.8h
        fadd    v28.8h,v28.8h,v4.8h
        fadd    v30.8h,v30.8h,v5.8h

M6N8SkipAccumulateOutput
        STR     q20,  [x3], 16
        MOV     v20.16b, v21.16b
        STR     q22, [x16], 16
        MOV     v22.16b, v23.16b
        STR     q24, [x17], 16
        MOV     v24.16b, v25.16b
        STR     q26, [x14], 16
        MOV     v26.16b, v27.16b
        STR     q28, [x13], 16
        MOV     v28.16b, v29.16b
        STR     q30,  [x4], 16
        MOV     v30.16b, v31.16b

M6StoreRemainderN4
        TBZ     x1, 2, M6StoreRemainderN2
        cbnz    x19,M6N4SkipAccumulateOutput
        ldr     d0,[x3]
        ldr     d1,[x16]
        ldr     d2,[x17]
        ldr     d3,[x14]
        ldr     d4,[x13]
        ldr     d5,[x4]
        fadd    v20.4h,v20.4h,v0.4h
        fadd    v22.4h,v22.4h,v1.4h
        fadd    v24.4h,v24.4h,v2.4h
        fadd    v26.4h,v26.4h,v3.4h
        fadd    v28.4h,v28.4h,v4.4h
        fadd    v30.4h,v30.4h,v5.4h

M6N4SkipAccumulateOutput
        STR     d20,  [x3], 8
        STR     d22, [x16], 8
        DUP     d20, v20.d[1]
        DUP     d22, v22.d[1]
        STR     d24, [x17], 8
        STR     d26, [x14], 8
        DUP     d24, v24.d[1]
        DUP     d26, v26.d[1]
        STR     d28, [x13], 8
        STR     d30,  [x4], 8
        DUP     d28, v28.d[1]
        DUP     d30, v30.d[1]

M6StoreRemainderN2
        TBZ     x1, 1, M6StoreRemainderN1
        cbnz    x19,M6N2SkipAccumulateOutput
        ldr     s0,[x3]
        ldr     s1,[x16]
        ldr     s2,[x17]
        ldr     s3,[x14]
        ldr     s4,[x13]
        ldr     s5,[x4]
        fadd    v20.4h,v20.4h,v0.4h
        fadd    v22.4h,v22.4h,v1.4h
        fadd    v24.4h,v24.4h,v2.4h
        fadd    v26.4h,v26.4h,v3.4h
        fadd    v28.4h,v28.4h,v4.4h
        fadd    v30.4h,v30.4h,v5.4h

M6N2SkipAccumulateOutput
        STR     s20,  [x3], 4
        STR     s22, [x16], 4
        DUP     s20, v20.s[1]
        DUP     s22, v22.s[1]
        STR     s24, [x17], 4
        STR     s26, [x14], 4
        DUP     s24, v24.s[1]
        DUP     s26, v26.s[1]
        STR     s28, [x13], 4
        STR     s30,  [x4], 4
        DUP     s28, v28.s[1]
        DUP     s30, v30.s[1]

M6StoreRemainderN1
        TBZ     x1, 0, ExitKernel
        cbnz    x19,M6N1SkipAccumulateOutput
        ldr     h0,[x3]
        ldr     h1,[x16]
        ldr     h2,[x17]
        ldr     h3,[x14]
        ldr     h4,[x13]
        ldr     h5,[x4]
        fadd    v20.4h,v20.4h,v0.4h
        fadd    v22.4h,v22.4h,v1.4h
        fadd    v24.4h,v24.4h,v2.4h
        fadd    v26.4h,v26.4h,v3.4h
        fadd    v28.4h,v28.4h,v4.4h
        fadd    v30.4h,v30.4h,v5.4h

M6N1SkipAccumulateOutput
        STR     h20,  [x3]
        STR     h22, [x16]
        STR     h24, [x17]
        STR     h26, [x14]
        STR     h28, [x13]
        STR     h30,  [x4]
        b       ExitKernel

        LEAF_END MlasHalfGemmKernelNeon

        END
