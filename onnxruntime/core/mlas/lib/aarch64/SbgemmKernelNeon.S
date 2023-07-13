/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    SbgemmKernelNeon.s

Abstract:

    This module implements the kernels for the bfloat16 half precision matrix/matrix
    multiply operation (SBGEMM).

--*/

#include "asmmacro.h"

        .text

//
// Stack frame layout for the sbgemm kernel. d8-d15, x19-x30 need save
//
        .equ  .LMlasSbgemmKernel_backup_x19_x20,    0
        .equ  .LMlasSbgemmKernel_backup_x21_x22,    16
        .equ  .LMlasSbgemmKernel_backup_x23_x24,    32
        .equ  .LMlasSbgemmKernel_backup_x25_x26,    48
        .equ  .LMlasSbgemmKernel_backup_x27_x28,    64
        .equ  .LMlasSbgemmKernel_backup_d8_d9,      80
        .equ  .LMlasSbgemmKernel_backup_d10_d11,    96
        .equ  .LMlasSbgemmKernel_backup_d12_d13,    112
        .equ  .LMlasSbgemmKernel_backup_d14_d15,    128
        .equ  .LMlasSbgemmKernel_SavedRegisters,    144
        .equ  .LMlasSbgemmKernel_SavedRegisters_Neg, -144


//
// ClearRowAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearRowAccumulators Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        movi    v\Vec1Reg\().16b,#0
.if \Columns\() > 2
        movi    v\Vec2Reg\().16b,#0
.endif
.if \Columns\() > 4
        movi    v\Vec3Reg\().16b,#0
.endif
.if \Columns\() > 6
        movi    v\Vec4Reg\().16b,#0
.endif

        .endm

//
// ClearBlockAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearBlockAccumulators Columns, Rows

        ClearRowAccumulators \Columns\(),16,17,18,19
.if \Rows\() > 2
        ClearRowAccumulators \Columns\(),20,21,22,23
.endif
.if \Rows\() > 4
        ClearRowAccumulators \Columns\(),24,25,26,27
.endif
.if \Rows\() > 6
        ClearRowAccumulators \Columns\(),28,29,30,31
.endif

        .endm

// LoadMatrixAElementsBy8
//
// Generates the code to load 4 or 8 elements from matrix A.
//
        .macro  LoadMatrixAElementsBy8 Rows

        ldr     q8,[x0],#16
        bfcvtn  v8.4h, v8.4s
.if \Rows\() > 1
        ldr     q1,[x10],#16
        bfcvtn2 v8.8h, v1.4s
.endif

.if \Rows\() > 2
        ldr     q9,[x11],#16
        bfcvtn  v9.4h, v9.4s
.endif
.if \Rows\() > 3
        ldr     q1,[x12],#16
        bfcvtn2 v9.8h, v1.4s
.endif

.if \Rows\() > 4
        ldr     q10,[x20],#16
        bfcvtn  v10.4h, v10.4s
.endif
.if \Rows\() > 5
        ldr     q1,[x21],#16
        bfcvtn2 v10.8h, v1.4s
.endif

.if \Rows\() > 6
        ldr     q11,[x22],#16
        bfcvtn  v11.4h, v11.4s
.endif
.if \Rows\() > 7
        ldr     q1,[x23],#16
        bfcvtn2 v11.8h, v1.4s
.endif

        .endm


//
// MultiplyAccumulateRow
//
// Generates the code to multiply and accumulate a single row of the output
// block.
//

        .macro  MultiplyAccumulateRow Columns, MatrixAReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        bfmmla    v\Vec1Reg\().4s, \MatrixAReg\().8h, v4.8h
.if \Columns\() > 2
        bfmmla    v\Vec2Reg\().4s, \MatrixAReg\().8h, v5.8h
.endif
.if \Columns\() > 4
	bfmmla    v\Vec3Reg\().4s, \MatrixAReg\().8h, v6.8h
.endif
.if \Columns\() > 6
        bfmmla    v\Vec4Reg\().4s, \MatrixAReg\().8h, v7.8h
.endif

        .endm

//
// MultiplyAccumulateBlock
//
// Generates the code to multiply and accumulate into the output block.
//

        .macro  MultiplyAccumulateBlock Columns, Rows

        MultiplyAccumulateRow \Columns\(),v8,16,17,18,19
.if \Rows\() > 2
        MultiplyAccumulateRow \Columns\(),v9,20,21,22,23
.endif
.if \Rows\() > 4
        MultiplyAccumulateRow \Columns\(),v10,24,25,26,27
.endif
.if \Rows\() > 6
        MultiplyAccumulateRow \Columns\(),v11,28,29,30,31
.endif

        .endm

//
// ComputeBlockLoop
//
// Generates the code to loop over K entries of the input matrices to produce
// the output block.
//

        .macro  ComputeBlockLoop Mode, Columns, Rows

        ClearBlockAccumulators \Columns\(),\Rows\()

        add     x10,x0,x6,lsl #2            // compute matrix A plus 1 row
.if \Rows\() > 2
        add     x11,x10,x6,lsl #2           // compute matrix A plus 2 rows
        add     x12,x11,x6,lsl #2           // compute matrix A plus 3 rows
.endif
.if \Rows\() > 4
        add     x20,x12,x6,lsl #2           // compute matrix A plus 4 rows
        add     x21,x20,x6,lsl #2           // compute matrix A plus 5 rows
.endif
.if \Rows\() > 6
        add     x22,x21,x6,lsl #2           // compute matrix A plus 6 rows
        add     x23,x22,x6,lsl #2           // compute matrix A plus 7 rows
.endif
        sub    x9,x3,#4                   //  block count to process
        tbnz    x9,#7,.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop:

        LoadMatrixAElementsBy8 \Rows\()
        ldr     q4, [x1],#16
.if \Columns\() > 2
	ldr     q5,[x1],#16
.endif
.if \Columns\() > 4
        ldr     q6,[x1],#16
.endif
.if \Columns\() > 6
        ldr     q7,[x1],#16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\()

        sub     x9,x9,#4
        tbz     x9,#7,.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop
.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks:
        add     x9,x9,#4                    // correct for over-subtract above
        cbz     x9,.L\Mode\().Output\Columns\().x\Rows\().Block

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4PaddedLoop:
        LoadMatrixAElementsBy8 \Rows\()
        ldr     q4, [x1],#16
.if \Columns\() > 2
        ldr     q5,[x1],#16
.endif
.if \Columns\() > 4
        ldr     q6,[x1],#16
.endif
.if \Columns\() > 6
        ldr     q7,[x1],#16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\()

.L\Mode\().Output\Columns\().x\Rows\().Block:

        .endm

//
// MultiplyAlphaRow
//
// Generates the code to multiply a single row of the output block by the alpha
// value.
//

        .macro  MultiplyAlphaRow Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        fmul    v\Vec1Reg\().4s,v\Vec1Reg\().4s,v0.s[0]
.if \Columns\() > 2
        fmul    v\Vec2Reg\().4s,v\Vec2Reg\().4s,v0.s[0]
.endif
.if \Columns\() > 4
        fmul    v\Vec3Reg\().4s,v\Vec3Reg\().4s,v0.s[0]
.endif
.if \Columns\() > 6
        fmul    v\Vec4Reg\().4s,v\Vec4Reg\().4s,v0.s[0]
.endif

        .endm

//
// MultiplyAlphaBlock
//
// Generates the code to multiply the output block by the alpha value.
//

        .macro  MultiplyAlphaBlock Columns, Rows

        MultiplyAlphaRow \Columns\(),16,17,18,19
.if \Rows\() > 2
        MultiplyAlphaRow \Columns\(),20,21,22,23
.endif
.if \Rows\() > 4
        MultiplyAlphaRow \Columns\(),24,25,26,27
.endif
.if \Rows\() > 6
        MultiplyAlphaRow \Columns\(),28,29,30,31
.endif

        .endm

//
// OutputRow2Element
// OutputRow4Element
// OutputRow6Element
// OutputRow8Element
// OutputRow10Element
// OutputRow12Element
// OutputRow14Element
// OutputRow16Element
//
// Generates the code to store elements to the output block.
//

        .macro  OutputRow2Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#4
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#4
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif
        mov v8.4S[2], v9.4S[0]
        fmla    v8.4s,v\Vec1Reg\().4s,v0.s[0]

        mov     w27, v8.4S[0]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     w27, v8.4S[2]
        str   w27, [\AddrReg2\()],#4
.endif

.else
        mov     w27, v\Vec1Reg\().4S[0]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     w27, v\Vec1Reg\().4S[2]
        str   w27, [\AddrReg2\()],#4
.endif

.endif

        .endm


        .macro  OutputRow4Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#8
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#8
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif

        mov v8.2D[1], v9.2D[0]

        fmla    v8.4s,v\Vec1Reg\().4s,v0.s[0]

        mov     x27, v8.2D[0]
        mov     x28, v8.2D[1]

        str      x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str      x28, [\AddrReg2\()],#8
.endif

.else
        mov     x27, v\Vec1Reg\().2D[0]
        mov     x28, v\Vec1Reg\().2D[1]

        str      x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str      x28, [\AddrReg2\()],#8
.endif

.endif

        .endm


        .macro  OutputRow6Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#12
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#12
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]

        mov     x27, v8.2D[0]
        str   x27, [\AddrReg1\()],#8
        mov     w27, v8.4S[2]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     x27, v9.2D[0]
        str   x27, [\AddrReg2\()],#8
        mov     w27, v9.4S[2]
        str   w27, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        mov     x27, v4.2D[0]
        str   x27, [\AddrReg1\()],#8
        mov     w27, v4.4S[2]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     x27, v5.2D[0]
        str   x27, [\AddrReg2\()],#8
        mov     w27, v5.4S[2]
        str   w27, [\AddrReg2\()],#4
.endif

.endif

        .endm


        .macro  OutputRow8Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]

        str     q8,[\AddrReg1\()],#16
.if \last_row\() != 0
        str     q9,[\AddrReg2\()],#16
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        str     q4,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
.endif

.endif

        .endm


        .macro  OutputRow10Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]

        str     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#16
.endif

        ldr     q8,[\AddrReg1\()],#4
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#4
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif

        mov v8.4S[2],v9.4S[0]

        fmla    v8.4s,v\Vec3Reg\().4s,v0.s[0]

        mov     w27, v8.4S[0]
        mov     w28, v8.4S[2]

        str      w27, [\AddrReg1\()],#4
.if \last_row\() == 0
        str      w28, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        str     q4,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
.endif
        mov     w27, v\Vec3Reg\().4S[0]
        mov     w28, v\Vec3Reg\().4S[2]

        str      w27, [\AddrReg1\()],#4
.if \last_row\() == 0
        str      w28, [\AddrReg2\()],#4
.endif
.endif

.endm


        .macro  OutputRow12Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]

        str     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#16
.endif

        mov v8.2D[1], v9.2D[0]

        fmla    v8.4s,v\Vec3Reg\().4s,v0.s[0]

        mov     x27, v8.2D[0]
        mov     x28, v8.2D[1]

        str      x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str      x28, [\AddrReg2\()],#8
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        str     q4,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
.endif
        mov     x27, v\Vec3Reg\().2D[0]
        mov     x28, v\Vec3Reg\().2D[1]

        str	 x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str      x28, [\AddrReg2\()],#8
.endif
.endif

        .endm

       .macro  OutputRow14Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
        ldr     q10,[\AddrReg1\()],#12
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
        ldr     q11,[\AddrReg1\()],#12
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27

        mov v11.2D[0],x27
        mov v11.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]
        fmla    v10.4s,v6.4s,v0.s[0]
        fmla    v11.4s,v7.4s,v0.s[0]

        str     q8,[\AddrReg1\()],#16

        mov     x27, v10.2D[0]
        str   x27, [\AddrReg1\()],#8
        mov     w27, v10.4S[2]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#32
        mov     x27, v11.2D[0]
        str   x27, [\AddrReg2\()],#8
        mov     w27, v11.4S[2]
        str   w27, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        str     q4,[\AddrReg1\()],#16
        mov     x27, v6.2D[0]
        str   x27, [\AddrReg1\()],#8
        mov     w27, v6.4S[2]
        str   w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
        mov     x27, v7.2D[0]
        str   x27, [\AddrReg2\()],#8
        mov     w27, v7.4S[2]
        str   w27, [\AddrReg2\()],#4
.endif
.endif

        .endm


        .macro  OutputRow16Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldp     q8,q10,[\AddrReg1\()],#32
.if \last_row\() != 1
        ldp     q9,q11,[\AddrReg2\()],#32
.else
        mov x27,#0
        mov v9.2D[0],x27
        mov v9.2D[1],x27

        mov v11.2D[0],x27
        mov v11.2D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        fmla    v8.4s,v4.4s,v0.s[0]
        fmla    v9.4s,v5.4s,v0.s[0]
        fmla    v10.4s,v6.4s,v0.s[0]
        fmla    v11.4s,v7.4s,v0.s[0]

        stp     q8,q10,[\AddrReg1\()],#32
.if \last_row\() == 0
        stp     q9,q11,[\AddrReg2\()],#32
.endif
.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        stp 	q4,q6,[\AddrReg1\()],#32
.if \last_row\() == 0
        stp	q5,q7,[\AddrReg2\()],#32
.endif
.endif

        .endm

//
// OutputBlock
//
// Generates the code to store the output block.
//

        .macro  OutputBlock Mode, Columns, Rows

        OutputRow\Columns\()Element \Mode\(),x2,x13,16,17,18,19,(\Rows\() == 1)

.if \Rows\() > 2
        OutputRow\Columns\()Element \Mode\(),x14,x15,20,21,22,23,(\Rows\() == 3)
.endif

.if \Rows\() > 4
        OutputRow\Columns\()Element \Mode\(),x16,x17,24,25,26,27,(\Rows\() == 5)
.endif

.if \Rows\() > 6
        OutputRow\Columns\()Element \Mode\(),x18,x19,28,29,30,31,(\Rows\() == 7)
.endif

        .endm
//
// ProcessRows
//
// Generates the code to process a compute and store the output block for a
// fixed number of rows.
//

        .macro  ProcessRows Mode, Rows
        mov     x4,#\Rows\()                   // return number of rows handled
        cmp     x5,#6
        ble     .L\Mode\().ProcessNextColumnLoop6x\Rows\()

.L\Mode\().ProcessNextColumnLoop8x\Rows\():
        ComputeBlockLoop \Mode\(),8,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 8,\Rows\()
.endif

        sub     x5,x5,#8
        cmp   x5,#0
        blt   .L\Mode\().Output14ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),16,\Rows\()
        mov     x0,x8               // reload matrix A
        cmp     x5,#6
        bgt     .L\Mode\().ProcessNextColumnLoop8x\Rows\()
        cbz     x5,.L\Mode\().ExitKernel


.L\Mode\().ProcessNextColumnLoop6x\Rows\():

        cmp    x5,#4
        ble .L\Mode\().ProcessNextColumnLoop4x\Rows\()
        ComputeBlockLoop \Mode\(),6,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 6,\Rows\()
.endif
        sub 	x5,x5,#6
                cmp   x5,#0
        blt   .L\Mode\().Output10ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),12,\Rows\()

        mov     x0,x8               // reload matrix A
        cmp     x5,#4
        bgt     .L\Mode\().ProcessNextColumnLoop6x\Rows\()
        b     .L\Mode\().ExitKernel

.L\Mode\().ProcessNextColumnLoop4x\Rows\():
        cmp     x5,#2
        ble .L\Mode\().ProcessNextColumnLoop2x\Rows\()
        ComputeBlockLoop \Mode\(),4,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 4,\Rows\()
.endif
        sub     x5,x5,#4
        cmp   x5,#0
        blt   .L\Mode\().Output6ElementsOnlyFor\Rows\()

        OutputBlock \Mode\(),8,\Rows\()

        mov     x0,x8               // reload matrix A
        cmp     x5,#2
        bgt     .L\Mode\().ProcessNextColumnLoop4x\Rows\()
        b     .L\Mode\().ExitKernel

.L\Mode\().ProcessNextColumnLoop2x\Rows\():
        ComputeBlockLoop \Mode\(),2,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 2,\Rows\()
.endif
        sub     x5,x5,#2
        cmp   x5,#0
        blt   .L\Mode\().Output2ElementsOnlyFor\Rows\()

        OutputBlock \Mode\(),4,\Rows\()

        mov     x0,x8               // reload matrix A
        cmp     x5,#2
        b     .L\Mode\().ExitKernel

.L\Mode\().Output14ElementsOnlyFor\Rows\():
	OutputBlock \Mode\(),14,\Rows\()
        b     .L\Mode\().ExitKernel


.L\Mode\().Output10ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),10,\Rows\()
        b     .L\Mode\().ExitKernel


.L\Mode\().Output6ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),6,\Rows\()
        b     .L\Mode\().ExitKernel


.L\Mode\().Output2ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),2,\Rows\()
        b     .L\Mode\().ExitKernel

        .endm


/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (x2) - Supplies the address of matrix C.

    CountK (x3) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (x4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (x5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda (x6) - Supplies the first dimension of matrix A.

    ldc (x7) - Supplies the first dimension of matrix C.

    Alpha (s0) - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

        .macro  SbgemmKernelNeonFunction Mode

        FUNCTION_ENTRY MlasSbgemmKernel\Mode\()

        stp     x19, x20, [sp, #.LMlasSbgemmKernel_SavedRegisters_Neg]!
        stp     x21, x22, [sp, #.LMlasSbgemmKernel_backup_x21_x22]
        stp     x23, x24, [sp, #.LMlasSbgemmKernel_backup_x23_x24]
        stp     x25, x26, [sp, #.LMlasSbgemmKernel_backup_x25_x26]
        stp     x27, x28, [sp, #.LMlasSbgemmKernel_backup_x27_x28]
        stp     d8, d9, [sp, #.LMlasSbgemmKernel_backup_d8_d9]
        stp     d10, d11, [sp, #.LMlasSbgemmKernel_backup_d10_d11]
        stp     d12, d13, [sp, #.LMlasSbgemmKernel_backup_d12_d13]
        stp     d14, d15, [sp, #.LMlasSbgemmKernel_backup_d14_d15]

        add     x13,x2,x7,lsl #2            // compute matrix C plus 1 row
        add     x14,x13,x7,lsl #2           // compute matrix C plus 2 rows
        add     x15,x14,x7,lsl #2           // compute matrix C plus 3 rows
        add     x16,x15,x7,lsl #2           // compute matrix C plus 4 rows
        add     x17,x16,x7,lsl #2           // compute matrix C plus 5 rows
        add     x18,x17,x7,lsl #2           // compute matrix C plus 6 rows
        add     x19,x18,x7,lsl #2           // compute matrix C plus 7 rows

        mov     x8,x0                       // save matrix A

//
// Process 8 rows of the matrices.
//
        cmp     x4,#8
        blt     .L\Mode\().ProcessCountMLessThan8
        ProcessRows \Mode\(),8

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     x0,x4

        ldp     d14, d15, [sp, #.LMlasSbgemmKernel_backup_d14_d15]
        ldp     d12, d13, [sp, #.LMlasSbgemmKernel_backup_d12_d13]
        ldp     d10, d11, [sp, #.LMlasSbgemmKernel_backup_d10_d11]
        ldp     d8, d9, [sp, #.LMlasSbgemmKernel_backup_d8_d9]
        ldp     x27, x28, [sp, #.LMlasSbgemmKernel_backup_x27_x28]
        ldp     x25, x26, [sp, #.LMlasSbgemmKernel_backup_x25_x26]
        ldp     x23, x24, [sp, #.LMlasSbgemmKernel_backup_x23_x24]
        ldp     x21, x22, [sp, #.LMlasSbgemmKernel_backup_x21_x22]
        ldp     x19, x20, [sp], #.LMlasSbgemmKernel_SavedRegisters

        ret

//
// Process 4 rows of the matrix.
//

.L\Mode\().ProcessCountMLessThan8:
        cmp     x4,#4
        blt     .L\Mode\().ProcessCountMLessThan4
        ProcessRows \Mode\(),4
        b       .L\Mode\().ExitKernel

//
// Process 2 row of the matrix.
//

.L\Mode\().ProcessCountMLessThan4:
        cmp     x4,#2
        blt     .L\Mode\().ProcessCountMLessThan2

        ProcessRows \Mode\(),2
        b       .L\Mode\().ExitKernel


//
// Process the last row of the matrix.
//

.L\Mode\().ProcessCountMLessThan2:
        ProcessRows \Mode\(),1
        b       .L\Mode\().ExitKernel


        .endm

        SbgemmKernelNeonFunction Zero
        SbgemmKernelNeonFunction Add

        .end
