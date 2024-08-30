/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelUmmla.s

Abstract:

    This module implements the kernels for the Int8 precision matrix/matrix
    multiply operation (QGEMM).

--*/

#include "asmmacro.h"

        .text

//
// Stack frame layout for the ummla kernel. d8-d15, x19-x30 need save
//
        .equ  .LMlasQgemmKernel_backup_x19_x20,    0
        .equ  .LMlasQgemmKernel_backup_x21_x22,    16
        .equ  .LMlasQgemmKernel_backup_x23_x24,    32
        .equ  .LMlasQgemmKernel_backup_x25_x26,    48
        .equ  .LMlasQgemmKernel_backup_x27_x28,    64
        .equ  .LMlasQgemmKernel_backup_d8_d9,      80
        .equ  .LMlasQgemmKernel_backup_d10_d11,    96
        .equ  .LMlasQgemmKernel_backup_d12_d13,    112
        .equ  .LMlasQgemmKernel_backup_d14_d15,    128
        .equ  .LMlasQgemmKernel_SavedRegisters,    144
        .equ  .LMlasQgemmKernel_SavedRegisters_Neg, -144


//
// Init Row Accumulators
//
// Generates the code to initialize the accumulators for a single row of the output
// block.
//
//
//  Accumulators are initialized to ZeroPointB * RowSum + ColumnSum
//  x7 for RowSumsBuffer pointer
//  x10 for ColumnSumBuffer pointer
//  x11 for ZeroPointB buffer pointer
//
//  v12~v13 for RowSums values
//  v14~v15 for ColumnSums values
//  v0~v3 for ZeroPointB values
//
        .macro  InitRowAccumulators Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, RowSumReg

        mul     v7.4s, v\RowSumReg\().4s, v8.4s
        mov     v\Vec1Reg\().16b, v7.16b
        add     v\Vec1Reg\().4s, v\Vec1Reg\().4s, v0.4s
.if \Columns\() > 2
        mul     v7.4s, v\RowSumReg\().4s, v9.4s
        mov     v\Vec2Reg\().16b, v7.16b
        add     v\Vec2Reg\().4s, v\Vec2Reg\().4s, v1.4s
.endif
.if \Columns\() > 4
        mul     v7.4s, v\RowSumReg\().4s, v10.4s
        mov     v\Vec3Reg\().16b, v7.16b
        add     v\Vec3Reg\().4s, v\Vec3Reg\().4s, v2.4s
.endif
.if \Columns\() > 6
        mul     v7.4s, v\RowSumReg\().4s, v11.4s
        mov     v\Vec4Reg\().16b, v7.16b
        add     v\Vec4Reg\().4s, v\Vec4Reg\().4s, v3.4s
.endif

        .endm


//
// InitBlockAccumulators
//
// Generates the code to initialize the accumulators for 8x8 output
// block.
//
        .macro  InitBlockAccumulators Mode, Columns, Rows

        ld1     {v14.4s},[x10],#16            // load ColumnSumBuffer[0]
.if \Columns\() > 4
        ld1     {v15.4s},[x10],#16            // load ColumnSumBuffer[4]
.endif
        // v4~v7 will be set to matrixB after this, so, they can used now
        dup     v4.4s,v14.s[0]              // broadcast column
        dup     v5.4s,v14.s[1]
        dup     v6.4s,v14.s[2]
        dup     v7.4s,v14.s[3]

        zip1    v0.4s, v4.4s, v5.4s
        zip2    v1.4s, v6.4s, v7.4s
.if \Columns\() > 4
        dup     v4.4s,v15.s[0]              // broadcast column
        dup     v5.4s,v15.s[1]
        dup     v6.4s,v15.s[2]
        dup     v7.4s,v15.s[3]

        zip1    v2.4s, v4.4s, v5.4s
        zip2    v3.4s, v6.4s, v7.4s
.endif

        // v8~v11 will anyway get set in MatrixA loading, so they are free to use now
        movi    v8.4s, #1
        movi    v9.4s, #1
        movi    v10.4s, #1
        movi    v11.4s, #1

        cbz     x11,.L\Mode\().InitBlock\Columns\().x\Rows\().SkipScaleByZeroPointB

        ld1     {v4.4s},[x11],#16           // load ZeroPointB[0]
        ld1     {v5.4s},[x11],#16           // load ZeroPointB[4]

        dup     v6.4s, v4.s[0]
        dup     v7.4s, v4.s[1]
        zip1    v8.4s, v6.4s, v7.4s

        dup     v6.4s, v4.s[2]
        dup     v7.4s, v4.s[3]
        zip1    v9.4s, v6.4s, v7.4s

        dup     v6.4s, v5.s[0]
        dup     v7.4s, v5.s[1]
        zip1    v10.4s, v6.4s, v7.4s

        dup     v6.4s, v5.s[2]
        dup     v7.4s, v5.s[3]
        zip1    v11.4s, v6.4s, v7.4s

.L\Mode\().InitBlock\Columns\().x\Rows\().SkipScaleByZeroPointB:
        dup     v4.4s, v12.s[0]           //boardcast RowSums
        dup     v5.4s, v12.s[1]

        uzp1    v6.2d, v4.2d, v5.2d

        InitRowAccumulators \Columns\(),16,17,18,19,6
.if \Rows\() > 2
        dup     v4.4s, v12.s[2]           //boardcast RowSums
        dup     v5.4s, v12.s[3]

        uzp1    v6.2d, v4.2d, v5.2d

        InitRowAccumulators \Columns\(),20,21,22,23,6
.endif
.if \Rows\() > 4
        dup     v4.4s,v13.s[0]         // broadcast row sums
        dup     v5.4s,v13.s[1]

        uzp1    v6.2d, v4.2d, v5.2d

        InitRowAccumulators \Columns\(),24,25,26,27,6
.endif
.if \Rows\() > 6
        dup     v4.4s,v13.s[2]         // broadcast row sums
        dup     v5.4s,v13.s[3]

        uzp1    v6.2d, v4.2d, v5.2d
        InitRowAccumulators \Columns\(),28,29,30,31,6
.endif

        .endm


// LoadPackedMatrixABy16Elements
//
// Generates the code to load 16 elements from matrix A.
//
        .macro  LoadPackedMatrixABy16Elements Rows
.if \Rows\() == 1
        ldr     q8,[x0],#8
.else
        ldr     q8,[x0],#16

.if \Rows\() > 2
        ldr     q9,[x0],#16
.endif

.if \Rows\() > 4
        ldr     q10,[x0],#16
.endif

.if \Rows\() > 6
        ldr     q11,[x0],#16
.endif
.endif
        .endm


//
// MultiplyAccumulateRow
//
// Generates the code to multiply and accumulate a single row of the output
// block.
//

        .macro  MultiplyAccumulateRow Columns, MatrixAReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        ummla   v\Vec1Reg\().4s, \MatrixAReg\().16b, v4.16b
.if \Columns\() > 2
        ummla   v\Vec2Reg\().4s, \MatrixAReg\().16b, v5.16b
.endif
.if \Columns\() > 4
	ummla   v\Vec3Reg\().4s, \MatrixAReg\().16b, v6.16b
.endif
.if \Columns\() > 6
        ummla   v\Vec4Reg\().4s, \MatrixAReg\().16b, v7.16b
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

        InitBlockAccumulators \Mode\(), \Columns\(),\Rows\()

        sub     x9,x3,#1                   //  block count to process
        tbnz    x9,#63,.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop:

        LoadPackedMatrixABy16Elements \Rows\()
        ld1     {v4.16b - v7.16b}, [x1], #64
        MultiplyAccumulateBlock \Columns\(),\Rows\()

        sub     x9,x9,#1
        tbz     x9,#63,.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop
.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks:
        add     x9,x9,#1                    // correct for over-subtract above
        cbz     x9,.L\Mode\().Output\Columns\().x\Rows\().Block

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4PaddedLoop:
        LoadPackedMatrixABy16Elements \Rows\()
        ld1     {v4.16b - v7.16b}, [x1], #64
        MultiplyAccumulateBlock \Columns\(),\Rows\()

.L\Mode\().Output\Columns\().x\Rows\().Block:

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
        ldr     s8,[\AddrReg1\()],#0
.if \last_row\() == 0
        ldr     s9,[\AddrReg2\()],#0
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
.endif
        mov     v8.S[2], v9.S[0]
        add     v8.4s,v8.4s,v\Vec1Reg\().4s

        mov     w27, v8.S[0]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     w27, v8.S[2]
        str     w27, [\AddrReg2\()],#4
.endif

.else
        mov     w27, v\Vec1Reg\().S[0]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov    w27, v\Vec1Reg\().S[2]
        str    w27, [\AddrReg2\()],#4
.endif

.endif

        .endm


        .macro  OutputRow4Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     d8,[\AddrReg1\()],#0
.if \last_row\() == 0
        ldr     d9,[\AddrReg2\()],#0
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
.endif

        mov     v8.D[1], v9.D[0]

        add     v8.4s,v8.4s,v\Vec1Reg\().4s

        mov     x27, v8.D[0]
        mov     x28, v8.D[1]

        str     x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str     x28, [\AddrReg2\()],#8
.endif

.else
        mov     x27, v\Vec1Reg\().D[0]
        mov     x28, v\Vec1Reg\().D[1]

        str     x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str     x28, [\AddrReg2\()],#8
.endif

.endif

        .endm


        .macro  OutputRow6Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     d8,[\AddrReg1\()],#8
        ldr     w28,[\AddrReg1\()],#-8
        mov     v8.S[2], w28
.if \last_row\() == 0
        ldr     d9,[\AddrReg2\()],#8
        ldr     w27,[\AddrReg2\()],#-8
        mov     v9.S[2], w27
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s

        mov     x27, v8.D[0]
        str     x27, [\AddrReg1\()],#8
        mov     w27, v8.S[2]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     x27, v9.D[0]
        str     x27, [\AddrReg2\()],#8
        mov     w27, v9.S[2]
        str     w27, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        mov     x27, v4.D[0]
        str     x27, [\AddrReg1\()],#8
        mov     w27, v4.S[2]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        mov     x27, v5.D[0]
        str     x27, [\AddrReg2\()],#8
        mov     w27, v5.S[2]
        str     w27, [\AddrReg2\()],#4
.endif

.endif

        .endm


        .macro  OutputRow8Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#0
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#0
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s

        str     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
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
        ldr     w28, [\AddrReg1\()],#-16

.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
        ldr     w27,[\AddrReg2\()],#-16
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s

        str     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#16
.endif
        mov     v8.S[0], w28
        mov     v8.S[2], w27

        add     v8.4s,v8.4s,v\Vec3Reg\().4s

        mov     w27, v8.S[0]
        mov     w28, v8.S[2]

        str     w27, [\AddrReg1\()],#4
.if \last_row\() == 0
        str     w28, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        str     q4,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
.endif
        mov     w27, v\Vec3Reg\().S[0]
        mov     w28, v\Vec3Reg\().S[2]

        str     w27, [\AddrReg1\()],#4
.if \last_row\() == 0
        str     w28, [\AddrReg2\()],#4
.endif
.endif

.endm


        .macro  OutputRow12Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
        ldr     d10,[\AddrReg1\()],#-16
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
        ldr     d11,[\AddrReg2\()],#-16
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27
        mov     v11.D[0],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s

        str     q8,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#16
.endif

        mov v10.D[1], v11.D[0]

        add     v10.4s,v10.4s,v\Vec3Reg\().4s

        mov     x27, v10.D[0]
        mov     x28, v10.D[1]

        str     x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str     x28, [\AddrReg2\()],#8
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d

        str     q4,[\AddrReg1\()],#16
.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
.endif
        mov     x27, v\Vec3Reg\().D[0]
        mov     x28, v\Vec3Reg\().D[1]

        str     x27, [\AddrReg1\()],#8
.if \last_row\() == 0
        str     x28, [\AddrReg2\()],#8
.endif
.endif

        .endm

       .macro  OutputRow14Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldr     q8,[\AddrReg1\()],#16
        ldr     d10,[\AddrReg1\()],#8
        ldr     w28, [\AddrReg1\()],#-24
        mov     v10.S[2], w28
.if \last_row\() == 0
        ldr     q9,[\AddrReg2\()],#16
        ldr     d11,[\AddrReg2\()],#8
        ldr     w27,[\AddrReg2\()],#-24
        mov     v11.S[2], w27
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27

        mov     v11.D[0],x27
        mov     v11.D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s
        add     v10.4s,v10.4s,v6.4s
        add     v11.4s,v11.4s,v7.4s

        str     q8,[\AddrReg1\()],#16

        mov     x27, v10.D[0]
        str     x27, [\AddrReg1\()],#8
        mov     w27, v10.S[2]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        str     q9,[\AddrReg2\()],#16
        mov     x27, v11.D[0]
        str     x27, [\AddrReg2\()],#8
        mov     w27, v11.S[2]
        str     w27, [\AddrReg2\()],#4
.endif

.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        str     q4,[\AddrReg1\()],#16
        mov     x27, v6.D[0]
        str     x27, [\AddrReg1\()],#8
        mov     w27, v6.S[2]
        str     w27, [\AddrReg1\()],#4

.if \last_row\() == 0
        str     q5,[\AddrReg2\()],#16
        mov     x27, v7.D[0]
        str     x27, [\AddrReg2\()],#8
        mov     w27, v7.S[2]
        str     w27, [\AddrReg2\()],#4
.endif
.endif

        .endm


        .macro  OutputRow16Element Mode, AddrReg1, AddrReg2, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg, last_row

.ifeqs "\Mode\()","Add"
        ldp     q8,q10,[\AddrReg1\()],#0
.if \last_row\() == 0
        ldp     q9,q11,[\AddrReg2\()],#0
.else
        mov     x27,#0
        mov     v9.D[0],x27
        mov     v9.D[1],x27

        mov     v11.D[0],x27
        mov     v11.D[1],x27
.endif
        uzp1    v4.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d,v\Vec1Reg\().2d,v\Vec2Reg\().2d

        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        add     v8.4s,v8.4s,v4.4s
        add     v9.4s,v9.4s,v5.4s
        add     v10.4s,v10.4s,v6.4s
        add     v11.4s,v11.4s,v7.4s

        stp     q8,q10,[\AddrReg1\()],#32
.if \last_row\() == 0
        stp     q9,q11,[\AddrReg2\()],#32
.endif
.else
        uzp1    v4.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp2    v5.2d, v\Vec1Reg\().2d,v\Vec2Reg\().2d
        uzp1    v6.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d
        uzp2    v7.2d, v\Vec3Reg\().2d,v\Vec4Reg\().2d

        stp     q4,q6,[\AddrReg1\()],#32
.if \last_row\() == 0
        stp     q5,q7,[\AddrReg2\()],#32
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

        sub     x5,x5,#8
        cmp     x5,#0
        blt     .L\Mode\().Output14ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),16,\Rows\()
        mov     x0,x8               // reload matrix A
        cmp     x5,#6
        bgt     .L\Mode\().ProcessNextColumnLoop8x\Rows\()
        cbz     x5,.L\Mode\().ExitKernel

.L\Mode\().ProcessNextColumnLoop6x\Rows\():

        cmp     x5,#4
        ble     .L\Mode\().ProcessNextColumnLoop4x\Rows\()
        ComputeBlockLoop \Mode\(),6,\Rows\()
        sub     x5,x5,#6
        cmp     x5,#0
        blt     .L\Mode\().Output10ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),12,\Rows\()
        mov     x0,x8               // reload matrix A
        cmp     x5,#4
        bgt     .L\Mode\().ProcessNextColumnLoop6x\Rows\()
        b       .L\Mode\().ExitKernel

.L\Mode\().ProcessNextColumnLoop4x\Rows\():
        cmp     x5,#2
        ble     .L\Mode\().ProcessNextColumnLoop2x\Rows\()
        ComputeBlockLoop \Mode\(),4,\Rows\()
        sub     x5,x5,#4
        cmp     x5,#0
        blt     .L\Mode\().Output6ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),8,\Rows\()
        mov     x0,x8               // reload matrix A
        cmp     x5,#2
        bgt     .L\Mode\().ProcessNextColumnLoop4x\Rows\()
        b       .L\Mode\().ExitKernel

.L\Mode\().ProcessNextColumnLoop2x\Rows\():
        ComputeBlockLoop \Mode\(),2,\Rows\()
        sub     x5,x5,#2
        cmp     x5,#0
        blt     .L\Mode\().Output2ElementsOnlyFor\Rows\()
        OutputBlock \Mode\(),4,\Rows\()
        mov     x0,x8               // reload matrix A
        cmp     x5,#2
        b       .L\Mode\().ExitKernel

.L\Mode\().Output14ElementsOnlyFor\Rows\():
	OutputBlock \Mode\(),14,\Rows\()
        b       .L\Mode\().ExitKernel


.L\Mode\().Output10ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),10,\Rows\()
        b       .L\Mode\().ExitKernel


.L\Mode\().Output6ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),6,\Rows\()
        b       .L\Mode\().ExitKernel


.L\Mode\().Output2ElementsOnlyFor\Rows\():
        OutputBlock \Mode\(),2,\Rows\()
        b       .L\Mode\().ExitKernel

        .endm


/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_UMMLA>.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_UMMLA>.

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

Return Value:

    Returns the number of rows handled.

--*/

       .macro  QgemmU8X8KernelUmmlaFunction Mode

        FUNCTION_ENTRY MlasGemmU8X8KernelUmmla\Mode\()

        ldr     x10,[sp, #0]
        ldr     x11,[sp,#8]

        stp     x19, x20, [sp, #.LMlasQgemmKernel_SavedRegisters_Neg]!
        stp     x21, x22, [sp, #.LMlasQgemmKernel_backup_x21_x22]
        stp     x23, x24, [sp, #.LMlasQgemmKernel_backup_x23_x24]
        stp     x25, x26, [sp, #.LMlasQgemmKernel_backup_x25_x26]
        stp     x27, x28, [sp, #.LMlasQgemmKernel_backup_x27_x28]
        stp     d8, d9, [sp, #.LMlasQgemmKernel_backup_d8_d9]
        stp     d10, d11, [sp, #.LMlasQgemmKernel_backup_d10_d11]
        stp     d12, d13, [sp, #.LMlasQgemmKernel_backup_d12_d13]
        stp     d14, d15, [sp, #.LMlasQgemmKernel_backup_d14_d15]

        add     x13,x2,x6,lsl #2            // compute matrix C plus 1 row
        add     x14,x13,x6,lsl #2           // compute matrix C plus 2 rows
        add     x15,x14,x6,lsl #2           // compute matrix C plus 3 rows
        add     x16,x15,x6,lsl #2           // compute matrix C plus 4 rows
        add     x17,x16,x6,lsl #2           // compute matrix C plus 5 rows
        add     x18,x17,x6,lsl #2           // compute matrix C plus 6 rows
        add     x19,x18,x6,lsl #2           // compute matrix C plus 7 rows

        mov     x8,x0                       // save matrix A

//
// Process 8 rows of the matrices.
//
        ld1     {v12.4s},[x7],#16            // load row sum 1 ~ 4
        cmp     x4,#8
        blt     .L\Mode\().ProcessCountMLessThan8
        ld1     {v13.4s},[x7],#16            // load row sum 5 ~ 8
        ProcessRows \Mode\(),8

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     x0,x4

        ldp     d14, d15, [sp, #.LMlasQgemmKernel_backup_d14_d15]
        ldp     d12, d13, [sp, #.LMlasQgemmKernel_backup_d12_d13]
        ldp     d10, d11, [sp, #.LMlasQgemmKernel_backup_d10_d11]
        ldp     d8, d9, [sp, #.LMlasQgemmKernel_backup_d8_d9]
        ldp     x27, x28, [sp, #.LMlasQgemmKernel_backup_x27_x28]
        ldp     x25, x26, [sp, #.LMlasQgemmKernel_backup_x25_x26]
        ldp     x23, x24, [sp, #.LMlasQgemmKernel_backup_x23_x24]
        ldp     x21, x22, [sp, #.LMlasQgemmKernel_backup_x21_x22]
        ldp     x19, x20, [sp], #.LMlasQgemmKernel_SavedRegisters

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

        QgemmU8X8KernelUmmlaFunction Zero
        QgemmU8X8KernelUmmlaFunction Add

        .end
