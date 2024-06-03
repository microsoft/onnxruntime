/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelLsx.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses Lsx instructions.

--*/

#include "asmmacro.h"
#include "FgemmKernelLsxCommon.h"

FGEMM_TYPED_INSTRUCTION(vfadd, vfadd.d)
/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    a1 (rsi) - Supplies the address into the matrix B data.

    vr0-vr1 - Supplies up to two elements loaded from matrix A and matrix A
        plus one row.

    vr8-vr15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy8 RowCount

        vld     $vr4, $a1, 0
        vld     $vr5, $a1, 16
.if \RowCount\() == 2
        vmove   $vr6, $vr4
        vmove   $vr7, $vr5
.endif
        vfmadd.d    $vr8, $vr4, $vr0, $vr8
        vfmadd.d    $vr9, $vr5, $vr0, $vr9
.if \RowCount\() == 2
        vfmadd.d    $vr12, $vr6, $vr1, $vr12
        vfmadd.d    $vr13, $vr7, $vr1, $vr13
.endif
        vld     $vr4, $a1, 32
        vld     $vr5, $a1, 48
.if \RowCount\() == 2
        vmove   $vr6, $vr4
        vmove   $vr7, $vr5
.endif
        vfmadd.d    $vr10, $vr4, $vr0, $vr10
        vfmadd.d    $vr11, $vr5, $vr0, $vr11
.if \RowCount\() == 2
        vfmadd.d    $vr14, $vr6, $vr1, $vr14
        vfmadd.d    $vr15, $vr7, $vr1, $vr15
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernel label.

Implicit Arguments:

    a0 - Supplies the address of matrix A.

    a1 - Supplies the address of matrix B.

    t8 - Supplies the address of matrix A.

    a5 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    a2 - Supplies the address of matrix C.

    a3 - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    t7 - Supplies the length in bytes of a row from matrix A.

    t5 - Supplies the length in bytes of a row from matrix C.

    s3 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount, Fallthrough
.LProcessNextColumnLoop8xN\@:
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr8,$vr8,$vr8"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr9,$vr9,$vr9"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr10,$vr10,$vr10"
        EmitIfCountGE \RowCount\(), 1, "vxor.v $vr11,$vr11,$vr11"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr12,$vr12,$vr12"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr13,$vr13,$vr13"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr14,$vr14,$vr14"
        EmitIfCountGE \RowCount\(), 2, "vxor.v $vr15,$vr15,$vr15"
        move     $t7,$a3                     # reload CountK
.LCompute8xNBlockBy1Loop\@:
        EmitIfCountGE \RowCount\(), 1, "ld.d    $s0, $a0, 0"
        EmitIfCountGE \RowCount\(), 1, "vreplgr2vr.d    $vr0, $s0"
        EmitIfCountGE \RowCount\(), 2, "ldx.d    $s0, $a0, $t0"
        EmitIfCountGE \RowCount\(), 2, "vreplgr2vr.d    $vr1, $s0"
        ComputeBlockSseBy8 \RowCount\()
        addi.d     $a1, $a1, 8*8                     # advance matrix B by 8 columns
        addi.d     $a0, $a0, 8                       # advance matrix A by 1 column
        addi.d     $t7, $t7, -1
        bnez       $t7, .LCompute8xNBlockBy1Loop\@

.LOutput8xNBlock\@:
        movfr2gr.d      $s0,  $f24
        vreplgr2vr.d    $vr2, $s0
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "vfmul.d $vr8, $vr8, $vr2"
        EmitIfCountGE \RowCount\(), 1, "vfmul.d $vr9, $vr9, $vr2"
        EmitIfCountGE \RowCount\(), 1, "vfmul.d $vr10,$vr10, $vr2"
        EmitIfCountGE \RowCount\(), 1, "vfmul.d $vr11,$vr11, $vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.d $vr12,$vr12, $vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.d $vr13,$vr13, $vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.d $vr14,$vr14, $vr2"
        EmitIfCountGE \RowCount\(), 2, "vfmul.d $vr15,$vr15, $vr2"
        li.d    $s0, 8
        blt     $a5, $s0, .LOutputPartial8xNBlock\@
        sub.d   $a5, $a5, $s0
        AccumulateAndStoreBlock \RowCount\(), 4
        addi.d  $a2, $a2, 8*8       # advance matrix C by 8 columns
        move    $a0, $t1            # reload matrix A
        bnez    $a5, .LProcessNextColumnLoop8xN\@
        b       .LExitKernel

//
// Output a partial 8xN block to the matrix.
//

.LOutputPartial8xNBlock\@:
        li.d    $s0, 2
        blt     $a5, $s0, .LOutputPartial1xNBlock\@
        li.d    $s0, 4
        blt     $a5, $s0, .LOutputPartialLessThan4xNBlock\@
        li.d    $s0, 6
        blt     $a5, $s0, .LOutputPartialLessThan6xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 3
        andi    $s0, $a5, 1                  # check if remaining count is small
        beqz    $s0, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8,$vr11"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12,$vr15"
        addi.d     $a2, $a2, 6*8                     # advance matrix C by 6 columns
        b     .LOutputPartial1xNBlock\@

.LOutputPartialLessThan6xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 2
        andi    $s0, $a5,1                       # check if remaining count is small
        beqz    $s0, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8,$vr10"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12,$vr14"
        addi.d     $a2, $a2, 4*8                     # advance matrix C by 4 columns
        b     .LOutputPartial1xNBlock\@

.LOutputPartialLessThan4xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 1
        andi    $s0, $a5,1                       # check if remaining count is small
        beqz    $s0, .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "vmove $vr8,$vr9"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "vmove $vr12,$vr13"
        addi.d     $a2, $a2, 2*8                     # advance matrix C by 2 columns

.LOutputPartial1xNBlock\@:
        bnez    $t5, .LSkipAccumulateOutput1xN\@     # ZeroMode?

        EmitIfCountGE \RowCount\(), 1, "fld.d    $f15, $a2, 0"
        EmitIfCountGE \RowCount\(), 1, "fadd.d  $f15, $f15, $f8"
        EmitIfCountGE \RowCount\(), 2, "fldx.d   $f16, $a2, $t6"
        EmitIfCountGE \RowCount\(), 2, "fadd.d  $f16, $f16, $f12"

.LSkipAccumulateOutput1xN\@:
        EmitIfCountGE \RowCount\(), 1, "fst.d    $f15, $a2, 0"
        EmitIfCountGE \RowCount\(), 2, "fstx.d    $f16, $a2, $t6"
.ifb \Fallthrough\()
        b     .LExitKernel
.endif

        .endm

//
// Generate the GEMM kernel.
//

FgemmKernelLsxFunction MlasGemmDoubleKernelLSX

        .end
