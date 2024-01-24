/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelLsx.S

Abstract:

    This module implements the kernels for the single precision convolution
    operation.

    This implementation uses Lsx instructions.

--*/

#include "asmmacro.h"
#include "SconvKernelLsxCommon.h"

/*++

Macro Description:

    This macro generates code to clear the block accumulators.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    vr0-vr7 - Supplies the block accumulators.

--*/

        .macro ClearBlock FilterCount, OutputCount

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vxor.v $vr0,$vr0,$vr0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vxor.v $vr1,$vr1,$vr1"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vxor.v $vr2,$vr2,$vr2"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vxor.v $vr3,$vr3,$vr3"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vxor.v $vr4,$vr4,$vr4"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vxor.v $vr5,$vr5,$vr5"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vxor.v $vr6,$vr6,$vr6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vxor.v $vr7,$vr7,$vr7"

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for FilterCount by OutputCount block
    of the output buffer.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

    VectorOffset - Supplies the byte offset from the filter buffer to fetch
        elements.

    BroadcastOffset - Supplies the byte offset from the input buffer to fetch
        elements.

Implicit Arguments:

    a3 - Supplies the address of the input buffer.

    a2 - Supplies the address of the filter buffer.

    a1 - Supplies the FilterStride parameter (see function description).

    t6 - Supplies the address of the filter buffer plus 2 * FilterStride.

    a5 - Supplies the StrideWidth parameter (see function description).

    vr0-vr7 - Supplies the block accumulators.

--*/
        .macro ComputeBlock KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

.ifeqs "\KernelType\()","Depthwise"
        vld     $vr8, $a2, 0
        vld     $vr9, $a2, 16
        vld     $vr10, $a3, 0
        vld     $vr11, $a3, 16
        vfmadd.s $vr0, $vr8, $vr10, $vr0
        vfmadd.s $vr1, $vr9, $vr11, $vr1
.else
        EmitIfCountGE \OutputCount\(), 1, "ld.w $s0, $a3, \BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 1, "vreplgr2vr.w $vr12, $s0"
        EmitIfCountGE \FilterCount\(), 1, "vld  $vr8, $a2, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 1, "vld  $vr9, $a2, \VectorOffset\()+16"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfmadd.s $vr0, $vr8, $vr12, $vr0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfmadd.s $vr1, $vr9, $vr12, $vr1"
        EmitIfCountGE \FilterCount\(), 2, "addi.d   $s0, $a1, +\VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 2, "vldx  $vr8, $a2, $s0"
        EmitIfCountGE \FilterCount\(), 2, "addi.d   $s0, $a1, +\VectorOffset\()+16"
        EmitIfCountGE \FilterCount\(), 2, "vldx  $vr9, $a2, $s0"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfmadd.s $vr2, $vr8, $vr12, $vr2"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfmadd.s $vr3, $vr9, $vr12, $vr3"
        EmitIfCountGE \FilterCount\(), 3, "vld  $vr8, $t7, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 3, "vld  $vr9, $t7, \VectorOffset\()+16"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfmadd.s $vr4, $vr8, $vr12, $vr4"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfmadd.s $vr5, $vr9, $vr12, $vr5"
        EmitIfCountGE \FilterCount\(), 4, "addi.d   $s0, $a1, \VectorOffset\()"
        EmitIfCountGE \FilterCount\(), 4, "vldx  $vr8, $t7, $s0"
        EmitIfCountGE \FilterCount\(), 4, "addi.d   $s0, $a1, \VectorOffset\()+16"
        EmitIfCountGE \FilterCount\(), 4, "vldx  $vr9, $t7, $s0"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfmadd.s $vr6, $vr8, $vr12, $vr6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfmadd.s $vr7, $vr9, $vr12, $vr7"
.endif
        .endm
/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    s8 - Supplies the DilationWidth parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    s3 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessFilterCountN KernelFrame, KernelType, FilterCount
        ld.d    $s0, $sp, OutputCountLeftPad_arg   //OutputCountLeftPad
        ld.d    $s1, $sp, OutputCount_arg   //OutputCount
        add.d   $s0, $s0, $s1
        ld.d    $s1, $sp, OutputCountRightPad_arg   //OutputCountRightPad
        add.d   $t0, $s0, $s1
.L\KernelType\().\FilterCount\().ProcessNextOutputCount:
        ProcessOutputCountN Sse, \KernelFrame\(), \KernelType\(), 8, \FilterCount\(), 1
        add.d   $a0, $a0, $a5
        addi.d  $t0, $t0, -1
        bnez    $t0, .L\KernelType\().\FilterCount\().ProcessNextOutputCount
        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows for a pointwise convolution.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    a0 - Supplies the address of the input buffer.

    a1 - Supplies the FilterStride parameter (see function description).

    s8 - Supplies the InputStride parameter (see function description).

    a4 - Supplies the address of the output buffer.

    a5 - Supplies the StrideWidth parameter (see function description).

    t7 - Supplies the OutputCount parameter (see function description).

    s5 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseFilterCountN FilterCount
.LPointwise.\FilterCount\().ProcessNextOutputCount:
        ProcessPointwiseOutputCountN Sse, 8, \FilterCount\(), 1
        add.d   $a0, $a0, $a5
        addi.d  $t0, $t0, -1
        bnez    $t0, .LPointwise.\FilterCount\().ProcessNextOutputCount
        .endm

//
// Generate the convolution kernels.
//

        SconvKernelFunction Nchw, 8, LSX
        SconvKernelFunction Nchwc, 8, LSX, BiasFilter
        SconvKernelDepthwiseFunction 8, LSX
        SconvKernelPointwiseFunction LSX, BiasFilter

/*++

Macro Description:

    This macro generates code to process an output block after the inner
    convolution kernel has executed and then stores the output block to the
    output buffer.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.
--*/

        .macro PostProcessBlock FilterCount, OutputCount

        .globl  MlasConvPostProcessFloatSseFilter\FilterCount\()Output\OutputCount\()
#if !defined(__APPLE__)
        .hidden MlasConvPostProcessFloatSseFilter\FilterCount\()Output\OutputCount\()
#endif
MlasConvPostProcessFloatSseFilter\FilterCount\()Output\OutputCount\():

.if \FilterCount\() > 2
        li.d    $s0, 2
        mul.d   $s0, $s0, $t6
        add.d   $t7, $a4, $s0
.endif
        andi    $s0, $a2, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        andi    $s0, $s0, 0xff
        beqz    $s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vld $vr8, $a4, 0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vld $vr9, $a4, 16"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vldx $vr10, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "addi.d  $s0, $t6, 16"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vldx $vr11, $a4, $s0"

        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vld $vr12, $t7, 0"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vld $vr13, $t7, 16"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vldx $vr14, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "addi.d  $s0, $t6, 16"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vldx    $vr15, $t7, $s0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfadd.s $vr0, $vr0, $vr8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfadd.s $vr1, $vr1, $vr9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfadd.s $vr2, $vr2, $vr10"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfadd.s $vr3, $vr3, $vr11"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfadd.s $vr4, $vr4, $vr12"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfadd.s $vr5, $vr5, $vr13"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfadd.s $vr6, $vr6, $vr14"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfadd.s $vr7, $vr7, $vr15"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput:
//
// Test if the bias buffer should be accumulated with the output block.
//

        andi    $s0, $a2, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        andi    $s0, $s0, 0xff
        beqz    $s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vld $vr8, $a3, 0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vld $vr9, $a3, 16"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vld $vr10, $a3, 32"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vld $vr11, $a3, 48"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vld $vr12, $a3, 64"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vld $vr13, $a3, 80"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vld $vr14, $a3, 96"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vld $vr15, $a3, 112"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfadd.s $vr0, $vr0, $vr8"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfadd.s $vr1, $vr1, $vr9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfadd.s $vr2, $vr2, $vr10"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfadd.s $vr3, $vr3, $vr11"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfadd.s $vr4, $vr4, $vr12"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfadd.s $vr5, $vr5, $vr13"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfadd.s $vr6, $vr6, $vr14"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfadd.s $vr7, $vr7, $vr15"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

        andi        $s0, $a2, MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        andi        $s0, $s0, 0xff
        beqz        $s0, .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation
        vxor.v   $vr15,$vr15, $vr15
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfmax.s $vr0, $vr0, $vr15"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vfmax.s $vr1, $vr1, $vr15"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfmax.s $vr2, $vr2, $vr15"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vfmax.s $vr3, $vr3, $vr15"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfmax.s $vr4, $vr4, $vr15"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vfmax.s $vr5, $vr5, $vr15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfmax.s $vr6, $vr6, $vr15"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vfmax.s $vr7, $vr7, $vr15"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation:

//
// Store the output block in the output buffer.
//

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vst $vr0, $a4,0"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "vst $vr1, $a4, 16"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vstx $vr2, $a4, $t6"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "addi.d $s0, $t6, 16"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "vstx $vr3, $a4, $s0"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vst $vr4, $t7, 0"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "vst $vr5, $t7, 16"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vstx $vr6, $t7, $t6"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "addi.d $s0, $t6, 16"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "vstx $vr7, $t7, $s0"
        add_immed  $a4, \OutputCount\()*8*4    # advance output by N nchw8c blocks
        jr $ra

        .endm

        .irp    FilterCount, 1, 2, 3, 4
        .irp    OutputCount, 1
            PostProcessBlock \FilterCount\(), \OutputCount\()
        .endr
        .endr

        .end
