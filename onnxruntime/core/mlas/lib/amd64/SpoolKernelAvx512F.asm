;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SpoolKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision pooling
;   operation.
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SpoolKernelAvxCommon.inc
        .list

;
; Macro Description:
;
;   This macro generates code to initialize registers used across the kernel.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;

InitializeKernel MACRO PoolingType

IFIDNI <PoolingType>, <Maximum>
        mov     DWORD PTR SpoolKernelFrame.PreviousP1Home[rsp],0FF7FFFFFh
        vbroadcastss zmm5,DWORD PTR SpoolKernelFrame.PreviousP1Home[rsp]
ELSE
        vxorps  xmm5,xmm5,xmm5              ; initialize default divisor vector
IFIDNI <PoolingType>, <AverageExcludePad>
        mov     rax,SpoolKernelFrame.KernelHeight[rsp]
        imul    rax,SpoolKernelFrame.KernelWidth[rsp]
        vcvtsi2ss xmm5,xmm5,rax
ELSE
        vcvtsi2ss xmm5,xmm5,SpoolKernelFrame.ActualKernelSize[rsp]
ENDIF
        vbroadcastss zmm5,xmm5
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to clear the pooling intermediates.
;
;   For PoolingType==Maximum, the pooling intermediates are set to the minimum
;   float value. Otherwise, the pooling intermediates are cleared to zero.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rsi - Supplies the number of blocks accessed by ComputeBlock, if
;       PoolingType=AverageExcludePad and OutputCount=1.
;
;   zmm0-zmm2 - Supplies the pooling intermediates.
;
;   zmm5 - Supplies a vector containing the minimum float value broadcasted,
;       if PoolingType==Maximum.
;

ClearBlock MACRO PoolingType, OutputCount

IFIDNI <PoolingType>, <Maximum>
        EmitIfCountGE OutputCount, 1, <vmovaps zmm0,zmm5>
        EmitIfCountGE OutputCount, 2, <vmovaps zmm1,zmm5>
        EmitIfCountGE OutputCount, 3, <vmovaps zmm2,zmm5>
ELSE
        EmitIfCountGE OutputCount, 1, <vxorps xmm0,xmm0,xmm0>
        EmitIfCountGE OutputCount, 2, <vxorps xmm1,xmm1,xmm1>
        EmitIfCountGE OutputCount, 3, <vxorps xmm2,xmm2,xmm2>
ENDIF

IFIDNI <PoolingType>, <AverageExcludePad>
IF OutputCount EQ 1
        xor     rsi,rsi                     ; reset valid block counter
ENDIF
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to sample the input buffer and update the pooling
;   intermediates as appropriate.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the input buffer.
;
;   rsi - Supplies the number of blocks accessed by ComputeBlock, if
;       PoolingType=AverageExcludePad and OutputCount=1.
;
;   r8 - Supplies the StrideWidth parameter (see function description).
;
;   zmm0-zmm2 - Supplies the pooling intermediates.
;

ComputeBlock MACRO PoolingType, OutputCount

IFIDNI <PoolingType>, <Maximum>
        EmitIfCountGE OutputCount, 1, <vmaxps zmm0,zmm0,ZMMWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 2, <vmaxps zmm1,zmm1,ZMMWORD PTR [rcx+r8]>
        EmitIfCountGE OutputCount, 3, <vmaxps zmm2,zmm2,ZMMWORD PTR [rcx+r8*2]>
ELSE
        EmitIfCountGE OutputCount, 1, <vaddps zmm0,zmm0,ZMMWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 2, <vaddps zmm1,zmm1,ZMMWORD PTR [rcx+r8]>
        EmitIfCountGE OutputCount, 3, <vaddps zmm2,zmm2,ZMMWORD PTR [rcx+r8*2]>
ENDIF

IFIDNI <PoolingType>, <AverageExcludePad>
IF OutputCount EQ 1
        inc     rsi                         ; increment valid block counter
ENDIF
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to process and store the pooling intermediates.
;
; Arguments:
;
;   PoolingType - Supplies the pooling type string.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   rdx - Supplies the address of the output buffer.
;
;   rsi - Supplies the number of blocks accessed by ComputeBlock, if
;       PoolingType=AverageExcludePad and OutputCount=1.
;
;   zmm0-zmm2 - Supplies the pooling intermediates.
;
;   zmm5 - Supplies the kernel size computed by InitializeKernel, if
;       PoolingType=AverageExcludePad, else the actual kernel size, if
;       PoolingType=AverageIncludePad.
;

PostProcessBlock MACRO PoolingType, OutputCount

;
; If PoolingType=AverageExcludePad, divide the sum by the number of non-padding
; blocks. OutputCount=1 generates code to count the number of blocks accessed by
; ComputeBlock. Other cases use the kernel size computed by InitializeKernel.
;

IFIDNI <PoolingType>, <AverageExcludePad>
IF OutputCount EQ 1
        vxorps  xmm4,xmm4,xmm4
        vcvtsi2ss xmm4,xmm4,rsi             ; convert valid block counter
        vbroadcastss zmm4,xmm4
        vdivps  zmm0,zmm0,zmm4
ELSE
        EmitIfCountGE OutputCount, 1, <vdivps zmm0,zmm0,zmm5>
        EmitIfCountGE OutputCount, 2, <vdivps zmm1,zmm1,zmm5>
        EmitIfCountGE OutputCount, 3, <vdivps zmm2,zmm2,zmm5>
ENDIF
ENDIF

;
; If PoolingType=AverageIncludePad, divide the sum by the actual kernel size.
;

IFIDNI <PoolingType>, <AverageIncludePad>
        EmitIfCountGE OutputCount, 1, <vdivps zmm0,zmm0,zmm5>
        EmitIfCountGE OutputCount, 2, <vdivps zmm1,zmm1,zmm5>
        EmitIfCountGE OutputCount, 3, <vdivps zmm2,zmm2,zmm5>
ENDIF

        EmitIfCountGE OutputCount, 1, <vmovups ZMMWORD PTR [rdx],zmm0>
        EmitIfCountGE OutputCount, 2, <vmovups ZMMWORD PTR [rdx+16*4],zmm1>
        EmitIfCountGE OutputCount, 3, <vmovups ZMMWORD PTR [rdx+32*4],zmm2>
        add     rdx,OutputCount*16*4        ; advance output by N nchw16c blocks

        ENDM

;
; Generate the pooling kernels.
;

SpoolKernelFunction Maximum, Avx512F
SpoolKernelFunction AverageExcludePad, Avx512F
SpoolKernelFunction AverageIncludePad, Avx512F

        END
