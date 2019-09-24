;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SpoolKernelAvx.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision pooling
;   operation.
;
;   This implementation uses AVX instructions.
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
        vbroadcastss ymm5,DWORD PTR SpoolKernelFrame.PreviousP1Home[rsp]
ELSE
        vxorps  xmm5,xmm5,xmm5              ; initialize default divisor vector
IFIDNI <PoolingType>, <AverageExcludePad>
        mov     rax,SpoolKernelFrame.KernelHeight[rsp]
        imul    rax,SpoolKernelFrame.KernelWidth[rsp]
        vcvtsi2ss xmm5,xmm5,rax
ELSE
        vcvtsi2ss xmm5,xmm5,SpoolKernelFrame.ActualKernelSize[rsp]
ENDIF
        vshufps xmm5,xmm5,xmm5,0
        vinsertf128 ymm5,ymm5,xmm5,1        ; AVX lacks "vbroadcastss ymm5,xmm5"
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
;   ymm0-ymm2 - Supplies the pooling intermediates.
;
;   ymm5 - Supplies a vector containing the minimum float value broadcasted,
;       if PoolingType==Maximum.
;

ClearBlock MACRO PoolingType, OutputCount

IFIDNI <PoolingType>, <Maximum>
        EmitIfCountGE OutputCount, 1, <vmovaps ymm0,ymm5>
        EmitIfCountGE OutputCount, 2, <vmovaps ymm1,ymm5>
        EmitIfCountGE OutputCount, 3, <vmovaps ymm2,ymm5>
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
;   ymm0-ymm2 - Supplies the pooling intermediates.
;

ComputeBlock MACRO PoolingType, OutputCount

IFIDNI <PoolingType>, <Maximum>
        EmitIfCountGE OutputCount, 1, <vmaxps ymm0,ymm0,YMMWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 2, <vmaxps ymm1,ymm1,YMMWORD PTR [rcx+r8]>
        EmitIfCountGE OutputCount, 3, <vmaxps ymm2,ymm2,YMMWORD PTR [rcx+r8*2]>
ELSE
        EmitIfCountGE OutputCount, 1, <vaddps ymm0,ymm0,YMMWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 2, <vaddps ymm1,ymm1,YMMWORD PTR [rcx+r8]>
        EmitIfCountGE OutputCount, 3, <vaddps ymm2,ymm2,YMMWORD PTR [rcx+r8*2]>
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
;   ymm0-ymm2 - Supplies the pooling intermediates.
;
;   ymm5 - Supplies the kernel size computed by InitializeKernel, if
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
        vshufps xmm4,xmm4,xmm4,0
        vinsertf128 ymm4,ymm4,xmm4,1        ; AVX lacks "vbroadcastss ymm4,xmm4"
        vdivps  ymm0,ymm0,ymm4
ELSE
        EmitIfCountGE OutputCount, 1, <vdivps ymm0,ymm0,ymm5>
        EmitIfCountGE OutputCount, 2, <vdivps ymm1,ymm1,ymm5>
        EmitIfCountGE OutputCount, 3, <vdivps ymm2,ymm2,ymm5>
ENDIF
ENDIF

;
; If PoolingType=AverageIncludePad, divide the sum by the actual kernel size.
;

IFIDNI <PoolingType>, <AverageIncludePad>
        EmitIfCountGE OutputCount, 1, <vdivps ymm0,ymm0,ymm5>
        EmitIfCountGE OutputCount, 2, <vdivps ymm1,ymm1,ymm5>
        EmitIfCountGE OutputCount, 3, <vdivps ymm2,ymm2,ymm5>
ENDIF

;
; Store the output block in the output buffer.
;

        EmitIfCountGE OutputCount, 1, <vmovups YMMWORD PTR [rdx],ymm0>
        EmitIfCountGE OutputCount, 2, <vmovups YMMWORD PTR [rdx+8*4],ymm1>
        EmitIfCountGE OutputCount, 3, <vmovups YMMWORD PTR [rdx+16*4],ymm2>
        add     rdx,OutputCount*8*4         ; advance output by N nchw8c blocks

        ENDM

;
; Generate the pooling kernels.
;

SpoolKernelFunction Maximum, Avx
SpoolKernelFunction AverageExcludePad, Avx
SpoolKernelFunction AverageIncludePad, Avx

        END
