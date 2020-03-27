;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;    QLinearBinaryOpKernelAvx2.asm
;
; Abstract:
;
;    This module implements the kernels for the quantized linear add
;    for element type int8_t and uint8_t.
;
;    This implementation uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MLasPackBytesMM256VpshufbControl:NEAR
        EXTERN  MLasPackBytesMM256VpermpsControl:NEAR

;
; Stack frame layout for the QLinearBinaryOp kernels.
;

QLinearBinaryOpFrame STRUCT
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        Padding0 QWORD ?
        Padding1 QWORD ?
        Padding2 QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        ScaleB  QWORD ?
        ZeroPointB QWORD ?
        ScaleC QWORD ?
        ZeroPointC QWORD ?
        OutputC QWORD ?
        N QWORD ?
QLinearBinaryOpFrame ENDS

;
; Macro Description:
;
;   This macro generates code to extend signed/unsigned int8 to
;   signed/unsigned int64/32/16 respectively, according to DataType
;   and target.
;
; Arguments:
;
;   Target - target register, could be 16/32/64 bits one.
;
;   Source - Supplies address of 8 x 8bits integers.
;
;   DataType - S8 or U8
;

Extend8BitsInt MACRO Target, Source, DataType
IFIDN <DataType>, <S8>
        movsx Target,BYTE PTR Source
ELSE
        movzx Target,BYTE PTR Source
ENDIF
        ENDM


;
; Macro Description:
;
;   This macro generates code to unpack 8 x (s/u)int8 to 8 x int32,
;   according to signed/unsigned.
;
; Arguments:
;
;   Source - Supplies address of 8 x 8bits integers.
;
;   Target - target ymm register.
;
;   DataType - S8 or U8
;

UnpackBytesDWords MACRO Target, Source, DataType
IFIDN <DataType>, <S8> 
        vpmovsxbd Target,QWORD PTR Source
ELSE
        vpmovzxbd Target,QWORD PTR Source
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code to set Target 64bits register with the
;   max value of signed/unsigned int8 specified by DataType.
;
; Arguments:
;
;   Target - target 64bits register.
;
;   DataType - S8 or U8.
;

SetMax8BitsValue MACRO Target, DataType
IFIDN <DataType>, <S8>
        mov Target,QWORD PTR 127
ELSE
        mov Target,QWORD PTR 255
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code to set Target 64bits register with the
;   min value of signed/unsigned int8 specified by DataType.
;
; Arguments:
;
;   Target - target 64bits register.
;
;   DataType - S8 or U8.
;

SetMin8BitsValue MACRO Target, DataType
IFIDN <DataType>, <S8>
        mov Target,QWORD PTR -128
ELSE
        mov Target,QWORD PTR 0
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code for function QLinearOpName()
;   on the specified signed/unsigned int8 DataType.
;
; Arguments:
;
;   DataType - S8 or U8.
;
;   OpName - Name of the QLinearOp, like Add, Mul, etc.
;
;   OpInstruction - the assembly code prefix which op() two ymm vector of floats,
;                   like vaddps, vmulps, etc
;

QLinearBinaryOpAvx2 MACRO DataType, OpName, OpInstruction

;
; Routine Description:
;
;    This routine implements the kernels for the Quantize Linear OpName
;    for element type DataType, vector on vector.
;
; Arguments:
;
;    InputA (rcx) - Supplies the address of InputA.
;
;    ScaleA (xmm1) - Supplies A's Scale value in float.
;
;    ZeroPointA (r8) - Supplies A's zero point value.
;
;    InputB (r9) - Supplies the address of InputB.
;
;    ScaleB - Supplies B's Scale value in float.
;
;    ZeroPointB - Supplies B's zero point value.
;
;    ScaleC - Supplies C's Scale value in float.
;
;    ZeroPointC - Supplies C's zero point value.
;
;    OutputC - Supplies the address of OutputC.
;
;    N - Supplies the number of elements to calculate.
;
; Return Value:
;
;    None.
;

        NESTED_ENTRY MlasQLinear&OpName&&DataType&KernelAvx2, _TEXT

        alloc_stack (QLinearBinaryOpFrame.ReturnAddress)

        save_xmm128_avx xmm6,QLinearBinaryOpFrame.SavedXmm6
        save_xmm128_avx xmm7,QLinearBinaryOpFrame.SavedXmm7
        save_xmm128_avx xmm8,QLinearBinaryOpFrame.SavedXmm8
        save_xmm128_avx xmm9,QLinearBinaryOpFrame.SavedXmm9
        save_xmm128_avx xmm10,QLinearBinaryOpFrame.SavedXmm10
        save_xmm128_avx xmm11,QLinearBinaryOpFrame.SavedXmm11

        END_PROLOGUE

        vbroadcastss ymm0,xmm1                  ; Vector of ScaleA
        vbroadcastss ymm1,DWORD PTR QLinearBinaryOpFrame.ScaleB[rsp]
        vbroadcastss ymm2,DWORD PTR QLinearBinaryOpFrame.ScaleC[rsp]
        Extend8BitsInt r8,r8b,DataType          ; Zero Point A,B,C
        Extend8BitsInt rdx,BYTE PTR QLinearBinaryOpFrame.ZeroPointB[rsp],DataType
        Extend8BitsInt rax,BYTE PTR QLinearBinaryOpFrame.ZeroPointC[rsp],DataType
        movq    xmm3,r8
        movq    xmm4,rdx
        movq    xmm5,rax
        vbroadcastss ymm3,xmm3                  ; Vector of ZeroPointA
        vbroadcastss ymm4,xmm4                  ; Vector of ZeroPointB
        vbroadcastss ymm5,xmm5                  ; Vector of ZeroPointC

        lea     rdx,MLasPackBytesMM256VpshufbControl
        lea     r8,MLasPackBytesMM256VpermpsControl
        vmovaps ymm10,[rdx]
        vmovaps ymm11,[r8]

        SetMax8BitsValue rdx,DataType
        movq         xmm6,rdx
        vbroadcastss ymm6,xmm6
        vpsubd       ymm6,ymm6,ymm5

        SetMin8BitsValue r8,DataType
        movq         xmm7,r8
        vbroadcastss ymm7,xmm7
        vpsubd       ymm7,ymm7,ymm5

        mov     rdx,QLinearBinaryOpFrame.N[rsp]
        mov     rax,QWORD PTR 8
        mov     r8,QLinearBinaryOpFrame.OutputC[rsp]

QLinear&OpName&&DataType&Avx2Process8Loop:
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit

        UnpackBytesDWords ymm8,[rcx],DataType
        UnpackBytesDWords ymm9,[r9],DataType
        vpsubd  ymm9,ymm9,ymm4                  ; - Zero Points respectively
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8                     ; convert to float
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm8,ymm8,ymm0                  ; * Scales respectively 
        vmulps  ymm9,ymm9,ymm1

        OpInstruction  ymm8,ymm8,ymm9           ; OpName two float values

        vdivps  ymm8,ymm8,ymm2                  ; Quantize 8 values, / ScaleC
        add     rcx,rax                         ; two out-of-order instructions
        add     r9,rax
        vcvtps2dq ymm8,ymm8                     ; round()
        vpmaxsd ymm8,ymm8,ymm7
        vpminsd ymm8,ymm8,ymm6
        vpaddd  ymm8,ymm8,ymm5                  ; + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        sub     rdx,rax
        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8

        movsd   QWORD PTR [r8],xmm8
        add     r8,rax
        jmp     QLinear&OpName&&DataType&Avx2Process8Loop

QLinear&OpName&&DataType&Avx2StoreLessThan8:
        add     rdx,rax
        pextrq  rax,xmm8,0

QLinear&OpName&&DataType&Avx2StoreLoop:
        mov     BYTE PTR [r8],al
        shr     rax,8
        inc     r8
        dec     rdx
        jnz     QLinear&OpName&&DataType&Avx2StoreLoop

QLinear&OpName&&DataType&Avx2Exit:
        vzeroupper
        vmovaps xmm6,QLinearBinaryOpFrame.SavedXmm6[rsp]
        vmovaps xmm7,QLinearBinaryOpFrame.SavedXmm7[rsp]
        vmovaps xmm8,QLinearBinaryOpFrame.SavedXmm8[rsp]
        vmovaps xmm9,QLinearBinaryOpFrame.SavedXmm9[rsp]
        vmovaps xmm10,QLinearBinaryOpFrame.SavedXmm10[rsp]
        vmovaps xmm11,QLinearBinaryOpFrame.SavedXmm11[rsp]
        add     rsp,(QLinearBinaryOpFrame.ReturnAddress)

        BEGIN_EPILOGUE

        ret

        NESTED_END MlasQLinear&OpName&&DataType&KernelAvx2, _TEXT

        ENDM

;
; Generate the QLinearAdd Avx2 S8 kernel.
;

QLinearBinaryOpAvx2 S8,Add,vaddps


;
; Generate the QLinearAdd Avx2 U8 kernel.
;

QLinearBinaryOpAvx2 U8,Add,vaddps


        END