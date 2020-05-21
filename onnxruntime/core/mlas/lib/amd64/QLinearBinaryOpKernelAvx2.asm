;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QLinearBinaryOpKernelAvx2.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized linear add for element
;   type int8_t and uint8_t.
;
;   This implementation uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasPackBytesMM256VpshufbControl:NEAR
        EXTERN  MlasPackBytesMM256VpermpsControl:NEAR

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
        ScaleB QWORD ?
        ZeroPointB QWORD ?
        ScaleC QWORD ?
        ZeroPointC QWORD ?
        OutputC QWORD ?
        LengthA QWORD ?
        LengthB QWORD ?

QLinearBinaryOpFrame ENDS

;
; Some constants for U8/S8
;

MaxValue_S8             EQU     127
MinValue_S8             EQU     -128
MaxValue_U8             EQU     255
MinValue_U8             EQU     0

;
; Some instruction alias related with U8/S8
;

ExtendBits_S8           EQU     movsx
ExtendBits_U8           EQU     movzx
UnpackBytesDWords_S8    EQU     vpmovsxbd
UnpackBytesDWords_U8    EQU     vpmovzxbd

;
; Macro Description:
;
;   This macro generates code for function QLinearOpName() on the specified
;   signed/unsigned int8 DataType.
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
;    This routine implements the kernels for the Quantize Linear OpName for
;    element type DataType, vector on vector.
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
;    LengthA - Supplies the number of elements of InputA.
;
;    LengthB - Supplies the number of elements of InputB.
;              LengthB == LengthA or LengthA == 1 or LengthB == 1.
;
; Return Value:
;
;    None.
;

        NESTED_ENTRY MlasQLinear&OpName&&DataType&KernelAvx2, _TEXT

        alloc_stack (QLinearBinaryOpFrame.ReturnAddress)

        save_xmm128 xmm6,QLinearBinaryOpFrame.SavedXmm6
        save_xmm128 xmm7,QLinearBinaryOpFrame.SavedXmm7
        save_xmm128 xmm8,QLinearBinaryOpFrame.SavedXmm8
        save_xmm128 xmm9,QLinearBinaryOpFrame.SavedXmm9
        save_xmm128 xmm10,QLinearBinaryOpFrame.SavedXmm10
        save_xmm128 xmm11,QLinearBinaryOpFrame.SavedXmm11

        END_PROLOGUE

        vbroadcastss ymm0,xmm1                  ; Vector of ScaleA
        vbroadcastss ymm1,DWORD PTR QLinearBinaryOpFrame.ScaleB[rsp] ; Vector of ScaleB
        vbroadcastss ymm2,DWORD PTR QLinearBinaryOpFrame.ScaleC[rsp] ; Vector of ScaleC
        vmovd   xmm3,r8d
        vmovd   xmm4,DWORD PTR QLinearBinaryOpFrame.ZeroPointB[rsp]
        vmovd   xmm5,DWORD PTR QLinearBinaryOpFrame.ZeroPointC[rsp]
        vpbroadcastd ymm3,xmm3                  ; Vector of ZeroPointA
        vpbroadcastd ymm4,xmm4                  ; Vector of ZeroPointB
        vpbroadcastd ymm5,xmm5                  ; Vector of ZeroPointC

        vmovaps ymm10,YMMWORD PTR [MlasPackBytesMM256VpshufbControl]
        vmovaps ymm11,YMMWORD PTR [MlasPackBytesMM256VpermpsControl]

        mov     eax,MaxValue_&DataType&
        vmovd   xmm6,eax
        vpbroadcastd ymm6,xmm6                  ; Max U8/S8 Value Vector

IFE MinValue_&DataType&
        vxorps  ymm7,ymm7,ymm7                  ; Min U8 Value Vector
ELSE
        mov     eax,MinValue_&DataType&
        vmovd   xmm7,eax
        vpbroadcastd ymm7,xmm7                  ; Min S8 Value Vector
ENDIF

        vpsubd  ymm6,ymm6,ymm5
        vpsubd  ymm7,ymm7,ymm5
        vcvtdq2ps ymm6,ymm6                     ; Float Max Value Vector
        vcvtdq2ps ymm7,ymm7                     ; Float Min Value Vector

        mov     r8,QLinearBinaryOpFrame.OutputC[rsp]
        mov     rdx,QLinearBinaryOpFrame.LengthA[rsp]
        dec     rdx
        jz      QLinear&OpName&&DataType&Avx2Process8EntranceScalarOnVector
        mov     rdx,QLinearBinaryOpFrame.LengthB[rsp]
        dec     rdx
        jz      QLinear&OpName&&DataType&Avx2Process8EntranceVectorOnScalar
        inc     rdx
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopVectorOnVector

QLinear&OpName&&DataType&Avx2Process8EntranceScalarOnVector:
        mov     rdx,QLinearBinaryOpFrame.LengthB[rsp]
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        ExtendBits_&DataType& eax,BYTE PTR[rcx] ; Scalar ValueA
        vmovd   xmm8,eax
        vpbroadcastd ymm8,xmm8                  ; IntVectorA
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA

QLinear&OpName&&DataType&Avx2Process8LoopScalarOnVector:
        UnpackBytesDWords_&DataType& ymm9,QWORD PTR [r9]  ; IntegerVectorB
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

        OpInstruction  ymm9,ymm8,ymm9           ; OpName two float values

        vdivps  ymm9,ymm9,ymm2                  ; Quantize 8 values, / ScaleC
        add     r9,8                            ; out-of-order instruction(s)
        sub     rdx,8                           ; out-of-order instruction(s), set flag for jb below
        vminps  ymm9,ymm9,ymm6
        vmaxps  ymm9,ymm9,ymm7
        vcvtps2dq ymm9,ymm9                     ; nearbyintf()
        vpaddd  ymm9,ymm9,ymm5                  ; + ZeroPointC
        vpshufb ymm9,ymm9,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm9,ymm11,ymm9

        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8ScalarOnVector

        vmovq   QWORD PTR [r8],xmm9
        add     r8,8
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopScalarOnVector

QLinear&OpName&&DataType&Avx2StoreLessThan8ScalarOnVector:
        add     rdx,8
        vpextrq rax,xmm9,0
        jmp     QLinear&OpName&&DataType&Avx2StoreLoopGeneral

QLinear&OpName&&DataType&Avx2Process8EntranceVectorOnScalar:
        mov     rdx,QLinearBinaryOpFrame.LengthA[rsp]
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        ExtendBits_&DataType& eax,BYTE PTR [r9] ; Scalar ValueB
        vmovd   xmm9,eax
        vpbroadcastd ymm9,xmm9                  ; IntVectorB
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

QLinear&OpName&&DataType&Avx2Process8LoopVectorOnScalar:
        UnpackBytesDWords_&DataType& ymm8,QWORD PTR [rcx] ; IntegerVectorA
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA

        OpInstruction  ymm8,ymm8,ymm9           ; OpName two float values

        vdivps  ymm8,ymm8,ymm2                  ; Quantize 8 values, / ScaleC
        add     rcx,8                           ; out-of-order instruction(s)
        sub     rdx,8                           ; out-of-order instruction(s)
        vminps  ymm8,ymm8,ymm6
        vmaxps  ymm8,ymm8,ymm7
        vcvtps2dq ymm8,ymm8                     ; nearbyintf()()
        vpaddd  ymm8,ymm8,ymm5                  ; + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnScalar

        vmovq   QWORD PTR [r8],xmm8
        add     r8,8
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopVectorOnScalar

QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnScalar:
        add     rdx,8
        vpextrq rax,xmm8,0
        jmp     QLinear&OpName&&DataType&Avx2StoreLoopGeneral

QLinear&OpName&&DataType&Avx2Process8LoopVectorOnVector:
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit

        UnpackBytesDWords_&DataType& ymm8,QWORD PTR [rcx] ; IntegerVectorA
        UnpackBytesDWords_&DataType& ymm9,QWORD PTR [r9]  ; IntegerVectorB
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

        OpInstruction  ymm8,ymm8,ymm9           ; OpName two float values

        vdivps  ymm8,ymm8,ymm2                  ; Quantize 8 values, / ScaleC
        add     rcx,8                           ; out-of-order instruction(s)
        add     r9,8                            ; out-of-order instruction(s)
        sub     rdx,8                           ; out-of-order instruction(s), set flag for jb below
        vminps  ymm8,ymm8,ymm6
        vmaxps  ymm8,ymm8,ymm7
        vcvtps2dq ymm8,ymm8                     ; nearbyintf()
        vpaddd  ymm8,ymm8,ymm5                  ; + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnVector

        vmovq   QWORD PTR [r8],xmm8
        add     r8,8
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopVectorOnVector

QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnVector:
        add     rdx,8
        vpextrq rax,xmm8,0

QLinear&OpName&&DataType&Avx2StoreLoopGeneral:
        mov     BYTE PTR [r8],al
        shr     rax,8
        inc     r8
        dec     rdx
        jnz     QLinear&OpName&&DataType&Avx2StoreLoopGeneral

QLinear&OpName&&DataType&Avx2Exit:
        vzeroupper
        movaps  xmm6,QLinearBinaryOpFrame.SavedXmm6[rsp]
        movaps  xmm7,QLinearBinaryOpFrame.SavedXmm7[rsp]
        movaps  xmm8,QLinearBinaryOpFrame.SavedXmm8[rsp]
        movaps  xmm9,QLinearBinaryOpFrame.SavedXmm9[rsp]
        movaps  xmm10,QLinearBinaryOpFrame.SavedXmm10[rsp]
        movaps  xmm11,QLinearBinaryOpFrame.SavedXmm11[rsp]
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
