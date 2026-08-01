/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    elementwise_sve_asm.S

Abstract:

    Portable machine-code variant of the SVE elementwise_sve kernels, in the style
    of Arm's KleidiAI library: every instruction is emitted as a raw
    instruction word (GAS ".inst" / armasm64 "DCD" via KAI_ASM_INST), so the
    file assembles with toolchains that have no SVE support, and the bytes
    are identical on every platform.

    GENERATED FILE - DO NOT EDIT BY HAND.

    Generated from the SVE intrinsics translation unit(s) (sve/elementwise_sve.cpp, sve/elementwise_sve_fp16.cpp), which
    remain the reference implementation and regeneration source (script:
    sve/gen_sve_asm.py). The functions here export the same extern "C"
    symbols; the build links exactly one of the two implementations. The
    code is fully position-independent (the generator verifies there are no
    relocations, calls, or literal pools).

--*/

#include "kai_asm_macros.h"

    KAI_ASM_CODE(elementwise_sve)


    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveErfKernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveErfKernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveErfKernelImpl)
    KAI_ASM_INST(0x910003eb)  // mov	x11, sp
    KAI_ASM_INST(0x0423e3ec)  // cntb	x12, all, mul #4
    KAI_ASM_INST(0x9101018c)  // add	x12, x12, #0x40
    KAI_ASM_INST(0xf1403d9f)  // cmp	x12, #0xf, lsl #12
    KAI_ASM_INST(0x540000ab)  // b.lt	24 <.SVLPEND0>
    KAI_ASM_INST(0xd1403fff)  // sub	sp, sp, #0xf, lsl #12
    KAI_ASM_INST(0xf90003ff)  // str	xzr, [sp]
    KAI_ASM_INST(0xd1403d8c)  // sub	x12, x12, #0xf, lsl #12
    KAI_ASM_INST(0x17fffffb)  // b	c <.SVLPSPL0>
    KAI_ASM_INST(0xcb2c63ff)  // sub	sp, sp, x12
    KAI_ASM_INST(0x6d0027e8)  // stp	d8, d9, [sp]
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8544c069)  // ld1rw	{z9.s}, p0/z, [x3, #16]
    KAI_ASM_INST(0x8545c068)  // ld1rw	{z8.s}, p0/z, [x3, #20]
    KAI_ASM_INST(0x8546c07f)  // ld1rw	{z31.s}, p0/z, [x3, #24]
    KAI_ASM_INST(0x8547c07e)  // ld1rw	{z30.s}, p0/z, [x3, #28]
    KAI_ASM_INST(0x8549c07d)  // ld1rw	{z29.s}, p0/z, [x3, #36]
    KAI_ASM_INST(0x854ac07c)  // ld1rw	{z28.s}, p0/z, [x3, #40]
    KAI_ASM_INST(0x854bc07b)  // ld1rw	{z27.s}, p0/z, [x3, #44]
    KAI_ASM_INST(0x854cc07a)  // ld1rw	{z26.s}, p0/z, [x3, #48]
    KAI_ASM_INST(0x854dc079)  // ld1rw	{z25.s}, p0/z, [x3, #52]
    KAI_ASM_INST(0x854ec078)  // ld1rw	{z24.s}, p0/z, [x3, #56]
    KAI_ASM_INST(0x854fc077)  // ld1rw	{z23.s}, p0/z, [x3, #60]
    KAI_ASM_INST(0x8553c076)  // ld1rw	{z22.s}, p0/z, [x3, #76]
    KAI_ASM_INST(0x8554c075)  // ld1rw	{z21.s}, p0/z, [x3, #80]
    KAI_ASM_INST(0x8555c074)  // ld1rw	{z20.s}, p0/z, [x3, #84]
    KAI_ASM_INST(0x8556c073)  // ld1rw	{z19.s}, p0/z, [x3, #88]
    KAI_ASM_INST(0x855ec065)  // ld1rw	{z5.s}, p0/z, [x3, #120]
    KAI_ASM_INST(0x8557c072)  // ld1rw	{z18.s}, p0/z, [x3, #92]
    KAI_ASM_INST(0x8558c071)  // ld1rw	{z17.s}, p0/z, [x3, #96]
    KAI_ASM_INST(0x8559c070)  // ld1rw	{z16.s}, p0/z, [x3, #100]
    KAI_ASM_INST(0x6d012fea)  // stp	d10, d11, [sp, #16]
    KAI_ASM_INST(0x8542c06b)  // ld1rw	{z11.s}, p0/z, [x3, #8]
    KAI_ASM_INST(0x8543c06a)  // ld1rw	{z10.s}, p0/z, [x3, #12]
    KAI_ASM_INST(0x910103e5)  // add	x5, sp, #0x40
    KAI_ASM_INST(0x6d0237ec)  // stp	d12, d13, [sp, #32]
    KAI_ASM_INST(0x8541c06d)  // ld1rw	{z13.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0x8551c06c)  // ld1rw	{z12.s}, p0/z, [x3, #68]
    KAI_ASM_INST(0x6d033fee)  // stp	d14, d15, [sp, #48]
    KAI_ASM_INST(0x8540c06e)  // ld1rw	{z14.s}, p0/z, [x3]
    KAI_ASM_INST(0x855ac067)  // ld1rw	{z7.s}, p0/z, [x3, #104]
    KAI_ASM_INST(0x855cc060)  // ld1rw	{z0.s}, p0/z, [x3, #112]
    KAI_ASM_INST(0xe58040a0)  // str	z0, [x5]
    KAI_ASM_INST(0x855dc060)  // ld1rw	{z0.s}, p0/z, [x3, #116]
    KAI_ASM_INST(0xe58044a0)  // str	z0, [x5, #1, mul vl]
    KAI_ASM_INST(0x855fc060)  // ld1rw	{z0.s}, p0/z, [x3, #124]
    KAI_ASM_INST(0xe58048a0)  // str	z0, [x5, #2, mul vl]
    KAI_ASM_INST(0x855bc066)  // ld1rw	{z6.s}, p0/z, [x3, #108]
    KAI_ASM_INST(0x8550c064)  // ld1rw	{z4.s}, p0/z, [x3, #64]
    KAI_ASM_INST(0xb4000762)  // cbz	x2, 1b0 <.SVLPEND0+0x18c>
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0x04a0e3e3)  // cntw	x3
    KAI_ASM_INST(0xe5804ca6)  // str	z6, [x5, #3, mul vl]
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25a21c80)  // whilelo	p0.s, x4, x2
    KAI_ASM_INST(0xa5444003)  // ld1w	{z3.s}, p0/z, [x0, x4, lsl #2]
    KAI_ASM_INST(0x04e43061)  // bic	z1.d, z3.d, z4.d
    KAI_ASM_INST(0x658781c1)  // fmin	z1.s, p0/m, z1.s, z14.s
    KAI_ASM_INST(0x0420bf80)  // movprfx	z0, z28
    KAI_ASM_INST(0x65a103a0)  // fmla	z0.s, p0/m, z29.s, z1.s
    KAI_ASM_INST(0x65bb8020)  // fmad	z0.s, p0/m, z1.s, z27.s
    KAI_ASM_INST(0x65ba8020)  // fmad	z0.s, p0/m, z1.s, z26.s
    KAI_ASM_INST(0x65b98020)  // fmad	z0.s, p0/m, z1.s, z25.s
    KAI_ASM_INST(0x65b88020)  // fmad	z0.s, p0/m, z1.s, z24.s
    KAI_ASM_INST(0x65b78020)  // fmad	z0.s, p0/m, z1.s, z23.s
    KAI_ASM_INST(0x65a18020)  // fmad	z0.s, p0/m, z1.s, z1.s
    KAI_ASM_INST(0x04a43000)  // eor	z0.d, z0.d, z4.d
    KAI_ASM_INST(0x658682c0)  // fmax	z0.s, p0/m, z0.s, z22.s
    KAI_ASM_INST(0x0420bcaf)  // movprfx	z15, z5
    KAI_ASM_INST(0x65b5000f)  // fmla	z15.s, p0/m, z0.s, z21.s
    KAI_ASM_INST(0x658180af)  // fsub	z15.s, p0/m, z15.s, z5.s
    KAI_ASM_INST(0x65b401e0)  // fmla	z0.s, p0/m, z15.s, z20.s
    KAI_ASM_INST(0x910103e5)  // add	x5, sp, #0x40
    KAI_ASM_INST(0x0420bde2)  // movprfx	z2, z15
    KAI_ASM_INST(0x659ca1e2)  // fcvtzs	z2.s, p0/m, z15.s
    KAI_ASM_INST(0x85804ca6)  // ldr	z6, [x5, #3, mul vl]
    KAI_ASM_INST(0x65a0826f)  // fmad	z15.s, p0/m, z19.s, z0.s
    KAI_ASM_INST(0x858048a0)  // ldr	z0, [x5, #2, mul vl]
    KAI_ASM_INST(0x04a00042)  // add	z2.s, z2.s, z0.s
    KAI_ASM_INST(0x0420be20)  // movprfx	z0, z17
    KAI_ASM_INST(0x65af0240)  // fmla	z0.s, p0/m, z18.s, z15.s
    KAI_ASM_INST(0x04779c42)  // lsl	z2.s, z2.s, #23
    KAI_ASM_INST(0x65b081e0)  // fmad	z0.s, p0/m, z15.s, z16.s
    KAI_ASM_INST(0x65a781e0)  // fmad	z0.s, p0/m, z15.s, z7.s
    KAI_ASM_INST(0x65a681e0)  // fmad	z0.s, p0/m, z15.s, z6.s
    KAI_ASM_INST(0x858040a6)  // ldr	z6, [x5]
    KAI_ASM_INST(0x658d4031)  // fcmgt	p1.s, p0/z, z1.s, z13.s
    KAI_ASM_INST(0x65a681e0)  // fmad	z0.s, p0/m, z15.s, z6.s
    KAI_ASM_INST(0x04233083)  // and	z3.d, z4.d, z3.d
    KAI_ASM_INST(0x858044a6)  // ldr	z6, [x5, #1, mul vl]
    KAI_ASM_INST(0x65a681e0)  // fmad	z0.s, p0/m, z15.s, z6.s
    KAI_ASM_INST(0x0420bc2f)  // movprfx	z15, z1
    KAI_ASM_INST(0x6582802f)  // fmul	z15.s, p0/m, z15.s, z1.s
    KAI_ASM_INST(0x65828040)  // fmul	z0.s, p0/m, z0.s, z2.s
    KAI_ASM_INST(0x0420bd42)  // movprfx	z2, z10
    KAI_ASM_INST(0x65af0162)  // fmla	z2.s, p0/m, z11.s, z15.s
    KAI_ASM_INST(0x65838180)  // fsubr	z0.s, p0/m, z0.s, z12.s
    KAI_ASM_INST(0x65a981e2)  // fmad	z2.s, p0/m, z15.s, z9.s
    KAI_ASM_INST(0x65a881e2)  // fmad	z2.s, p0/m, z15.s, z8.s
    KAI_ASM_INST(0x65bf81e2)  // fmad	z2.s, p0/m, z15.s, z31.s
    KAI_ASM_INST(0x65be81e2)  // fmad	z2.s, p0/m, z15.s, z30.s
    KAI_ASM_INST(0x65a10041)  // fmla	z1.s, p0/m, z2.s, z1.s
    KAI_ASM_INST(0x05a1c400)  // sel	z0.s, p1, z0.s, z1.s
    KAI_ASM_INST(0x04633000)  // orr	z0.d, z0.d, z3.d
    KAI_ASM_INST(0xe5444020)  // st1w	{z0.s}, p0, [x1, x4, lsl #2]
    KAI_ASM_INST(0x8b030084)  // add	x4, x4, x3
    KAI_ASM_INST(0xeb04005f)  // cmp	x2, x4
    KAI_ASM_INST(0x54fff968)  // b.hi	d8 <.SVLPEND0+0xb4>
    KAI_ASM_INST(0x6d4027e8)  // ldp	d8, d9, [sp]
    KAI_ASM_INST(0x6d412fea)  // ldp	d10, d11, [sp, #16]
    KAI_ASM_INST(0x6d4237ec)  // ldp	d12, d13, [sp, #32]
    KAI_ASM_INST(0x6d433fee)  // ldp	d14, d15, [sp, #48]
    KAI_ASM_INST(0x043f509f)  // addvl	sp, sp, #4
    KAI_ASM_INST(0x910103ff)  // add	sp, sp, #0x40
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveErfKernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveLogisticKernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveLogisticKernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveLogisticKernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8540c064)  // ld1rw	{z4.s}, p0/z, [x3]
    KAI_ASM_INST(0x8541c065)  // ld1rw	{z5.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0x8542c066)  // ld1rw	{z6.s}, p0/z, [x3, #8]
    KAI_ASM_INST(0x8543c067)  // ld1rw	{z7.s}, p0/z, [x3, #12]
    KAI_ASM_INST(0x8544c070)  // ld1rw	{z16.s}, p0/z, [x3, #16]
    KAI_ASM_INST(0x8545c071)  // ld1rw	{z17.s}, p0/z, [x3, #20]
    KAI_ASM_INST(0x8546c072)  // ld1rw	{z18.s}, p0/z, [x3, #24]
    KAI_ASM_INST(0x8547c073)  // ld1rw	{z19.s}, p0/z, [x3, #28]
    KAI_ASM_INST(0x8548c074)  // ld1rw	{z20.s}, p0/z, [x3, #32]
    KAI_ASM_INST(0x8549c075)  // ld1rw	{z21.s}, p0/z, [x3, #36]
    KAI_ASM_INST(0x854ac076)  // ld1rw	{z22.s}, p0/z, [x3, #40]
    KAI_ASM_INST(0x854bc077)  // ld1rw	{z23.s}, p0/z, [x3, #44]
    KAI_ASM_INST(0x854cc078)  // ld1rw	{z24.s}, p0/z, [x3, #48]
    KAI_ASM_INST(0x854dc079)  // ld1rw	{z25.s}, p0/z, [x3, #52]
    KAI_ASM_INST(0xb40003e2)  // cbz	x2, b8 <MlasSveLogisticKernelImpl+0xb8>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x04a0e3e4)  // cntw	x4
    KAI_ASM_INST(0x25b8c01b)  // mov	z27.s, #0
    KAI_ASM_INST(0x25b9ce1a)  // fmov	z26.s, #1.000000000000000000e+00
    KAI_ASM_INST(0x25a21c60)  // whilelo	p0.s, x3, x2
    KAI_ASM_INST(0xa5434003)  // ld1w	{z3.s}, p0/z, [x0, x3, lsl #2]
    KAI_ASM_INST(0x65868083)  // fmax	z3.s, p0/m, z3.s, z4.s
    KAI_ASM_INST(0x658780a3)  // fmin	z3.s, p0/m, z3.s, z5.s
    KAI_ASM_INST(0x0420bc62)  // movprfx	z2, z3
    KAI_ASM_INST(0x65828062)  // fmul	z2.s, p0/m, z2.s, z3.s
    KAI_ASM_INST(0x0420bce0)  // movprfx	z0, z7
    KAI_ASM_INST(0x65a60040)  // fmla	z0.s, p0/m, z2.s, z6.s
    KAI_ASM_INST(0x0420be81)  // movprfx	z1, z20
    KAI_ASM_INST(0x65b30041)  // fmla	z1.s, p0/m, z2.s, z19.s
    KAI_ASM_INST(0x65b08040)  // fmad	z0.s, p0/m, z2.s, z16.s
    KAI_ASM_INST(0x65b58041)  // fmad	z1.s, p0/m, z2.s, z21.s
    KAI_ASM_INST(0x65b18040)  // fmad	z0.s, p0/m, z2.s, z17.s
    KAI_ASM_INST(0x65b68041)  // fmad	z1.s, p0/m, z2.s, z22.s
    KAI_ASM_INST(0x65b28040)  // fmad	z0.s, p0/m, z2.s, z18.s
    KAI_ASM_INST(0x65b78041)  // fmad	z1.s, p0/m, z2.s, z23.s
    KAI_ASM_INST(0x65828060)  // fmul	z0.s, p0/m, z0.s, z3.s
    KAI_ASM_INST(0x65b88041)  // fmad	z1.s, p0/m, z2.s, z24.s
    KAI_ASM_INST(0x658d8020)  // fdiv	z0.s, p0/m, z0.s, z1.s
    KAI_ASM_INST(0x65808320)  // fadd	z0.s, p0/m, z0.s, z25.s
    KAI_ASM_INST(0x65868360)  // fmax	z0.s, p0/m, z0.s, z27.s
    KAI_ASM_INST(0x65878340)  // fmin	z0.s, p0/m, z0.s, z26.s
    KAI_ASM_INST(0xe5434020)  // st1w	{z0.s}, p0, [x1, x3, lsl #2]
    KAI_ASM_INST(0x8b040063)  // add	x3, x3, x4
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffce8)  // b.hi	50 <MlasSveLogisticKernelImpl+0x50>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveLogisticKernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveTanhKernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveTanhKernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveTanhKernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8540c064)  // ld1rw	{z4.s}, p0/z, [x3]
    KAI_ASM_INST(0x8541c065)  // ld1rw	{z5.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0x8542c066)  // ld1rw	{z6.s}, p0/z, [x3, #8]
    KAI_ASM_INST(0x8543c067)  // ld1rw	{z7.s}, p0/z, [x3, #12]
    KAI_ASM_INST(0x8544c070)  // ld1rw	{z16.s}, p0/z, [x3, #16]
    KAI_ASM_INST(0x8545c071)  // ld1rw	{z17.s}, p0/z, [x3, #20]
    KAI_ASM_INST(0x8546c072)  // ld1rw	{z18.s}, p0/z, [x3, #24]
    KAI_ASM_INST(0x8547c073)  // ld1rw	{z19.s}, p0/z, [x3, #28]
    KAI_ASM_INST(0x8548c074)  // ld1rw	{z20.s}, p0/z, [x3, #32]
    KAI_ASM_INST(0x8549c075)  // ld1rw	{z21.s}, p0/z, [x3, #36]
    KAI_ASM_INST(0x854ac076)  // ld1rw	{z22.s}, p0/z, [x3, #40]
    KAI_ASM_INST(0x854bc077)  // ld1rw	{z23.s}, p0/z, [x3, #44]
    KAI_ASM_INST(0x854cc078)  // ld1rw	{z24.s}, p0/z, [x3, #48]
    KAI_ASM_INST(0xb4000362)  // cbz	x2, a4 <MlasSveTanhKernelImpl+0xa4>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x04a0e3e4)  // cntw	x4
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25a21c60)  // whilelo	p0.s, x3, x2
    KAI_ASM_INST(0xa5434002)  // ld1w	{z2.s}, p0/z, [x0, x3, lsl #2]
    KAI_ASM_INST(0x65868082)  // fmax	z2.s, p0/m, z2.s, z4.s
    KAI_ASM_INST(0x658780a2)  // fmin	z2.s, p0/m, z2.s, z5.s
    KAI_ASM_INST(0x0420bc41)  // movprfx	z1, z2
    KAI_ASM_INST(0x65828041)  // fmul	z1.s, p0/m, z1.s, z2.s
    KAI_ASM_INST(0x0420bce0)  // movprfx	z0, z7
    KAI_ASM_INST(0x65a60020)  // fmla	z0.s, p0/m, z1.s, z6.s
    KAI_ASM_INST(0x0420bec3)  // movprfx	z3, z22
    KAI_ASM_INST(0x65b50023)  // fmla	z3.s, p0/m, z1.s, z21.s
    KAI_ASM_INST(0x65b08020)  // fmad	z0.s, p0/m, z1.s, z16.s
    KAI_ASM_INST(0x65b78023)  // fmad	z3.s, p0/m, z1.s, z23.s
    KAI_ASM_INST(0x65b18020)  // fmad	z0.s, p0/m, z1.s, z17.s
    KAI_ASM_INST(0x65b88023)  // fmad	z3.s, p0/m, z1.s, z24.s
    KAI_ASM_INST(0x65b28020)  // fmad	z0.s, p0/m, z1.s, z18.s
    KAI_ASM_INST(0x65b38020)  // fmad	z0.s, p0/m, z1.s, z19.s
    KAI_ASM_INST(0x65b48020)  // fmad	z0.s, p0/m, z1.s, z20.s
    KAI_ASM_INST(0x65828040)  // fmul	z0.s, p0/m, z0.s, z2.s
    KAI_ASM_INST(0x658d8060)  // fdiv	z0.s, p0/m, z0.s, z3.s
    KAI_ASM_INST(0xe5434020)  // st1w	{z0.s}, p0, [x1, x3, lsl #2]
    KAI_ASM_INST(0x8b040063)  // add	x3, x3, x4
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffd48)  // b.hi	48 <MlasSveTanhKernelImpl+0x48>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveTanhKernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveComputeExpF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveComputeExpF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveComputeExpF32KernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8540c066)  // ld1rw	{z6.s}, p0/z, [x3]
    KAI_ASM_INST(0x8541c067)  // ld1rw	{z7.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0x8542c070)  // ld1rw	{z16.s}, p0/z, [x3, #8]
    KAI_ASM_INST(0x8543c071)  // ld1rw	{z17.s}, p0/z, [x3, #12]
    KAI_ASM_INST(0x8544c072)  // ld1rw	{z18.s}, p0/z, [x3, #16]
    KAI_ASM_INST(0x8545c064)  // ld1rw	{z4.s}, p0/z, [x3, #20]
    KAI_ASM_INST(0x8546c073)  // ld1rw	{z19.s}, p0/z, [x3, #24]
    KAI_ASM_INST(0x8547c074)  // ld1rw	{z20.s}, p0/z, [x3, #28]
    KAI_ASM_INST(0x8548c075)  // ld1rw	{z21.s}, p0/z, [x3, #32]
    KAI_ASM_INST(0x8549c076)  // ld1rw	{z22.s}, p0/z, [x3, #36]
    KAI_ASM_INST(0x854ac077)  // ld1rw	{z23.s}, p0/z, [x3, #40]
    KAI_ASM_INST(0x854bc078)  // ld1rw	{z24.s}, p0/z, [x3, #44]
    KAI_ASM_INST(0xb40003e2)  // cbz	x2, b0 <MlasSveComputeExpF32KernelImpl+0xb0>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x04a0e3e4)  // cntw	x4
    KAI_ASM_INST(0x25a21c60)  // whilelo	p0.s, x3, x2
    KAI_ASM_INST(0xa5434002)  // ld1w	{z2.s}, p0/z, [x0, x3, lsl #2]
    KAI_ASM_INST(0x0420bc80)  // movprfx	z0, z4
    KAI_ASM_INST(0x65b30040)  // fmla	z0.s, p0/m, z2.s, z19.s
    KAI_ASM_INST(0x0420bc01)  // movprfx	z1, z0
    KAI_ASM_INST(0x65818081)  // fsub	z1.s, p0/m, z1.s, z4.s
    KAI_ASM_INST(0x0420bc43)  // movprfx	z3, z2
    KAI_ASM_INST(0x65b40023)  // fmla	z3.s, p0/m, z1.s, z20.s
    KAI_ASM_INST(0x65824312)  // fcmgt	p2.s, p0/z, z24.s, z2.s
    KAI_ASM_INST(0x65a382a1)  // fmad	z1.s, p0/m, z21.s, z3.s
    KAI_ASM_INST(0x65974051)  // fcmgt	p1.s, p0/z, z2.s, z23.s
    KAI_ASM_INST(0x0420bc23)  // movprfx	z3, z1
    KAI_ASM_INST(0x65828023)  // fmul	z3.s, p0/m, z3.s, z1.s
    KAI_ASM_INST(0x0420bce5)  // movprfx	z5, z7
    KAI_ASM_INST(0x65a10205)  // fmla	z5.s, p0/m, z16.s, z1.s
    KAI_ASM_INST(0x0420be22)  // movprfx	z2, z17
    KAI_ASM_INST(0x65a10242)  // fmla	z2.s, p0/m, z18.s, z1.s
    KAI_ASM_INST(0x04779c00)  // lsl	z0.s, z0.s, #23
    KAI_ASM_INST(0x658280c1)  // fmul	z1.s, p0/m, z1.s, z6.s
    KAI_ASM_INST(0x65a58062)  // fmad	z2.s, p0/m, z3.s, z5.s
    KAI_ASM_INST(0x65a30041)  // fmla	z1.s, p0/m, z2.s, z3.s
    KAI_ASM_INST(0x65a00020)  // fmla	z0.s, p0/m, z1.s, z0.s
    KAI_ASM_INST(0x05924000)  // mov	z0.s, p2/m, #0
    KAI_ASM_INST(0x05a0c6c0)  // mov	z0.s, p1/m, z22.s
    KAI_ASM_INST(0xe5434020)  // st1w	{z0.s}, p0, [x1, x3, lsl #2]
    KAI_ASM_INST(0x8b040063)  // add	x3, x3, x4
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffca8)  // b.hi	40 <MlasSveComputeExpF32KernelImpl+0x40>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveComputeExpF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveComputeSumExpF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveComputeSumExpF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveComputeSumExpF32KernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x854ec0b1)  // ld1rw	{z17.s}, p0/z, [x5, #56]
    KAI_ASM_INST(0x8546c0b2)  // ld1rw	{z18.s}, p0/z, [x5, #24]
    KAI_ASM_INST(0x8547c0b3)  // ld1rw	{z19.s}, p0/z, [x5, #28]
    KAI_ASM_INST(0x8548c0b4)  // ld1rw	{z20.s}, p0/z, [x5, #32]
    KAI_ASM_INST(0x854cc0a7)  // ld1rw	{z7.s}, p0/z, [x5, #48]
    KAI_ASM_INST(0x854dc0b5)  // ld1rw	{z21.s}, p0/z, [x5, #52]
    KAI_ASM_INST(0x8540c076)  // ld1rw	{z22.s}, p0/z, [x3]
    KAI_ASM_INST(0xb4000602)  // cbz	x2, e0 <MlasSveComputeSumExpF32KernelImpl+0xe0>
    KAI_ASM_INST(0x04a0e3e4)  // cntw	x4
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x0460e3e5)  // cnth	x5
    KAI_ASM_INST(0x25b8c010)  // mov	z16.s, #0
    KAI_ASM_INST(0x04703200)  // mov	z0.d, z16.d
    KAI_ASM_INST(0x25a21c61)  // whilelo	p1.s, x3, x2
    KAI_ASM_INST(0x25a21c80)  // whilelo	p0.s, x4, x2
    KAI_ASM_INST(0xa5434404)  // ld1w	{z4.s}, p1/z, [x0, x3, lsl #2]
    KAI_ASM_INST(0xa5444003)  // ld1w	{z3.s}, p0/z, [x0, x4, lsl #2]
    KAI_ASM_INST(0x658086c4)  // fadd	z4.s, p1/m, z4.s, z22.s
    KAI_ASM_INST(0x658082c3)  // fadd	z3.s, p0/m, z3.s, z22.s
    KAI_ASM_INST(0x65868624)  // fmax	z4.s, p1/m, z4.s, z17.s
    KAI_ASM_INST(0x65868223)  // fmax	z3.s, p0/m, z3.s, z17.s
    KAI_ASM_INST(0x0420bce2)  // movprfx	z2, z7
    KAI_ASM_INST(0x65b20482)  // fmla	z2.s, p1/m, z4.s, z18.s
    KAI_ASM_INST(0x0420bce1)  // movprfx	z1, z7
    KAI_ASM_INST(0x65b20061)  // fmla	z1.s, p0/m, z3.s, z18.s
    KAI_ASM_INST(0x04a0b846)  // fexpa	z6.s, z2.s
    KAI_ASM_INST(0x04a0b825)  // fexpa	z5.s, z1.s
    KAI_ASM_INST(0x658184e2)  // fsub	z2.s, p1/m, z2.s, z7.s
    KAI_ASM_INST(0x658180e1)  // fsub	z1.s, p0/m, z1.s, z7.s
    KAI_ASM_INST(0x65b30444)  // fmla	z4.s, p1/m, z2.s, z19.s
    KAI_ASM_INST(0x65b30023)  // fmla	z3.s, p0/m, z1.s, z19.s
    KAI_ASM_INST(0x65a48682)  // fmad	z2.s, p1/m, z20.s, z4.s
    KAI_ASM_INST(0x65a38281)  // fmad	z1.s, p0/m, z20.s, z3.s
    KAI_ASM_INST(0x0420bc44)  // movprfx	z4, z2
    KAI_ASM_INST(0x65828444)  // fmul	z4.s, p1/m, z4.s, z2.s
    KAI_ASM_INST(0x0420bc43)  // movprfx	z3, z2
    KAI_ASM_INST(0x65b50483)  // fmla	z3.s, p1/m, z4.s, z21.s
    KAI_ASM_INST(0x0420bc22)  // movprfx	z2, z1
    KAI_ASM_INST(0x65828022)  // fmul	z2.s, p0/m, z2.s, z1.s
    KAI_ASM_INST(0x65a304c6)  // fmla	z6.s, p1/m, z6.s, z3.s
    KAI_ASM_INST(0x65b50041)  // fmla	z1.s, p0/m, z2.s, z21.s
    KAI_ASM_INST(0x65a100a5)  // fmla	z5.s, p0/m, z5.s, z1.s
    KAI_ASM_INST(0xb4000061)  // cbz	x1, b8 <MlasSveComputeSumExpF32KernelImpl+0xb8>
    KAI_ASM_INST(0xe5434426)  // st1w	{z6.s}, p1, [x1, x3, lsl #2]
    KAI_ASM_INST(0xe5444025)  // st1w	{z5.s}, p0, [x1, x4, lsl #2]
    KAI_ASM_INST(0x8b050063)  // add	x3, x3, x5
    KAI_ASM_INST(0x8b050084)  // add	x4, x4, x5
    KAI_ASM_INST(0x658084c0)  // fadd	z0.s, p1/m, z0.s, z6.s
    KAI_ASM_INST(0x658080b0)  // fadd	z16.s, p0/m, z16.s, z5.s
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffb68)  // b.hi	38 <MlasSveComputeSumExpF32KernelImpl+0x38>
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x65900000)  // fadd	z0.s, z0.s, z16.s
    KAI_ASM_INST(0x65802000)  // faddv	s0, p0, z0.s
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0x25b8c010)  // mov	z16.s, #0
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x04703200)  // mov	z0.d, z16.d
    KAI_ASM_INST(0x65900000)  // fadd	z0.s, z0.s, z16.s
    KAI_ASM_INST(0x65802000)  // faddv	s0, p0, z0.s
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveComputeSumExpF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveReduceMaximumF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveReduceMaximumF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveReduceMaximumF32KernelImpl)
    KAI_ASM_INST(0x0420e3e2)  // cntb	x2
    KAI_ASM_INST(0x05242000)  // mov	z0.s, s0
    KAI_ASM_INST(0xeb02003f)  // cmp	x1, x2
    KAI_ASM_INST(0x540002c3)  // b.cc	64 <MlasSveReduceMaximumF32KernelImpl+0x64>
    KAI_ASM_INST(0x0423e3e3)  // cntb	x3, all, mul #4
    KAI_ASM_INST(0x04603002)  // mov	z2.d, z0.d
    KAI_ASM_INST(0x04603001)  // mov	z1.d, z0.d
    KAI_ASM_INST(0x04603003)  // mov	z3.d, z0.d
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xa540a005)  // ld1w	{z5.s}, p0/z, [x0]
    KAI_ASM_INST(0xa541a004)  // ld1w	{z4.s}, p0/z, [x0, #1, mul vl]
    KAI_ASM_INST(0xcb020021)  // sub	x1, x1, x2
    KAI_ASM_INST(0x658680a0)  // fmax	z0.s, p0/m, z0.s, z5.s
    KAI_ASM_INST(0x65868083)  // fmax	z3.s, p0/m, z3.s, z4.s
    KAI_ASM_INST(0xa542a005)  // ld1w	{z5.s}, p0/z, [x0, #2, mul vl]
    KAI_ASM_INST(0xa543a004)  // ld1w	{z4.s}, p0/z, [x0, #3, mul vl]
    KAI_ASM_INST(0x658680a1)  // fmax	z1.s, p0/m, z1.s, z5.s
    KAI_ASM_INST(0x8b030000)  // add	x0, x0, x3
    KAI_ASM_INST(0x65868082)  // fmax	z2.s, p0/m, z2.s, z4.s
    KAI_ASM_INST(0xeb02003f)  // cmp	x1, x2
    KAI_ASM_INST(0x54fffea2)  // b.cs	28 <MlasSveReduceMaximumF32KernelImpl+0x28>
    KAI_ASM_INST(0x65868060)  // fmax	z0.s, p0/m, z0.s, z3.s
    KAI_ASM_INST(0x65868041)  // fmax	z1.s, p0/m, z1.s, z2.s
    KAI_ASM_INST(0x65868020)  // fmax	z0.s, p0/m, z0.s, z1.s
    KAI_ASM_INST(0xb4000121)  // cbz	x1, 88 <MlasSveReduceMaximumF32KernelImpl+0x88>
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0x04a0e3e3)  // cntw	x3
    KAI_ASM_INST(0x25a11c40)  // whilelo	p0.s, x2, x1
    KAI_ASM_INST(0xa5424001)  // ld1w	{z1.s}, p0/z, [x0, x2, lsl #2]
    KAI_ASM_INST(0x8b030042)  // add	x2, x2, x3
    KAI_ASM_INST(0x65868020)  // fmax	z0.s, p0/m, z0.s, z1.s
    KAI_ASM_INST(0xeb02003f)  // cmp	x1, x2
    KAI_ASM_INST(0x54ffff68)  // b.hi	70 <MlasSveReduceMaximumF32KernelImpl+0x70>
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x65862000)  // fmaxv	s0, p0, z0.s
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveReduceMaximumF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveReduceMinimumMaximumF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveReduceMinimumMaximumF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveReduceMinimumMaximumF32KernelImpl)
    KAI_ASM_INST(0x0420e3e5)  // cntb	x5
    KAI_ASM_INST(0xeb05007f)  // cmp	x3, x5
    KAI_ASM_INST(0x54000663)  // b.cc	d4 <MlasSveReduceMinimumMaximumF32KernelImpl+0xd4>
    KAI_ASM_INST(0x0423e3e6)  // cntb	x6, all, mul #4
    KAI_ASM_INST(0x05c043c4)  // mov	z4.s, #0xff7fffff
    KAI_ASM_INST(0x12b01004)  // mov	w4, #0x7f7fffff
    KAI_ASM_INST(0x04643085)  // mov	z5.d, z4.d
    KAI_ASM_INST(0x05a03880)  // mov	z0.s, w4
    KAI_ASM_INST(0x04643087)  // mov	z7.d, z4.d
    KAI_ASM_INST(0x04603010)  // mov	z16.d, z0.d
    KAI_ASM_INST(0x04603006)  // mov	z6.d, z0.d
    KAI_ASM_INST(0x04603011)  // mov	z17.d, z0.d
    KAI_ASM_INST(0x04643081)  // mov	z1.d, z4.d
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0xa540a003)  // ld1w	{z3.s}, p0/z, [x0]
    KAI_ASM_INST(0xa541a002)  // ld1w	{z2.s}, p0/z, [x0, #1, mul vl]
    KAI_ASM_INST(0xcb050063)  // sub	x3, x3, x5
    KAI_ASM_INST(0x65878060)  // fmin	z0.s, p0/m, z0.s, z3.s
    KAI_ASM_INST(0x65868061)  // fmax	z1.s, p0/m, z1.s, z3.s
    KAI_ASM_INST(0x65878051)  // fmin	z17.s, p0/m, z17.s, z2.s
    KAI_ASM_INST(0xa542a003)  // ld1w	{z3.s}, p0/z, [x0, #2, mul vl]
    KAI_ASM_INST(0x65868047)  // fmax	z7.s, p0/m, z7.s, z2.s
    KAI_ASM_INST(0x65878066)  // fmin	z6.s, p0/m, z6.s, z3.s
    KAI_ASM_INST(0xa543a002)  // ld1w	{z2.s}, p0/z, [x0, #3, mul vl]
    KAI_ASM_INST(0x65868065)  // fmax	z5.s, p0/m, z5.s, z3.s
    KAI_ASM_INST(0x8b060000)  // add	x0, x0, x6
    KAI_ASM_INST(0x65878050)  // fmin	z16.s, p0/m, z16.s, z2.s
    KAI_ASM_INST(0x65868044)  // fmax	z4.s, p0/m, z4.s, z2.s
    KAI_ASM_INST(0xeb05007f)  // cmp	x3, x5
    KAI_ASM_INST(0x54fffe22)  // b.cs	38 <MlasSveReduceMinimumMaximumF32KernelImpl+0x38>
    KAI_ASM_INST(0x65878220)  // fmin	z0.s, p0/m, z0.s, z17.s
    KAI_ASM_INST(0x65878206)  // fmin	z6.s, p0/m, z6.s, z16.s
    KAI_ASM_INST(0x658680e1)  // fmax	z1.s, p0/m, z1.s, z7.s
    KAI_ASM_INST(0x65868085)  // fmax	z5.s, p0/m, z5.s, z4.s
    KAI_ASM_INST(0x658780c0)  // fmin	z0.s, p0/m, z0.s, z6.s
    KAI_ASM_INST(0x658680a1)  // fmax	z1.s, p0/m, z1.s, z5.s
    KAI_ASM_INST(0xb4000163)  // cbz	x3, bc <MlasSveReduceMinimumMaximumF32KernelImpl+0xbc>
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0x04a0e3e5)  // cntw	x5
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25a31c80)  // whilelo	p0.s, x4, x3
    KAI_ASM_INST(0xa5444002)  // ld1w	{z2.s}, p0/z, [x0, x4, lsl #2]
    KAI_ASM_INST(0x8b050084)  // add	x4, x4, x5
    KAI_ASM_INST(0x65878040)  // fmin	z0.s, p0/m, z0.s, z2.s
    KAI_ASM_INST(0x65868041)  // fmax	z1.s, p0/m, z1.s, z2.s
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54ffff48)  // b.hi	a0 <MlasSveReduceMinimumMaximumF32KernelImpl+0xa0>
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x65872000)  // fminv	s0, p0, z0.s
    KAI_ASM_INST(0xbd000020)  // str	s0, [x1]
    KAI_ASM_INST(0x65862021)  // fmaxv	s1, p0, z1.s
    KAI_ASM_INST(0xbd000041)  // str	s1, [x2]
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0x12b01004)  // mov	w4, #0x7f7fffff
    KAI_ASM_INST(0x05c043c1)  // mov	z1.s, #0xff7fffff
    KAI_ASM_INST(0x05a03880)  // mov	z0.s, w4
    KAI_ASM_INST(0x17ffffec)  // b	90 <MlasSveReduceMinimumMaximumF32KernelImpl+0x90>
    KAI_ASM_FUNCTION_END(MlasSveReduceMinimumMaximumF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveComputeSoftmaxOutputF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveComputeSoftmaxOutputF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveComputeSoftmaxOutputF32KernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8540c041)  // ld1rw	{z1.s}, p0/z, [x2]
    KAI_ASM_INST(0xb4000161)  // cbz	x1, 34 <MlasSveComputeSoftmaxOutputF32KernelImpl+0x34>
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0x04a0e3e3)  // cntw	x3
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25a11c40)  // whilelo	p0.s, x2, x1
    KAI_ASM_INST(0xa5424000)  // ld1w	{z0.s}, p0/z, [x0, x2, lsl #2]
    KAI_ASM_INST(0x65828020)  // fmul	z0.s, p0/m, z0.s, z1.s
    KAI_ASM_INST(0xe5424000)  // st1w	{z0.s}, p0, [x0, x2, lsl #2]
    KAI_ASM_INST(0x8b030042)  // add	x2, x2, x3
    KAI_ASM_INST(0xeb02003f)  // cmp	x1, x2
    KAI_ASM_INST(0x54ffff48)  // b.hi	18 <MlasSveComputeSoftmaxOutputF32KernelImpl+0x18>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveComputeSoftmaxOutputF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveComputeLogSoftmaxOutputF32KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveComputeLogSoftmaxOutputF32KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveComputeLogSoftmaxOutputF32KernelImpl)
    KAI_ASM_INST(0x2518e3e0)  // ptrue	p0.b
    KAI_ASM_INST(0x8540c061)  // ld1rw	{z1.s}, p0/z, [x3]
    KAI_ASM_INST(0x8541c062)  // ld1rw	{z2.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0xb4000162)  // cbz	x2, 38 <MlasSveComputeLogSoftmaxOutputF32KernelImpl+0x38>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x04a0e3e4)  // cntw	x4
    KAI_ASM_INST(0x25a21c60)  // whilelo	p0.s, x3, x2
    KAI_ASM_INST(0xa5434000)  // ld1w	{z0.s}, p0/z, [x0, x3, lsl #2]
    KAI_ASM_INST(0x65808020)  // fadd	z0.s, p0/m, z0.s, z1.s
    KAI_ASM_INST(0x65818040)  // fsub	z0.s, p0/m, z0.s, z2.s
    KAI_ASM_INST(0xe5434020)  // st1w	{z0.s}, p0, [x1, x3, lsl #2]
    KAI_ASM_INST(0x8b040063)  // add	x3, x3, x4
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54ffff28)  // b.hi	18 <MlasSveComputeLogSoftmaxOutputF32KernelImpl+0x18>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveComputeLogSoftmaxOutputF32KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveTanhFP16KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveTanhFP16KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveTanhFP16KernelImpl)
    KAI_ASM_INST(0xb4000502)  // cbz	x2, a0 <MlasSveTanhFP16KernelImpl+0xa0>
    KAI_ASM_INST(0x4f078730)  // movi	v16.8h, #0xf9
    KAI_ASM_INST(0x12879ee4)  // mov	w4, #0xffffc308
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x05603891)  // mov	z17.h, w4
    KAI_ASM_INST(0x0460e3e6)  // cnth	x6
    KAI_ASM_INST(0x52822704)  // mov	w4, #0x1138
    KAI_ASM_INST(0x52886105)  // mov	w5, #0x4308
    KAI_ASM_INST(0x05603887)  // mov	z7.h, w4
    KAI_ASM_INST(0x5283a064)  // mov	w4, #0x1d03
    KAI_ASM_INST(0x056038b2)  // mov	z18.h, w5
    KAI_ASM_INST(0x05603884)  // mov	z4.h, w4
    KAI_ASM_INST(0x5280f8a5)  // mov	w5, #0x7c5
    KAI_ASM_INST(0x528314a4)  // mov	w4, #0x18a5
    KAI_ASM_INST(0x05302210)  // mov	z16.q, q16
    KAI_ASM_INST(0x056038a6)  // mov	z6.h, w5
    KAI_ASM_INST(0x2578c034)  // mov	z20.h, #1
    KAI_ASM_INST(0x05603885)  // mov	z5.h, w4
    KAI_ASM_INST(0x2578c293)  // mov	z19.h, #20
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25621c60)  // whilelo	p0.h, x3, x2
    KAI_ASM_INST(0xa4a34001)  // ld1h	{z1.h}, p0/z, [x0, x3, lsl #1]
    KAI_ASM_INST(0x65478241)  // fmin	z1.h, p0/m, z1.h, z18.h
    KAI_ASM_INST(0x65468221)  // fmax	z1.h, p0/m, z1.h, z17.h
    KAI_ASM_INST(0x0420bc22)  // movprfx	z2, z1
    KAI_ASM_INST(0x65428022)  // fmul	z2.h, p0/m, z2.h, z1.h
    KAI_ASM_INST(0x0420be00)  // movprfx	z0, z16
    KAI_ASM_INST(0x65620280)  // fmla	z0.h, p0/m, z20.h, z2.h
    KAI_ASM_INST(0x0420bcc3)  // movprfx	z3, z6
    KAI_ASM_INST(0x65620263)  // fmla	z3.h, p0/m, z19.h, z2.h
    KAI_ASM_INST(0x65678040)  // fmad	z0.h, p0/m, z2.h, z7.h
    KAI_ASM_INST(0x65658043)  // fmad	z3.h, p0/m, z2.h, z5.h
    KAI_ASM_INST(0x65648040)  // fmad	z0.h, p0/m, z2.h, z4.h
    KAI_ASM_INST(0x65648062)  // fmad	z2.h, p0/m, z3.h, z4.h
    KAI_ASM_INST(0x65428020)  // fmul	z0.h, p0/m, z0.h, z1.h
    KAI_ASM_INST(0x654d8040)  // fdiv	z0.h, p0/m, z0.h, z2.h
    KAI_ASM_INST(0xe4a34020)  // st1h	{z0.h}, p0, [x1, x3, lsl #1]
    KAI_ASM_INST(0x8b060063)  // add	x3, x3, x6
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffda8)  // b.hi	50 <MlasSveTanhFP16KernelImpl+0x50>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveTanhFP16KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveErfFP16KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveErfFP16KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveErfFP16KernelImpl)
    KAI_ASM_INST(0xb4000b22)  // cbz	x2, 164 <MlasSveErfFP16KernelImpl+0x164>
    KAI_ASM_INST(0x5286a7e6)  // mov	w6, #0x353f
    KAI_ASM_INST(0x52868045)  // mov	w5, #0x3402
    KAI_ASM_INST(0x056038d7)  // mov	z23.h, w6
    KAI_ASM_INST(0x056038b6)  // mov	z22.h, w5
    KAI_ASM_INST(0x12896c66)  // mov	w6, #0xffffb49c
    KAI_ASM_INST(0x5287b485)  // mov	w5, #0x3da4
    KAI_ASM_INST(0x056038d5)  // mov	z21.h, w6
    KAI_ASM_INST(0x056038b4)  // mov	z20.h, w5
    KAI_ASM_INST(0x128849c6)  // mov	w6, #0xffffbdb1
    KAI_ASM_INST(0x52878705)  // mov	w5, #0x3c38
    KAI_ASM_INST(0x056038d3)  // mov	z19.h, w6
    KAI_ASM_INST(0x056038b2)  // mov	z18.h, w5
    KAI_ASM_INST(0x12893846)  // mov	w6, #0xffffb63d
    KAI_ASM_INST(0x5287aa45)  // mov	w5, #0x3d52
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x0460e3e4)  // cnth	x4
    KAI_ASM_INST(0x056038d1)  // mov	z17.h, w6
    KAI_ASM_INST(0x056038b0)  // mov	z16.h, w5
    KAI_ASM_INST(0x5284ebe6)  // mov	w6, #0x275f
    KAI_ASM_INST(0x52875925)  // mov	w5, #0x3ac9
    KAI_ASM_INST(0x056038c7)  // mov	z7.h, w6
    KAI_ASM_INST(0x056038a6)  // mov	z6.h, w5
    KAI_ASM_INST(0x5287ab46)  // mov	w6, #0x3d5a
    KAI_ASM_INST(0x5286ef25)  // mov	w5, #0x3779
    KAI_ASM_INST(0x056038c5)  // mov	z5.h, w6
    KAI_ASM_INST(0x056038a4)  // mov	z4.h, w5
    KAI_ASM_INST(0x2579de19)  // fmov	z25.h, #-1.000000000000000000e+00
    KAI_ASM_INST(0x2579ce03)  // fmov	z3.h, #1.000000000000000000e+00
    KAI_ASM_INST(0x2579c202)  // fmov	z2.h, #4.000000000000000000e+00
    KAI_ASM_INST(0x2579c318)  // fmov	z24.h, #6.000000000000000000e+00
    KAI_ASM_INST(0xfc1f0fe8)  // str	d8, [sp, #-16]!
    KAI_ASM_INST(0x25621c60)  // whilelo	p0.h, x3, x2
    KAI_ASM_INST(0xa4a34000)  // ld1h	{z0.h}, p0/z, [x0, x3, lsl #1]
    KAI_ASM_INST(0x0420bc1e)  // movprfx	z30, z0
    KAI_ASM_INST(0x045ca01e)  // fabs	z30.h, p0/m, z0.h
    KAI_ASM_INST(0x65512001)  // fcmlt	p1.h, p0/z, z0.h, #0.0
    KAI_ASM_INST(0x0420bfc1)  // movprfx	z1, z30
    KAI_ASM_INST(0x65478041)  // fmin	z1.h, p0/m, z1.h, z2.h
    KAI_ASM_INST(0x0420bc7d)  // movprfx	z29, z3
    KAI_ASM_INST(0x656102fd)  // fmla	z29.h, p0/m, z23.h, z1.h
    KAI_ASM_INST(0x654e33a0)  // frecpe	z0.h, z29.h
    KAI_ASM_INST(0x65401ba8)  // frecps	z8.h, z29.h, z0.h
    KAI_ASM_INST(0x65428100)  // fmul	z0.h, p0/m, z0.h, z8.h
    KAI_ASM_INST(0x65401bbd)  // frecps	z29.h, z29.h, z0.h
    KAI_ASM_INST(0x65428021)  // fmul	z1.h, p0/m, z1.h, z1.h
    KAI_ASM_INST(0x654283a0)  // fmul	z0.h, p0/m, z0.h, z29.h
    KAI_ASM_INST(0x65478301)  // fmin	z1.h, p0/m, z1.h, z24.h
    KAI_ASM_INST(0x0420bc1d)  // movprfx	z29, z0
    KAI_ASM_INST(0x6542801d)  // fmul	z29.h, p0/m, z29.h, z0.h
    KAI_ASM_INST(0x0420bc3b)  // movprfx	z27, z1
    KAI_ASM_INST(0x6542803b)  // fmul	z27.h, p0/m, z27.h, z1.h
    KAI_ASM_INST(0x0420bec8)  // movprfx	z8, z22
    KAI_ASM_INST(0x65428008)  // fmul	z8.h, p0/m, z8.h, z0.h
    KAI_ASM_INST(0x657d02a8)  // fmla	z8.h, p0/m, z21.h, z29.h
    KAI_ASM_INST(0x0420bc7a)  // movprfx	z26, z3
    KAI_ASM_INST(0x0551779a)  // mov	z26.h, p1/m, #-17408
    KAI_ASM_INST(0x0420bfbf)  // movprfx	z31, z29
    KAI_ASM_INST(0x6542801f)  // fmul	z31.h, p0/m, z31.h, z0.h
    KAI_ASM_INST(0x0420be1c)  // movprfx	z28, z16
    KAI_ASM_INST(0x6561023c)  // fmla	z28.h, p0/m, z17.h, z1.h
    KAI_ASM_INST(0x0420bffd)  // movprfx	z29, z31
    KAI_ASM_INST(0x6542801d)  // fmul	z29.h, p0/m, z29.h, z0.h
    KAI_ASM_INST(0x657b00fc)  // fmla	z28.h, p0/m, z7.h, z27.h
    KAI_ASM_INST(0x6568829f)  // fmad	z31.h, p0/m, z20.h, z8.h
    KAI_ASM_INST(0x654283a0)  // fmul	z0.h, p0/m, z0.h, z29.h
    KAI_ASM_INST(0x656580c1)  // fmad	z1.h, p0/m, z6.h, z5.h
    KAI_ASM_INST(0x657f827d)  // fmad	z29.h, p0/m, z19.h, z31.h
    KAI_ASM_INST(0x657b0081)  // fmla	z1.h, p0/m, z4.h, z27.h
    KAI_ASM_INST(0x657d8240)  // fmad	z0.h, p0/m, z18.h, z29.h
    KAI_ASM_INST(0x654e303b)  // frecpe	z27.h, z1.h
    KAI_ASM_INST(0x655e4051)  // fcmgt	p1.h, p0/z, z2.h, z30.h
    KAI_ASM_INST(0x655b1828)  // frecps	z8.h, z1.h, z27.h
    KAI_ASM_INST(0x6542811b)  // fmul	z27.h, p0/m, z27.h, z8.h
    KAI_ASM_INST(0x655b1821)  // frecps	z1.h, z1.h, z27.h
    KAI_ASM_INST(0x6542803b)  // fmul	z27.h, p0/m, z27.h, z1.h
    KAI_ASM_INST(0x6542837c)  // fmul	z28.h, p0/m, z28.h, z27.h
    KAI_ASM_INST(0x65428380)  // fmul	z0.h, p0/m, z0.h, z28.h
    KAI_ASM_INST(0x655b8020)  // fsubr	z0.h, p0/m, z0.h, #1.0
    KAI_ASM_INST(0x65428340)  // fmul	z0.h, p0/m, z0.h, z26.h
    KAI_ASM_INST(0x655f8020)  // fmin	z0.h, p0/m, z0.h, #1.0
    KAI_ASM_INST(0x65468320)  // fmax	z0.h, p0/m, z0.h, z25.h
    KAI_ASM_INST(0x057ac400)  // sel	z0.h, p1, z0.h, z26.h
    KAI_ASM_INST(0xe4a34020)  // st1h	{z0.h}, p0, [x1, x3, lsl #1]
    KAI_ASM_INST(0x8b040063)  // add	x3, x3, x4
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fff948)  // b.hi	80 <MlasSveErfFP16KernelImpl+0x80>
    KAI_ASM_INST(0xfc4107e8)  // ldr	d8, [sp], #16
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveErfFP16KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveGeluTanhArgFP16KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveGeluTanhArgFP16KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveGeluTanhArgFP16KernelImpl)
    KAI_ASM_INST(0xb40002c2)  // cbz	x2, 58 <MlasSveGeluTanhArgFP16KernelImpl+0x58>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x0460e3e6)  // cnth	x6
    KAI_ASM_INST(0x52851225)  // mov	w5, #0x2891
    KAI_ASM_INST(0x52874c44)  // mov	w4, #0x3a62
    KAI_ASM_INST(0x056038a3)  // mov	z3.h, w5
    KAI_ASM_INST(0x05603882)  // mov	z2.h, w4
    KAI_ASM_INST(0x2579c285)  // fmov	z5.h, #5.000000000000000000e+00
    KAI_ASM_INST(0x2579d284)  // fmov	z4.h, #-5.000000000000000000e+00
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25621c60)  // whilelo	p0.h, x3, x2
    KAI_ASM_INST(0xa4a34000)  // ld1h	{z0.h}, p0/z, [x0, x3, lsl #1]
    KAI_ASM_INST(0x0420bc01)  // movprfx	z1, z0
    KAI_ASM_INST(0x65428001)  // fmul	z1.h, p0/m, z1.h, z0.h
    KAI_ASM_INST(0x65628061)  // fmad	z1.h, p0/m, z3.h, z2.h
    KAI_ASM_INST(0x65428020)  // fmul	z0.h, p0/m, z0.h, z1.h
    KAI_ASM_INST(0x654780a0)  // fmin	z0.h, p0/m, z0.h, z5.h
    KAI_ASM_INST(0x65468080)  // fmax	z0.h, p0/m, z0.h, z4.h
    KAI_ASM_INST(0xe4a34020)  // st1h	{z0.h}, p0, [x1, x3, lsl #1]
    KAI_ASM_INST(0x8b060063)  // add	x3, x3, x6
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54fffea8)  // b.hi	28 <MlasSveGeluTanhArgFP16KernelImpl+0x28>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveGeluTanhArgFP16KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveGeluScaleFP16KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveGeluScaleFP16KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveGeluScaleFP16KernelImpl)
    KAI_ASM_INST(0xb40001a2)  // cbz	x2, 34 <MlasSveGeluScaleFP16KernelImpl+0x34>
    KAI_ASM_INST(0xd2800003)  // mov	x3, #0x0
    KAI_ASM_INST(0x0460e3e5)  // cnth	x5
    KAI_ASM_INST(0x52873504)  // mov	w4, #0x39a8
    KAI_ASM_INST(0x05603881)  // mov	z1.h, w4
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25621c60)  // whilelo	p0.h, x3, x2
    KAI_ASM_INST(0xa4a34000)  // ld1h	{z0.h}, p0/z, [x0, x3, lsl #1]
    KAI_ASM_INST(0x65428020)  // fmul	z0.h, p0/m, z0.h, z1.h
    KAI_ASM_INST(0xe4a34020)  // st1h	{z0.h}, p0, [x1, x3, lsl #1]
    KAI_ASM_INST(0x8b050063)  // add	x3, x3, x5
    KAI_ASM_INST(0xeb03005f)  // cmp	x2, x3
    KAI_ASM_INST(0x54ffff48)  // b.hi	18 <MlasSveGeluScaleFP16KernelImpl+0x18>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveGeluScaleFP16KernelImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasSveGeluCombineFP16KernelImpl)
    KAI_ASM_FUNCTION_TYPE(MlasSveGeluCombineFP16KernelImpl)
KAI_ASM_FUNCTION_LABEL(MlasSveGeluCombineFP16KernelImpl)
    KAI_ASM_INST(0xb4000203)  // cbz	x3, 40 <MlasSveGeluCombineFP16KernelImpl+0x40>
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0x0460e3e5)  // cnth	x5
    KAI_ASM_INST(0x2579ce03)  // fmov	z3.h, #1.000000000000000000e+00
    KAI_ASM_INST(0x2579cc02)  // fmov	z2.h, #5.000000000000000000e-01
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25631c80)  // whilelo	p0.h, x4, x3
    KAI_ASM_INST(0xa4a44000)  // ld1h	{z0.h}, p0/z, [x0, x4, lsl #1]
    KAI_ASM_INST(0xa4a44021)  // ld1h	{z1.h}, p0/z, [x1, x4, lsl #1]
    KAI_ASM_INST(0x65408061)  // fadd	z1.h, p0/m, z1.h, z3.h
    KAI_ASM_INST(0x65428020)  // fmul	z0.h, p0/m, z0.h, z1.h
    KAI_ASM_INST(0x65428040)  // fmul	z0.h, p0/m, z0.h, z2.h
    KAI_ASM_INST(0xe4a44040)  // st1h	{z0.h}, p0, [x2, x4, lsl #1]
    KAI_ASM_INST(0x8b050084)  // add	x4, x4, x5
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54fffee8)  // b.hi	18 <MlasSveGeluCombineFP16KernelImpl+0x18>
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_FUNCTION_END(MlasSveGeluCombineFP16KernelImpl)

    KAI_ASM_END
