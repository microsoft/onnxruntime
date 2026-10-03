/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemv_sve_asm.S

Abstract:

    Portable machine-code variant of the SVE halfgemv_sve kernels, in the style
    of Arm's KleidiAI library: every instruction is emitted as a raw
    instruction word (GAS ".inst" / armasm64 "DCD" via KAI_ASM_INST), so the
    file assembles with toolchains that have no SVE support, and the bytes
    are identical on every platform.

    GENERATED FILE - DO NOT EDIT BY HAND.

    Generated from the SVE intrinsics translation unit(s) (sve/halfgemv_kernel_sve.cpp), which
    remain the reference implementation and regeneration source (script:
    sve/gen_sve_asm.py). The functions here export the same extern "C"
    symbols; the build links exactly one of the two implementations. The
    code is fully position-independent (the generator verifies there are no
    relocations, calls, or literal pools).

--*/

#include "kai_asm_macros.h"

    KAI_ASM_CODE(halfgemv_sve)


    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasHgemvNKernel_sve)
    KAI_ASM_FUNCTION_TYPE(MlasHgemvNKernel_sve)
KAI_ASM_FUNCTION_LABEL(MlasHgemvNKernel_sve)
    KAI_ASM_BTI_C
    KAI_ASM_INST(0xa9be7bfd)  // stp	x29, x30, [sp, #-32]!
    KAI_ASM_INST(0x1e23c011)  // fcvt	h17, s0
    KAI_ASM_INST(0x910003fd)  // mov	x29, sp
    KAI_ASM_INST(0xa90153f3)  // stp	x19, x20, [sp, #16]
    KAI_ASM_INST(0x12001cf4)  // and	w20, w7, #0xff
    KAI_ASM_INST(0xf1000c7f)  // cmp	x3, #0x3
    KAI_ASM_INST(0x540009a9)  // b.ls	14c <MlasHgemvNKernel_sve+0x14c>
    KAI_ASM_INST(0xd37ff8a9)  // lsl	x9, x5, #1
    KAI_ASM_INST(0xd37ff8d1)  // lsl	x17, x6, #1
    KAI_ASM_INST(0x8b09000b)  // add	x11, x0, x9
    KAI_ASM_INST(0x8b11004e)  // add	x14, x2, x17
    KAI_ASM_INST(0x8b09016a)  // add	x10, x11, x9
    KAI_ASM_INST(0xcb1103f1)  // neg	x17, x17
    KAI_ASM_INST(0x8b090149)  // add	x9, x10, x9
    KAI_ASM_INST(0xd37df0c7)  // lsl	x7, x6, #3
    KAI_ASM_INST(0xd37df0af)  // lsl	x15, x5, #3
    KAI_ASM_INST(0xaa0003ec)  // mov	x12, x0
    KAI_ASM_INST(0xd37ef4de)  // lsl	x30, x6, #2
    KAI_ASM_INST(0x12000293)  // and	w19, w20, #0x1
    KAI_ASM_INST(0xd2800090)  // mov	x16, #0x4
    KAI_ASM_INST(0x0460e3ed)  // cnth	x13
    KAI_ASM_INST(0x2518e3e1)  // ptrue	p1.b
    KAI_ASM_INST(0xb4000584)  // cbz	x4, 108 <MlasHgemvNKernel_sve+0x108>
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xd2800008)  // mov	x8, #0x0
    KAI_ASM_INST(0x2578c001)  // mov	z1.h, #0
    KAI_ASM_INST(0x04613022)  // mov	z2.d, z1.d
    KAI_ASM_INST(0x04613023)  // mov	z3.d, z1.d
    KAI_ASM_INST(0x04613024)  // mov	z4.d, z1.d
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x25641d00)  // whilelo	p0.h, x8, x4
    KAI_ASM_INST(0xa4a84020)  // ld1h	{z0.h}, p0/z, [x1, x8, lsl #1]
    KAI_ASM_INST(0xa4a84190)  // ld1h	{z16.h}, p0/z, [x12, x8, lsl #1]
    KAI_ASM_INST(0xa4a84167)  // ld1h	{z7.h}, p0/z, [x11, x8, lsl #1]
    KAI_ASM_INST(0xa4a84146)  // ld1h	{z6.h}, p0/z, [x10, x8, lsl #1]
    KAI_ASM_INST(0xa4a84125)  // ld1h	{z5.h}, p0/z, [x9, x8, lsl #1]
    KAI_ASM_INST(0x8b0d0108)  // add	x8, x8, x13
    KAI_ASM_INST(0x65600204)  // fmla	z4.h, p0/m, z16.h, z0.h
    KAI_ASM_INST(0x656000e3)  // fmla	z3.h, p0/m, z7.h, z0.h
    KAI_ASM_INST(0x656000c2)  // fmla	z2.h, p0/m, z6.h, z0.h
    KAI_ASM_INST(0x656000a1)  // fmla	z1.h, p0/m, z5.h, z0.h
    KAI_ASM_INST(0xeb08009f)  // cmp	x4, x8
    KAI_ASM_INST(0x54fffe88)  // b.hi	78 <MlasHgemvNKernel_sve+0x78>
    KAI_ASM_INST(0x65402484)  // faddv	h4, p1, z4.h
    KAI_ASM_INST(0x65402463)  // faddv	h3, p1, z3.h
    KAI_ASM_INST(0x65402442)  // faddv	h2, p1, z2.h
    KAI_ASM_INST(0x65402421)  // faddv	h1, p1, z1.h
    KAI_ASM_INST(0x1ef10884)  // fmul	h4, h4, h17
    KAI_ASM_INST(0x1ef10863)  // fmul	h3, h3, h17
    KAI_ASM_INST(0x1ef10842)  // fmul	h2, h2, h17
    KAI_ASM_INST(0x1ef10821)  // fmul	h1, h1, h17
    KAI_ASM_INST(0x34000293)  // cbz	w19, 11c <MlasHgemvNKernel_sve+0x11c>
    KAI_ASM_INST(0x7c3169c4)  // str	h4, [x14, x17]
    KAI_ASM_INST(0x7d0001c3)  // str	h3, [x14]
    KAI_ASM_INST(0x7c2679c2)  // str	h2, [x14, x6, lsl #1]
    KAI_ASM_INST(0x91001208)  // add	x8, x16, #0x4
    KAI_ASM_INST(0x7c3e69c1)  // str	h1, [x14, x30]
    KAI_ASM_INST(0x8b0f018c)  // add	x12, x12, x15
    KAI_ASM_INST(0x8b0701ce)  // add	x14, x14, x7
    KAI_ASM_INST(0x8b0f016b)  // add	x11, x11, x15
    KAI_ASM_INST(0x8b0f014a)  // add	x10, x10, x15
    KAI_ASM_INST(0x8b0f0129)  // add	x9, x9, x15
    KAI_ASM_INST(0xeb03011f)  // cmp	x8, x3
    KAI_ASM_INST(0x540002a8)  // b.hi	150 <MlasHgemvNKernel_sve+0x150>
    KAI_ASM_INST(0xaa0803f0)  // mov	x16, x8
    KAI_ASM_INST(0xb5fffae4)  // cbnz	x4, 60 <MlasHgemvNKernel_sve+0x60>
    KAI_ASM_INST(0x2578c001)  // mov	z1.h, #0
    KAI_ASM_INST(0x04613022)  // mov	z2.d, z1.d
    KAI_ASM_INST(0x04613023)  // mov	z3.d, z1.d
    KAI_ASM_INST(0x04613024)  // mov	z4.d, z1.d
    KAI_ASM_INST(0x17ffffe5)  // b	ac <MlasHgemvNKernel_sve+0xac>
    KAI_ASM_INST(0x7c7169c0)  // ldr	h0, [x14, x17]
    KAI_ASM_INST(0x1ee42800)  // fadd	h0, h0, h4
    KAI_ASM_INST(0x7c3169c0)  // str	h0, [x14, x17]
    KAI_ASM_INST(0x7d4001c0)  // ldr	h0, [x14]
    KAI_ASM_INST(0x1ee32800)  // fadd	h0, h0, h3
    KAI_ASM_INST(0x7d0001c0)  // str	h0, [x14]
    KAI_ASM_INST(0x7c6679c0)  // ldr	h0, [x14, x6, lsl #1]
    KAI_ASM_INST(0x1ee22800)  // fadd	h0, h0, h2
    KAI_ASM_INST(0x7c2679c0)  // str	h0, [x14, x6, lsl #1]
    KAI_ASM_INST(0x7c7e69c0)  // ldr	h0, [x14, x30]
    KAI_ASM_INST(0x1ee02821)  // fadd	h1, h1, h0
    KAI_ASM_INST(0x17ffffe5)  // b	dc <MlasHgemvNKernel_sve+0xdc>
    KAI_ASM_INST(0xd2800010)  // mov	x16, #0x0
    KAI_ASM_INST(0xeb10007f)  // cmp	x3, x16
    KAI_ASM_INST(0x540003a9)  // b.ls	1c8 <MlasHgemvNKernel_sve+0x1c8>
    KAI_ASM_INST(0xd37ff8a8)  // lsl	x8, x5, #1
    KAI_ASM_INST(0xd37ff8c6)  // lsl	x6, x6, #1
    KAI_ASM_INST(0x12000293)  // and	w19, w20, #0x1
    KAI_ASM_INST(0x0460e3e5)  // cnth	x5
    KAI_ASM_INST(0x2518e3e1)  // ptrue	p1.b
    KAI_ASM_INST(0x9b060a07)  // madd	x7, x16, x6, x2
    KAI_ASM_INST(0x9b100102)  // madd	x2, x8, x16, x0
    KAI_ASM_INST(0xb4000304)  // cbz	x4, 1d4 <MlasHgemvNKernel_sve+0x1d4>
    KAI_ASM_INST(0xd2800000)  // mov	x0, #0x0
    KAI_ASM_INST(0x2578c000)  // mov	z0.h, #0
    KAI_ASM_INST(0x25641c00)  // whilelo	p0.h, x0, x4
    KAI_ASM_INST(0xa4a04022)  // ld1h	{z2.h}, p0/z, [x1, x0, lsl #1]
    KAI_ASM_INST(0xa4a04041)  // ld1h	{z1.h}, p0/z, [x2, x0, lsl #1]
    KAI_ASM_INST(0x8b050000)  // add	x0, x0, x5
    KAI_ASM_INST(0x65620020)  // fmla	z0.h, p0/m, z1.h, z2.h
    KAI_ASM_INST(0xeb00009f)  // cmp	x4, x0
    KAI_ASM_INST(0x54ffff48)  // b.hi	180 <MlasHgemvNKernel_sve+0x180>
    KAI_ASM_INST(0x65402400)  // faddv	h0, p1, z0.h
    KAI_ASM_INST(0x1ef10800)  // fmul	h0, h0, h17
    KAI_ASM_INST(0x35000073)  // cbnz	w19, 1b0 <MlasHgemvNKernel_sve+0x1b0>
    KAI_ASM_INST(0x7d4000e1)  // ldr	h1, [x7]
    KAI_ASM_INST(0x1ee12800)  // fadd	h0, h0, h1
    KAI_ASM_INST(0x91000610)  // add	x16, x16, #0x1
    KAI_ASM_INST(0x7d0000e0)  // str	h0, [x7]
    KAI_ASM_INST(0x8b080042)  // add	x2, x2, x8
    KAI_ASM_INST(0x8b0600e7)  // add	x7, x7, x6
    KAI_ASM_INST(0xeb10007f)  // cmp	x3, x16
    KAI_ASM_INST(0x54fffd81)  // b.ne	174 <MlasHgemvNKernel_sve+0x174>
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xa8c27bfd)  // ldp	x29, x30, [sp], #32
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0x2578c000)  // mov	z0.h, #0
    KAI_ASM_INST(0x17fffff1)  // b	19c <MlasHgemvNKernel_sve+0x19c>
    KAI_ASM_FUNCTION_END(MlasHgemvNKernel_sve)

    KAI_ASM_END
