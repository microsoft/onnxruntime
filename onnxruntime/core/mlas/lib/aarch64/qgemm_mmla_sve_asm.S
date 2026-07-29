/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_mmla_sve_asm.S

Abstract:

    Portable machine-code variant of the SVE qgemm_mmla_sve kernels, in the style
    of Arm's KleidiAI library: every instruction is emitted as a raw
    instruction word (GAS ".inst" / armasm64 "DCD" via KAI_ASM_INST), so the
    file assembles with toolchains that have no SVE support, and the bytes
    are identical on every platform.

    GENERATED FILE - DO NOT EDIT BY HAND.

    Generated from the SVE intrinsics translation unit (sve/qgemm_mmla_sve_impl.cpp), which
    remains the reference implementation and regeneration source (script:
    sve/gen_sve_asm.py). The functions here export the same extern "C"
    symbols; the build links exactly one of the two implementations. The
    code is fully position-independent (the generator verifies there are no
    relocations, calls, or literal pools).

--*/

#include "kai_asm_macros.h"

    KAI_ASM_CODE(qgemm_mmla_sve)


    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasGemmS8S8KernelSmmlaSveImpl)
    KAI_ASM_FUNCTION_TYPE(MlasGemmS8S8KernelSmmlaSveImpl)
KAI_ASM_FUNCTION_LABEL(MlasGemmS8S8KernelSmmlaSveImpl)
    KAI_ASM_INST(0xa9ac7bfd)  // stp	x29, x30, [sp, #-320]!
    KAI_ASM_INST(0xaa0003ed)  // mov	x13, x0
    KAI_ASM_INST(0xd1003089)  // sub	x9, x4, #0xc
    KAI_ASM_INST(0x910003fd)  // mov	x29, sp
    KAI_ASM_INST(0x394543e0)  // ldrb	w0, [sp, #336]
    KAI_ASM_INST(0xaa0203f0)  // mov	x16, x2
    KAI_ASM_INST(0xa90153f3)  // stp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xaa0503f1)  // mov	x17, x5
    KAI_ASM_INST(0xaa0703f3)  // mov	x19, x7
    KAI_ASM_INST(0xa90573fb)  // stp	x27, x28, [sp, #80]
    KAI_ASM_INST(0xd37ae46e)  // lsl	x14, x3, #6
    KAI_ASM_INST(0xb900fbe0)  // str	w0, [sp, #248]
    KAI_ASM_INST(0xf940a7fc)  // ldr	x28, [sp, #328]
    KAI_ASM_INST(0xf1000d3f)  // cmp	x9, #0x3
    KAI_ASM_INST(0x54008c29)  // b.ls	11bc <MlasGemmS8S8KernelSmmlaSveImpl+0x11bc>
    KAI_ASM_INST(0xa9025bf5)  // stp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa90363f7)  // stp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9046bf9)  // stp	x25, x26, [sp, #64]
    KAI_ASM_INST(0x6d0627e8)  // stp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d072fea)  // stp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d0837ec)  // stp	d12, d13, [sp, #128]
    KAI_ASM_INST(0xf1001c9f)  // cmp	x4, #0x7
    KAI_ASM_INST(0x540015c8)  // b.hi	310 <MlasGemmS8S8KernelSmmlaSveImpl+0x310>
    KAI_ASM_INST(0xf1000c9f)  // cmp	x4, #0x3
    KAI_ASM_INST(0x5400eb88)  // b.hi	1dd0 <MlasGemmS8S8KernelSmmlaSveImpl+0x1dd0>
    KAI_ASM_INST(0xf100049f)  // cmp	x4, #0x1
    KAI_ASM_INST(0x5400d828)  // b.hi	1b6c <MlasGemmS8S8KernelSmmlaSveImpl+0x1b6c>
    KAI_ASM_INST(0xd2800020)  // mov	x0, #0x1
    KAI_ASM_INST(0xaa0003e2)  // mov	x2, x0
    KAI_ASM_INST(0xd2800108)  // mov	x8, #0x8
    KAI_ASM_INST(0x2518e101)  // ptrue	p1.b, vl8
    KAI_ASM_INST(0xb4001371)  // cbz	x17, 2e8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2e8>
    KAI_ASM_INST(0xf100045f)  // cmp	x2, #0x1
    KAI_ASM_INST(0xb940fbe2)  // ldr	w2, [sp, #248]
    KAI_ASM_INST(0xf940a3e4)  // ldr	x4, [sp, #320]
    KAI_ASM_INST(0x2598e082)  // ptrue	p2.s, vl4
    KAI_ASM_INST(0xd37ef4d9)  // lsl	x25, x6, #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xd37df0d4)  // lsl	x20, x6, #3
    KAI_ASM_INST(0x1200005e)  // and	w30, w2, #0x1
    KAI_ASM_INST(0xd2800058)  // mov	x24, #0x2
    KAI_ASM_INST(0xd280009a)  // mov	x26, #0x4
    KAI_ASM_INST(0xaa1c03e9)  // mov	x9, x28
    KAI_ASM_INST(0x9a9a0318)  // csel	x24, x24, x26, eq
    KAI_ASM_INST(0xaa1103e7)  // mov	x7, x17
    KAI_ASM_INST(0x91004026)  // add	x6, x1, #0x10
    KAI_ASM_INST(0x91008025)  // add	x5, x1, #0x20
    KAI_ASM_INST(0xaa1103ef)  // mov	x15, x17
    KAI_ASM_INST(0x9100c022)  // add	x2, x1, #0x30
    KAI_ASM_INST(0x91004339)  // add	x25, x25, #0x10
    KAI_ASM_INST(0xd2800017)  // mov	x23, #0x0
    KAI_ASM_INST(0xd280000c)  // mov	x12, #0x0
    KAI_ASM_INST(0xa90be7ee)  // stp	x14, x25, [sp, #184]
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xf10020ff)  // cmp	x7, #0x8
    KAI_ASM_INST(0xd280010a)  // mov	x10, #0x8
    KAI_ASM_INST(0x9a8a90f9)  // csel	x25, x7, x10, ls
    KAI_ASM_INST(0xf94063ea)  // ldr	x10, [sp, #192]
    KAI_ASM_INST(0xf100133f)  // cmp	x25, #0x4
    KAI_ASM_INST(0xaa1303f1)  // mov	x17, x19
    KAI_ASM_INST(0xaa1003eb)  // mov	x11, x16
    KAI_ASM_INST(0xaa1903ee)  // mov	x14, x25
    KAI_ASM_INST(0x8b100155)  // add	x21, x10, x16
    KAI_ASM_INST(0xd280000a)  // mov	x10, #0x0
    KAI_ASM_INST(0xa90ac3ec)  // stp	x12, x16, [sp, #168]
    KAI_ASM_INST(0x9a9a9336)  // csel	x22, x25, x26, ls
    KAI_ASM_INST(0x25b61fe3)  // whilelo	p3.s, xzr, x22
    KAI_ASM_INST(0x91000559)  // add	x25, x10, #0x1
    KAI_ASM_INST(0xb9400230)  // ldr	w16, [x17]
    KAI_ASM_INST(0x5280000c)  // mov	w12, #0x0
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54000049)  // b.ls	124 <MlasGemmS8S8KernelSmmlaSveImpl+0x124>
    KAI_ASM_INST(0xb940062c)  // ldr	w12, [x17, #4]
    KAI_ASM_INST(0x4e040e04)  // dup	v4.4s, w16
    KAI_ASM_INST(0x4e141d84)  // mov	v4.s[2], w12
    KAI_ASM_INST(0x4e1c1d84)  // mov	v4.s[3], w12
    KAI_ASM_INST(0xb400d43c)  // cbz	x28, 1bb4 <MlasGemmS8S8KernelSmmlaSveImpl+0x1bb4>
    KAI_ASM_INST(0x29404936)  // ldp	w22, w18, [x9]
    KAI_ASM_INST(0x2d40848d)  // ldp	s13, s1, [x4, #4]
    KAI_ASM_INST(0x2d41808b)  // ldp	s11, s0, [x4, #12]
    KAI_ASM_INST(0x1b167e1b)  // mul	w27, w16, w22
    KAI_ASM_INST(0x2d413126)  // ldp	s6, s12, [x9, #8]
    KAI_ASM_INST(0x1b127e10)  // mul	w16, w16, w18
    KAI_ASM_INST(0x2d422925)  // ldp	s5, s10, [x9, #16]
    KAI_ASM_INST(0x1e270362)  // fmov	s2, w27
    KAI_ASM_INST(0x2d432123)  // ldp	s3, s8, [x9, #24]
    KAI_ASM_INST(0x4e0c1e02)  // mov	v2.s[1], w16
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd401489)  // ldr	s9, [x4, #20]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x1b0c7ed6)  // mul	w22, w22, w12
    KAI_ASM_INST(0x4e0404c6)  // dup	v6.4s, v6.s[0]
    KAI_ASM_INST(0x4d40c887)  // ld1r	{v7.4s}, [x4]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0x1b0c7e52)  // mul	w18, w18, w12
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c05a7)  // mov	v7.s[1], v13.s[0]
    KAI_ASM_INST(0x6e0c0561)  // mov	v1.s[1], v11.s[0]
    KAI_ASM_INST(0x6e0c0520)  // mov	v0.s[1], v9.s[0]
    KAI_ASM_INST(0x4e141ec2)  // mov	v2.s[2], w22
    KAI_ASM_INST(0x6e0c0586)  // mov	v6.s[1], v12.s[0]
    KAI_ASM_INST(0x6e0c0545)  // mov	v5.s[1], v10.s[0]
    KAI_ASM_INST(0x6e0c0503)  // mov	v3.s[1], v8.s[0]
    KAI_ASM_INST(0x6e1c05a7)  // mov	v7.s[3], v13.s[0]
    KAI_ASM_INST(0x6e1c0561)  // mov	v1.s[3], v11.s[0]
    KAI_ASM_INST(0x6e1c0520)  // mov	v0.s[3], v9.s[0]
    KAI_ASM_INST(0x4e1c1e42)  // mov	v2.s[3], w18
    KAI_ASM_INST(0x6e1c0586)  // mov	v6.s[3], v12.s[0]
    KAI_ASM_INST(0x6e1c0545)  // mov	v5.s[3], v10.s[0]
    KAI_ASM_INST(0x6e1c0503)  // mov	v3.s[3], v8.s[0]
    KAI_ASM_INST(0x4ea78442)  // add	v2.4s, v2.4s, v7.4s
    KAI_ASM_INST(0x4ea494c1)  // mla	v1.4s, v6.4s, v4.4s
    KAI_ASM_INST(0x4ea494a0)  // mla	v0.4s, v5.4s, v4.4s
    KAI_ASM_INST(0x4ea39c84)  // mul	v4.4s, v4.4s, v3.4s
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0x9104c3ec)  // add	x12, sp, #0x130
    KAI_ASM_INST(0xa540a982)  // ld1w	{z2.s}, p2/z, [x12]
    KAI_ASM_INST(0xbd401c85)  // ldr	s5, [x4, #28]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a983)  // ld1w	{z3.s}, p2/z, [x12]
    KAI_ASM_INST(0xd37df150)  // lsl	x16, x10, #3
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a981)  // ld1w	{z1.s}, p2/z, [x12]
    KAI_ASM_INST(0xbd401880)  // ldr	s0, [x4, #24]
    KAI_ASM_INST(0x9104c3ec)  // add	x12, sp, #0x130
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c04a0)  // mov	v0.s[1], v5.s[0]
    KAI_ASM_INST(0x6e1c04a0)  // mov	v0.s[3], v5.s[0]
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a984)  // ld1w	{z4.s}, p2/z, [x12]
    KAI_ASM_INST(0xb4000263)  // cbz	x3, 258 <MlasGemmS8S8KernelSmmlaSveImpl+0x258>
    KAI_ASM_INST(0x8b1001b0)  // add	x16, x13, x16
    KAI_ASM_INST(0xaa1703ec)  // mov	x12, x23
    KAI_ASM_INST(0xd2800012)  // mov	x18, #0x0
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xa40c4026)  // ld1b	{z6.b}, p0/z, [x1, x12]
    KAI_ASM_INST(0xa40c40c5)  // ld1b	{z5.b}, p0/z, [x6, x12]
    KAI_ASM_INST(0x91000652)  // add	x18, x18, #0x1
    KAI_ASM_INST(0xa400a600)  // ld1b	{z0.b}, p1/z, [x16]
    KAI_ASM_INST(0x8b080210)  // add	x16, x16, x8
    KAI_ASM_INST(0x45069802)  // smmla	z2.s, z0.b, z6.b
    KAI_ASM_INST(0x45059803)  // smmla	z3.s, z0.b, z5.b
    KAI_ASM_INST(0xa40c40a6)  // ld1b	{z6.b}, p0/z, [x5, x12]
    KAI_ASM_INST(0xa40c4045)  // ld1b	{z5.b}, p0/z, [x2, x12]
    KAI_ASM_INST(0x45069801)  // smmla	z1.s, z0.b, z6.b
    KAI_ASM_INST(0x9101018c)  // add	x12, x12, #0x40
    KAI_ASM_INST(0x45059804)  // smmla	z4.s, z0.b, z5.b
    KAI_ASM_INST(0xeb12007f)  // cmp	x3, x18
    KAI_ASM_INST(0x54fffe61)  // b.ne	220 <MlasGemmS8S8KernelSmmlaSveImpl+0x220>
    KAI_ASM_INST(0x05e46c20)  // uzp2	z0.d, z1.d, z4.d
    KAI_ASM_INST(0x05e46821)  // uzp1	z1.d, z1.d, z4.d
    KAI_ASM_INST(0x04603005)  // mov	z5.d, z0.d
    KAI_ASM_INST(0x05e36844)  // uzp1	z4.d, z2.d, z3.d
    KAI_ASM_INST(0x05e36c42)  // uzp2	z2.d, z2.d, z3.d
    KAI_ASM_INST(0x3500c8be)  // cbnz	w30, 1b80 <MlasGemmS8S8KernelSmmlaSveImpl+0x1b80>
    KAI_ASM_INST(0xa540ad63)  // ld1w	{z3.s}, p3/z, [x11]
    KAI_ASM_INST(0x04a40063)  // add	z3.s, z3.s, z4.s
    KAI_ASM_INST(0xe540ed63)  // st1w	{z3.s}, p3, [x11]
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0x5400cc49)  // b.ls	1c08 <MlasGemmS8S8KernelSmmlaSveImpl+0x1c08>
    KAI_ASM_INST(0xd10011cc)  // sub	x12, x14, #0x4
    KAI_ASM_INST(0x25ac1fe4)  // whilelo	p4.s, xzr, x12
    KAI_ASM_INST(0x9100416c)  // add	x12, x11, #0x10
    KAI_ASM_INST(0xa540b183)  // ld1w	{z3.s}, p4/z, [x12]
    KAI_ASM_INST(0x04a10061)  // add	z1.s, z3.s, z1.s
    KAI_ASM_INST(0xe540f181)  // st1w	{z1.s}, p4, [x12]
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x5400cc28)  // b.hi	1c24 <MlasGemmS8S8KernelSmmlaSveImpl+0x1c24>
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x9100094a)  // add	x10, x10, #0x2
    KAI_ASM_INST(0x91002231)  // add	x17, x17, #0x8
    KAI_ASM_INST(0x8b1402b5)  // add	x21, x21, x20
    KAI_ASM_INST(0x8b14016b)  // add	x11, x11, x20
    KAI_ASM_INST(0xeb18015f)  // cmp	x10, x24
    KAI_ASM_INST(0x54fff281)  // b.ne	10c <MlasGemmS8S8KernelSmmlaSveImpl+0x10c>
    KAI_ASM_INST(0xa94ac3ec)  // ldp	x12, x16, [sp, #168]
    KAI_ASM_INST(0xd10020e7)  // sub	x7, x7, #0x8
    KAI_ASM_INST(0xf9405fea)  // ldr	x10, [sp, #184]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0x91008129)  // add	x9, x9, #0x20
    KAI_ASM_INST(0x8b0a02f7)  // add	x23, x23, x10
    KAI_ASM_INST(0x9100218c)  // add	x12, x12, #0x8
    KAI_ASM_INST(0x91008210)  // add	x16, x16, #0x20
    KAI_ASM_INST(0xeb0c01ff)  // cmp	x15, x12
    KAI_ASM_INST(0x54ffefa8)  // b.hi	d8 <MlasGemmS8S8KernelSmmlaSveImpl+0xd8>
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0x6d093fee)  // stp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xd2800009)  // mov	x9, #0x0
    KAI_ASM_INST(0xf1002cbf)  // cmp	x5, #0xb
    KAI_ASM_INST(0x54003909)  // b.ls	a3c <MlasGemmS8S8KernelSmmlaSveImpl+0xa3c>
    KAI_ASM_INST(0x8b0604c7)  // add	x7, x6, x6, lsl #1
    KAI_ASM_INST(0xcb060bea)  // neg	x10, x6, lsl #2
    KAI_ASM_INST(0x910004c5)  // add	x5, x6, #0x1
    KAI_ASM_INST(0xd37ef4c2)  // lsl	x2, x6, #2
    KAI_ASM_INST(0xd37df0e9)  // lsl	x9, x7, #3
    KAI_ASM_INST(0x8b070a08)  // add	x8, x16, x7, lsl #2
    KAI_ASM_INST(0x8b09021e)  // add	x30, x16, x9
    KAI_ASM_INST(0xd37df0c0)  // lsl	x0, x6, #3
    KAI_ASM_INST(0xcb060fc4)  // sub	x4, x30, x6, lsl #3
    KAI_ASM_INST(0xd37ceca5)  // lsl	x5, x5, #4
    KAI_ASM_INST(0x8b0a0096)  // add	x22, x4, x10
    KAI_ASM_INST(0x8b060058)  // add	x24, x2, x6
    KAI_ASM_INST(0x8b0a02db)  // add	x27, x22, x10
    KAI_ASM_INST(0x91004108)  // add	x8, x8, #0x10
    KAI_ASM_INST(0x8b0a036c)  // add	x12, x27, x10
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x910040aa)  // add	x10, x5, #0x10
    KAI_ASM_INST(0x91004007)  // add	x7, x0, #0x10
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0x8b050205)  // add	x5, x16, x5
    KAI_ASM_INST(0xa90d17e8)  // stp	x8, x5, [sp, #208]
    KAI_ASM_INST(0x91008188)  // add	x8, x12, #0x20
    KAI_ASM_INST(0x9100412b)  // add	x11, x9, #0x10
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0x91008129)  // add	x9, x9, #0x20
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0x8b070202)  // add	x2, x16, x7
    KAI_ASM_INST(0xf90087e8)  // str	x8, [sp, #264]
    KAI_ASM_INST(0x910082c8)  // add	x8, x22, #0x20
    KAI_ASM_INST(0xf90067e2)  // str	x2, [sp, #200]
    KAI_ASM_INST(0x91004202)  // add	x2, x16, #0x10
    KAI_ASM_INST(0xa9110be8)  // stp	x8, x2, [sp, #272]
    KAI_ASM_INST(0x8b090208)  // add	x8, x16, x9
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xcb060017)  // sub	x23, x0, x6
    KAI_ASM_INST(0xb940fbe2)  // ldr	w2, [sp, #248]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005fe8)  // str	x8, [sp, #184]
    KAI_ASM_INST(0x8b0a0208)  // add	x8, x16, x10
    KAI_ASM_INST(0xd2800009)  // mov	x9, #0x0
    KAI_ASM_INST(0x8b0b020b)  // add	x11, x16, x11
    KAI_ASM_INST(0x8b000200)  // add	x0, x16, x0
    KAI_ASM_INST(0xf9005be8)  // str	x8, [sp, #176]
    KAI_ASM_INST(0x12000042)  // and	w2, w2, #0x1
    KAI_ASM_INST(0xf940a3e8)  // ldr	x8, [sp, #320]
    KAI_ASM_INST(0xd37ef718)  // lsl	x24, x24, #2
    KAI_ASM_INST(0xd37ef6f7)  // lsl	x23, x23, #2
    KAI_ASM_INST(0xaa1c03ef)  // mov	x15, x28
    KAI_ASM_INST(0x910041b5)  // add	x21, x13, #0x10
    KAI_ASM_INST(0x910081b4)  // add	x20, x13, #0x20
    KAI_ASM_INST(0xaa0b03e7)  // mov	x7, x11
    KAI_ASM_INST(0xaa0403ea)  // mov	x10, x4
    KAI_ASM_INST(0xf90073e9)  // str	x9, [sp, #224]
    KAI_ASM_INST(0xaa0003e9)  // mov	x9, x0
    KAI_ASM_INST(0x9100c1b2)  // add	x18, x13, #0x30
    KAI_ASM_INST(0xb900abe2)  // str	w2, [sp, #168]
    KAI_ASM_INST(0x91008202)  // add	x2, x16, #0x20
    KAI_ASM_INST(0xf90077f2)  // str	x18, [sp, #232]
    KAI_ASM_INST(0xf90083f1)  // str	x17, [sp, #256]
    KAI_ASM_INST(0xf90093e2)  // str	x2, [sp, #288]
    KAI_ASM_INST(0xf90097e6)  // str	x6, [sp, #296]
    KAI_ASM_INST(0xf900a7fc)  // str	x28, [sp, #328]
    KAI_ASM_INST(0xf94073f2)  // ldr	x18, [sp, #224]
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0x91001245)  // add	x5, x18, #0x4
    KAI_ASM_INST(0x91002a42)  // add	x2, x18, #0xa
    KAI_ASM_INST(0x91000a46)  // add	x6, x18, #0x2
    KAI_ASM_INST(0x9100224b)  // add	x11, x18, #0x8
    KAI_ASM_INST(0xd343fcba)  // lsr	x26, x5, #3
    KAI_ASM_INST(0xd37d08a5)  // ubfiz	x5, x5, #3, #3
    KAI_ASM_INST(0x4d40ca68)  // ld1r	{v8.4s}, [x19]
    KAI_ASM_INST(0xd343fcdc)  // lsr	x28, x6, #3
    KAI_ASM_INST(0xd37d08c6)  // ubfiz	x6, x6, #3, #3
    KAI_ASM_INST(0x91001a44)  // add	x4, x18, #0x6
    KAI_ASM_INST(0x9b0e1745)  // madd	x5, x26, x14, x5
    KAI_ASM_INST(0xd37d0a51)  // ubfiz	x17, x18, #3, #3
    KAI_ASM_INST(0xf9007be5)  // str	x5, [sp, #240]
    KAI_ASM_INST(0xd343fc45)  // lsr	x5, x2, #3
    KAI_ASM_INST(0xd37d0842)  // ubfiz	x2, x2, #3, #3
    KAI_ASM_INST(0xaa1203e0)  // mov	x0, x18
    KAI_ASM_INST(0xd343fd6b)  // lsr	x11, x11, #3
    KAI_ASM_INST(0xd343fc99)  // lsr	x25, x4, #3
    KAI_ASM_INST(0x6e140408)  // mov	v8.s[2], v0.s[0]
    KAI_ASM_INST(0x9b0e1b9c)  // madd	x28, x28, x14, x6
    KAI_ASM_INST(0x9b0e08a2)  // madd	x2, x5, x14, x2
    KAI_ASM_INST(0xd37d0884)  // ubfiz	x4, x4, #3, #3
    KAI_ASM_INST(0xf940a7e5)  // ldr	x5, [sp, #328]
    KAI_ASM_INST(0xd343fc06)  // lsr	x6, x0, #3
    KAI_ASM_INST(0x91003252)  // add	x18, x18, #0xc
    KAI_ASM_INST(0x9b0e456b)  // madd	x11, x11, x14, x17
    KAI_ASM_INST(0xf90073f2)  // str	x18, [sp, #224]
    KAI_ASM_INST(0x9b0e1324)  // madd	x4, x25, x14, x4
    KAI_ASM_INST(0x6e1c0408)  // mov	v8.s[3], v0.s[0]
    KAI_ASM_INST(0x9b0e44c6)  // madd	x6, x6, x14, x17
    KAI_ASM_INST(0xaa0b03f2)  // mov	x18, x11
    KAI_ASM_INST(0xb4006385)  // cbz	x5, 110c <MlasGemmS8S8KernelSmmlaSveImpl+0x110c>
    KAI_ASM_INST(0x2d409110)  // ldp	s16, s4, [x8, #4]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d41950e)  // ldp	s14, s5, [x8, #12]
    KAI_ASM_INST(0x2d428d0c)  // ldp	s12, s3, [x8, #20]
    KAI_ASM_INST(0x2d40e5f1)  // ldp	s17, s25, [x15, #4]
    KAI_ASM_INST(0x2d43890a)  // ldp	s10, s2, [x8, #28]
    KAI_ASM_INST(0x4d40c9e6)  // ld1r	{v6.4s}, [x15]
    KAI_ASM_INST(0x4d40c901)  // ld1r	{v1.4s}, [x8]
    KAI_ASM_INST(0x2d43d9eb)  // ldp	s11, s22, [x15, #28]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0xbd402507)  // ldr	s7, [x8, #36]
    KAI_ASM_INST(0x2d41ddef)  // ldp	s15, s23, [x15, #12]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0xbd402de0)  // ldr	s0, [x15, #44]
    KAI_ASM_INST(0x2d42cded)  // ldp	s13, s19, [x15, #20]
    KAI_ASM_INST(0x6e0c0601)  // mov	v1.s[1], v16.s[0]
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c0626)  // mov	v6.s[1], v17.s[0]
    KAI_ASM_INST(0x2d44c9e9)  // ldp	s9, s18, [x15, #36]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c05c4)  // mov	v4.s[1], v14.s[0]
    KAI_ASM_INST(0x4e040739)  // dup	v25.4s, v25.s[0]
    KAI_ASM_INST(0x4e0406d6)  // dup	v22.4s, v22.s[0]
    KAI_ASM_INST(0x6e0c0585)  // mov	v5.s[1], v12.s[0]
    KAI_ASM_INST(0x4e0406f7)  // dup	v23.4s, v23.s[0]
    KAI_ASM_INST(0x6e1c0601)  // mov	v1.s[3], v16.s[0]
    KAI_ASM_INST(0x6e0c0543)  // mov	v3.s[1], v10.s[0]
    KAI_ASM_INST(0x4e040673)  // dup	v19.4s, v19.s[0]
    KAI_ASM_INST(0x6e1c0626)  // mov	v6.s[3], v17.s[0]
    KAI_ASM_INST(0x6e0c04e2)  // mov	v2.s[1], v7.s[0]
    KAI_ASM_INST(0x4e040652)  // dup	v18.4s, v18.s[0]
    KAI_ASM_INST(0x6e0c0536)  // mov	v22.s[1], v9.s[0]
    KAI_ASM_INST(0x6e1c05c4)  // mov	v4.s[3], v14.s[0]
    KAI_ASM_INST(0x6e0c05f9)  // mov	v25.s[1], v15.s[0]
    KAI_ASM_INST(0x6e1c0585)  // mov	v5.s[3], v12.s[0]
    KAI_ASM_INST(0x6e0c05b7)  // mov	v23.s[1], v13.s[0]
    KAI_ASM_INST(0x4ea11c3a)  // mov	v26.16b, v1.16b
    KAI_ASM_INST(0x6e1c0543)  // mov	v3.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea894c1)  // mla	v1.4s, v6.4s, v8.4s
    KAI_ASM_INST(0x6e0c0573)  // mov	v19.s[1], v11.s[0]
    KAI_ASM_INST(0x6e1c04e2)  // mov	v2.s[3], v7.s[0]
    KAI_ASM_INST(0x6e0c0412)  // mov	v18.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0536)  // mov	v22.s[3], v9.s[0]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b5)  // ld1w	{z21.s}, p1/z, [x5]
    KAI_ASM_INST(0x6e1c05f9)  // mov	v25.s[3], v15.s[0]
    KAI_ASM_INST(0x4ea41c89)  // mov	v9.16b, v4.16b
    KAI_ASM_INST(0x6e1c05b7)  // mov	v23.s[3], v13.s[0]
    KAI_ASM_INST(0x4ea51ca7)  // mov	v7.16b, v5.16b
    KAI_ASM_INST(0x6e1c0573)  // mov	v19.s[3], v11.s[0]
    KAI_ASM_INST(0x4ea31c61)  // mov	v1.16b, v3.16b
    KAI_ASM_INST(0x6e1c0412)  // mov	v18.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea21c40)  // mov	v0.16b, v2.16b
    KAI_ASM_INST(0x4ea89729)  // mla	v9.4s, v25.4s, v8.4s
    KAI_ASM_INST(0x4ea896e7)  // mla	v7.4s, v23.4s, v8.4s
    KAI_ASM_INST(0x4ea89661)  // mla	v1.4s, v19.4s, v8.4s
    KAI_ASM_INST(0x4ea896c0)  // mla	v0.4s, v22.4s, v8.4s
    KAI_ASM_INST(0x4eb29d08)  // mul	v8.4s, v8.4s, v18.4s
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a4b4)  // ld1w	{z20.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4ad)  // ld1w	{z13.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4ac)  // ld1w	{z12.s}, p1/z, [x5]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d452500)  // ldp	s0, s9, [x8, #40]
    KAI_ASM_INST(0x3cc08267)  // ldur	q7, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x4eba1f4b)  // mov	v11.16b, v26.16b
    KAI_ASM_INST(0x4e8738e1)  // zip1	v1.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x4ea41c98)  // mov	v24.16b, v4.16b
    KAI_ASM_INST(0x6e0c0520)  // mov	v0.s[1], v9.s[0]
    KAI_ASM_INST(0x2d433a71)  // ldp	s17, s14, [x19, #24]
    KAI_ASM_INST(0x4ea6942b)  // mla	v11.4s, v1.4s, v6.4s
    KAI_ASM_INST(0x6e1c0520)  // mov	v0.s[3], v9.s[0]
    KAI_ASM_INST(0x4ea51caa)  // mov	v10.16b, v5.16b
    KAI_ASM_INST(0x4eb99438)  // mla	v24.4s, v1.4s, v25.4s
    KAI_ASM_INST(0x4ea31c7c)  // mov	v28.16b, v3.16b
    KAI_ASM_INST(0x4ea08508)  // add	v8.4s, v8.4s, v0.4s
    KAI_ASM_INST(0x4ea21c49)  // mov	v9.16b, v2.16b
    KAI_ASM_INST(0x4ea01c0f)  // mov	v15.16b, v0.16b
    KAI_ASM_INST(0x4e8778e7)  // zip2	v7.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x4eb7942a)  // mla	v10.4s, v1.4s, v23.4s
    KAI_ASM_INST(0x3d804fe8)  // str	q8, [sp, #304]
    KAI_ASM_INST(0xa540a4bf)  // ld1w	{z31.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eba1f5b)  // mov	v27.16b, v26.16b
    KAI_ASM_INST(0x3d804feb)  // str	q11, [sp, #304]
    KAI_ASM_INST(0xa540a4ab)  // ld1w	{z11.s}, p1/z, [x5]
    KAI_ASM_INST(0x4e040631)  // dup	v17.4s, v17.s[0]
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4be)  // ld1w	{z30.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb3943c)  // mla	v28.4s, v1.4s, v19.4s
    KAI_ASM_INST(0x3d804fea)  // str	q10, [sp, #304]
    KAI_ASM_INST(0xa540a4aa)  // ld1w	{z10.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea41c98)  // mov	v24.16b, v4.16b
    KAI_ASM_INST(0x4eb69429)  // mla	v9.4s, v1.4s, v22.4s
    KAI_ASM_INST(0x4ea1964f)  // mla	v15.4s, v18.4s, v1.4s
    KAI_ASM_INST(0x3d804ffc)  // str	q28, [sp, #304]
    KAI_ASM_INST(0xa540a4bd)  // ld1w	{z29.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea51ca1)  // mov	v1.16b, v5.16b
    KAI_ASM_INST(0x4ea694fb)  // mla	v27.4s, v7.4s, v6.4s
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4a9)  // ld1w	{z9.s}, p1/z, [x5]
    KAI_ASM_INST(0x6e1405d1)  // mov	v17.s[2], v14.s[0]
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a4bc)  // ld1w	{z28.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb994f8)  // mla	v24.4s, v7.4s, v25.4s
    KAI_ASM_INST(0x3d804ffb)  // str	q27, [sp, #304]
    KAI_ASM_INST(0xa540a4a8)  // ld1w	{z8.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb794e1)  // mla	v1.4s, v7.4s, v23.4s
    KAI_ASM_INST(0x4ea31c6f)  // mov	v15.16b, v3.16b
    KAI_ASM_INST(0x6e1c05d1)  // mov	v17.s[3], v14.s[0]
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4bb)  // ld1w	{z27.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea21c4e)  // mov	v14.16b, v2.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0x4ea01c18)  // mov	v24.16b, v0.16b
    KAI_ASM_INST(0x4eba1f41)  // mov	v1.16b, v26.16b
    KAI_ASM_INST(0x4eb394ef)  // mla	v15.4s, v7.4s, v19.4s
    KAI_ASM_INST(0x4eb694ee)  // mla	v14.4s, v7.4s, v22.4s
    KAI_ASM_INST(0x4eb294f8)  // mla	v24.4s, v7.4s, v18.4s
    KAI_ASM_INST(0xa540a4a7)  // ld1w	{z7.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb99624)  // mla	v4.4s, v17.4s, v25.4s
    KAI_ASM_INST(0x4ea69621)  // mla	v1.4s, v17.4s, v6.4s
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a4ba)  // ld1w	{z26.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb79625)  // mla	v5.4s, v17.4s, v23.4s
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a4a6)  // ld1w	{z6.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb39623)  // mla	v3.4s, v17.4s, v19.4s
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4b9)  // ld1w	{z25.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb69622)  // mla	v2.4s, v17.4s, v22.4s
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b3)  // ld1w	{z19.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb29620)  // mla	v0.4s, v17.4s, v18.4s
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a4b7)  // ld1w	{z23.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a4b2)  // ld1w	{z18.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a4b6)  // ld1w	{z22.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a4b1)  // ld1w	{z17.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4b8)  // ld1w	{z24.s}, p1/z, [x5]
    KAI_ASM_INST(0xb4000623)  // cbz	x3, 7b4 <MlasGemmS8S8KernelSmmlaSveImpl+0x7b4>
    KAI_ASM_INST(0x8b06003a)  // add	x26, x1, x6
    KAI_ASM_INST(0x8b120026)  // add	x6, x1, x18
    KAI_ASM_INST(0xa94e97f2)  // ldp	x18, x5, [sp, #232]
    KAI_ASM_INST(0x8b1c0039)  // add	x25, x1, x28
    KAI_ASM_INST(0x8b04002b)  // add	x11, x1, x4
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0x8b050031)  // add	x17, x1, x5
    KAI_ASM_INST(0x8b020025)  // add	x5, x1, x2
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xa4024344)  // ld1b	{z4.b}, p0/z, [x26, x2]
    KAI_ASM_INST(0xa40241a3)  // ld1b	{z3.b}, p0/z, [x13, x2]
    KAI_ASM_INST(0xa40242a2)  // ld1b	{z2.b}, p0/z, [x21, x2]
    KAI_ASM_INST(0xa4024281)  // ld1b	{z1.b}, p0/z, [x20, x2]
    KAI_ASM_INST(0xa4024240)  // ld1b	{z0.b}, p0/z, [x18, x2]
    KAI_ASM_INST(0x45049875)  // smmla	z21.s, z3.b, z4.b
    KAI_ASM_INST(0x4504984b)  // smmla	z11.s, z2.b, z4.b
    KAI_ASM_INST(0x45049828)  // smmla	z8.s, z1.b, z4.b
    KAI_ASM_INST(0x45049813)  // smmla	z19.s, z0.b, z4.b
    KAI_ASM_INST(0xa4024224)  // ld1b	{z4.b}, p0/z, [x17, x2]
    KAI_ASM_INST(0xa4024325)  // ld1b	{z5.b}, p0/z, [x25, x2]
    KAI_ASM_INST(0x45049874)  // smmla	z20.s, z3.b, z4.b
    KAI_ASM_INST(0x4504984a)  // smmla	z10.s, z2.b, z4.b
    KAI_ASM_INST(0x45049827)  // smmla	z7.s, z1.b, z4.b
    KAI_ASM_INST(0x45049812)  // smmla	z18.s, z0.b, z4.b
    KAI_ASM_INST(0x91000484)  // add	x4, x4, #0x1
    KAI_ASM_INST(0xa40240c4)  // ld1b	{z4.b}, p0/z, [x6, x2]
    KAI_ASM_INST(0x45059870)  // smmla	z16.s, z3.b, z5.b
    KAI_ASM_INST(0x4505985e)  // smmla	z30.s, z2.b, z5.b
    KAI_ASM_INST(0x4505983b)  // smmla	z27.s, z1.b, z5.b
    KAI_ASM_INST(0x45059817)  // smmla	z23.s, z0.b, z5.b
    KAI_ASM_INST(0x4504986c)  // smmla	z12.s, z3.b, z4.b
    KAI_ASM_INST(0xa4024165)  // ld1b	{z5.b}, p0/z, [x11, x2]
    KAI_ASM_INST(0x45049849)  // smmla	z9.s, z2.b, z4.b
    KAI_ASM_INST(0x45049826)  // smmla	z6.s, z1.b, z4.b
    KAI_ASM_INST(0x45049811)  // smmla	z17.s, z0.b, z4.b
    KAI_ASM_INST(0x4505986d)  // smmla	z13.s, z3.b, z5.b
    KAI_ASM_INST(0xa40240a4)  // ld1b	{z4.b}, p0/z, [x5, x2]
    KAI_ASM_INST(0x4505985d)  // smmla	z29.s, z2.b, z5.b
    KAI_ASM_INST(0x91010042)  // add	x2, x2, #0x40
    KAI_ASM_INST(0x4505983a)  // smmla	z26.s, z1.b, z5.b
    KAI_ASM_INST(0x45059816)  // smmla	z22.s, z0.b, z5.b
    KAI_ASM_INST(0x4504987f)  // smmla	z31.s, z3.b, z4.b
    KAI_ASM_INST(0x4504985c)  // smmla	z28.s, z2.b, z4.b
    KAI_ASM_INST(0x45049839)  // smmla	z25.s, z1.b, z4.b
    KAI_ASM_INST(0x45049818)  // smmla	z24.s, z0.b, z4.b
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54fffb61)  // b.ne	718 <MlasGemmS8S8KernelSmmlaSveImpl+0x718>
    KAI_ASM_INST(0xf90077f2)  // str	x18, [sp, #232]
    KAI_ASM_INST(0xb940abe2)  // ldr	w2, [sp, #168]
    KAI_ASM_INST(0x05f06aa0)  // uzp1	z0.d, z21.d, z16.d
    KAI_ASM_INST(0x05ed6a81)  // uzp1	z1.d, z20.d, z13.d
    KAI_ASM_INST(0x05ff6982)  // uzp1	z2.d, z12.d, z31.d
    KAI_ASM_INST(0x05f06eb0)  // uzp2	z16.d, z21.d, z16.d
    KAI_ASM_INST(0x05ed6e8d)  // uzp2	z13.d, z20.d, z13.d
    KAI_ASM_INST(0x05ff6d8c)  // uzp2	z12.d, z12.d, z31.d
    KAI_ASM_INST(0x350043a2)  // cbnz	w2, 1044 <MlasGemmS8S8KernelSmmlaSveImpl+0x1044>
    KAI_ASM_INST(0xd37ef402)  // lsl	x2, x0, #2
    KAI_ASM_INST(0x05fe696e)  // uzp1	z14.d, z11.d, z30.d
    KAI_ASM_INST(0x8b020205)  // add	x5, x16, x2
    KAI_ASM_INST(0xa540a4a3)  // ld1w	{z3.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00060)  // add	z0.s, z3.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa95117e6)  // ldp	x6, x5, [sp, #272]
    KAI_ASM_INST(0x05fd694f)  // uzp1	z15.d, z10.d, z29.d
    KAI_ASM_INST(0x05fe6d6b)  // uzp2	z11.d, z11.d, z30.d
    KAI_ASM_INST(0x05fd6d4a)  // uzp2	z10.d, z10.d, z29.d
    KAI_ASM_INST(0x05fc6934)  // uzp1	z20.d, z9.d, z28.d
    KAI_ASM_INST(0x05fc6d29)  // uzp2	z9.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6905)  // uzp1	z5.d, z8.d, z27.d
    KAI_ASM_INST(0x05fa68e4)  // uzp1	z4.d, z7.d, z26.d
    KAI_ASM_INST(0x8b180204)  // add	x4, x16, x24
    KAI_ASM_INST(0x05fb6d08)  // uzp2	z8.d, z8.d, z27.d
    KAI_ASM_INST(0x05fa6ce7)  // uzp2	z7.d, z7.d, z26.d
    KAI_ASM_INST(0x05f968c3)  // uzp1	z3.d, z6.d, z25.d
    KAI_ASM_INST(0x8b0200a5)  // add	x5, x5, x2
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xf94093e5)  // ldr	x5, [sp, #288]
    KAI_ASM_INST(0xa540a580)  // ld1w	{z0.s}, p1/z, [x12]
    KAI_ASM_INST(0x04b00000)  // add	z0.s, z0.s, z16.s
    KAI_ASM_INST(0xe540e580)  // st1w	{z0.s}, p1, [x12]
    KAI_ASM_INST(0x05f96cc6)  // uzp2	z6.d, z6.d, z25.d
    KAI_ASM_INST(0x8b0200a5)  // add	x5, x5, x2
    KAI_ASM_INST(0x8b0200c2)  // add	x2, x6, x2
    KAI_ASM_INST(0xf94063e6)  // ldr	x6, [sp, #192]
    KAI_ASM_INST(0xa540a4c0)  // ld1w	{z0.s}, p1/z, [x6]
    KAI_ASM_INST(0x04ad0000)  // add	z0.s, z0.s, z13.s
    KAI_ASM_INST(0xe540e4c0)  // st1w	{z0.s}, p1, [x6]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xf94087e5)  // ldr	x5, [sp, #264]
    KAI_ASM_INST(0xa54044a0)  // ld1w	{z0.s}, p1/z, [x5, x0, lsl #2]
    KAI_ASM_INST(0x04ac0000)  // add	z0.s, z0.s, z12.s
    KAI_ASM_INST(0xe54044a0)  // st1w	{z0.s}, p1, [x5, x0, lsl #2]
    KAI_ASM_INST(0xa540a760)  // ld1w	{z0.s}, p1/z, [x27]
    KAI_ASM_INST(0xf94067e5)  // ldr	x5, [sp, #200]
    KAI_ASM_INST(0x04a001ce)  // add	z14.s, z14.s, z0.s
    KAI_ASM_INST(0xe540e76e)  // st1w	{z14.s}, p1, [x27]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a001ef)  // add	z15.s, z15.s, z0.s
    KAI_ASM_INST(0xe540e4af)  // st1w	{z15.s}, p1, [x5]
    KAI_ASM_INST(0xa540a6c0)  // ld1w	{z0.s}, p1/z, [x22]
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0xe540e6cb)  // st1w	{z11.s}, p1, [x22]
    KAI_ASM_INST(0xf9406be5)  // ldr	x5, [sp, #208]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00140)  // add	z0.s, z10.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa540a520)  // ld1w	{z0.s}, p1/z, [x9]
    KAI_ASM_INST(0x04a00280)  // add	z0.s, z20.s, z0.s
    KAI_ASM_INST(0xe540e520)  // st1w	{z0.s}, p1, [x9]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a00120)  // add	z0.s, z9.s, z0.s
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a540)  // ld1w	{z0.s}, p1/z, [x10]
    KAI_ASM_INST(0x04a000a0)  // add	z0.s, z5.s, z0.s
    KAI_ASM_INST(0xf9406fe5)  // ldr	x5, [sp, #216]
    KAI_ASM_INST(0xe540e540)  // st1w	{z0.s}, p1, [x10]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00080)  // add	z0.s, z4.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04a00100)  // add	z0.s, z8.s, z0.s
    KAI_ASM_INST(0x91004082)  // add	x2, x4, #0x10
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a000e0)  // add	z0.s, z7.s, z0.s
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a00060)  // add	z0.s, z3.s, z0.s
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04a000c6)  // add	z6.s, z6.s, z0.s
    KAI_ASM_INST(0x91008302)  // add	x2, x24, #0x20
    KAI_ASM_INST(0x05f86e21)  // uzp2	z1.d, z17.d, z24.d
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0xe540e446)  // st1w	{z6.s}, p1, [x2]
    KAI_ASM_INST(0xb940abe2)  // ldr	w2, [sp, #168]
    KAI_ASM_INST(0x05f76a63)  // uzp1	z3.d, z19.d, z23.d
    KAI_ASM_INST(0x05f66a42)  // uzp1	z2.d, z18.d, z22.d
    KAI_ASM_INST(0x05f86a31)  // uzp1	z17.d, z17.d, z24.d
    KAI_ASM_INST(0x04613020)  // mov	z0.d, z1.d
    KAI_ASM_INST(0x05f76e73)  // uzp2	z19.d, z19.d, z23.d
    KAI_ASM_INST(0x05f66e52)  // uzp2	z18.d, z18.d, z22.d
    KAI_ASM_INST(0x35003d22)  // cbnz	w2, 10e8 <MlasGemmS8S8KernelSmmlaSveImpl+0x10e8>
    KAI_ASM_INST(0xa540a7c0)  // ld1w	{z0.s}, p1/z, [x30]
    KAI_ASM_INST(0x04a30000)  // add	z0.s, z0.s, z3.s
    KAI_ASM_INST(0xe540e7c0)  // st1w	{z0.s}, p1, [x30]
    KAI_ASM_INST(0xa540a4e0)  // ld1w	{z0.s}, p1/z, [x7]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0x8b170202)  // add	x2, x16, x23
    KAI_ASM_INST(0xe540e4e0)  // st1w	{z0.s}, p1, [x7]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04b30000)  // add	z0.s, z0.s, z19.s
    KAI_ASM_INST(0x91004044)  // add	x4, x2, #0x10
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04b20000)  // add	z0.s, z0.s, z18.s
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xf9405fe4)  // ldr	x4, [sp, #184]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04b10000)  // add	z0.s, z0.s, z17.s
    KAI_ASM_INST(0x91008042)  // add	x2, x2, #0x20
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0x910082e2)  // add	x2, x23, #0x20
    KAI_ASM_INST(0x91006000)  // add	x0, x0, #0x18
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0x9100c3de)  // add	x30, x30, #0x30
    KAI_ASM_INST(0x9100c14a)  // add	x10, x10, #0x30
    KAI_ASM_INST(0x9100c2d6)  // add	x22, x22, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9006fe2)  // str	x2, [sp, #216]
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0x9100c37b)  // add	x27, x27, #0x30
    KAI_ASM_INST(0x9100c18c)  // add	x12, x12, #0x30
    KAI_ASM_INST(0x9100c108)  // add	x8, x8, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9006be2)  // str	x2, [sp, #208]
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0x9100c1ef)  // add	x15, x15, #0x30
    KAI_ASM_INST(0x9100c0e7)  // add	x7, x7, #0x30
    KAI_ASM_INST(0x9100c129)  // add	x9, x9, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf90067e2)  // str	x2, [sp, #200]
    KAI_ASM_INST(0xf94063e2)  // ldr	x2, [sp, #192]
    KAI_ASM_INST(0x9100c318)  // add	x24, x24, #0x30
    KAI_ASM_INST(0x9100c2f7)  // add	x23, x23, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0xf9405fe2)  // ldr	x2, [sp, #184]
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9005fe2)  // str	x2, [sp, #184]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9005be2)  // str	x2, [sp, #176]
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0xeb00005f)  // cmp	x2, x0
    KAI_ASM_INST(0x54ffcf82)  // b.cs	418 <MlasGemmS8S8KernelSmmlaSveImpl+0x418>
    KAI_ASM_INST(0xf94073e9)  // ldr	x9, [sp, #224]
    KAI_ASM_INST(0xaa0203f1)  // mov	x17, x2
    KAI_ASM_INST(0xf94097e6)  // ldr	x6, [sp, #296]
    KAI_ASM_INST(0xf940a7fc)  // ldr	x28, [sp, #328]
    KAI_ASM_INST(0xeb09023f)  // cmp	x17, x9
    KAI_ASM_INST(0x54002ea9)  // b.ls	1014 <MlasGemmS8S8KernelSmmlaSveImpl+0x1014>
    KAI_ASM_INST(0xd37df0c7)  // lsl	x7, x6, #3
    KAI_ASM_INST(0xd37ff8c2)  // lsl	x2, x6, #1
    KAI_ASM_INST(0xcb0600e0)  // sub	x0, x7, x6
    KAI_ASM_INST(0x8b060045)  // add	x5, x2, x6
    KAI_ASM_INST(0x8b090017)  // add	x23, x0, x9
    KAI_ASM_INST(0x91001000)  // add	x0, x0, #0x4
    KAI_ASM_INST(0xcb0602f2)  // sub	x18, x23, x6
    KAI_ASM_INST(0x8b090000)  // add	x0, x0, x9
    KAI_ASM_INST(0xcb06024f)  // sub	x15, x18, x6
    KAI_ASM_INST(0x910010ab)  // add	x11, x5, #0x4
    KAI_ASM_INST(0x8b0608c8)  // add	x8, x6, x6, lsl #2
    KAI_ASM_INST(0x8b09016b)  // add	x11, x11, x9
    KAI_ASM_INST(0x8b000a00)  // add	x0, x16, x0, lsl #2
    KAI_ASM_INST(0x8b050525)  // add	x5, x9, x5, lsl #1
    KAI_ASM_INST(0x91001108)  // add	x8, x8, #0x4
    KAI_ASM_INST(0x910010a5)  // add	x5, x5, #0x4
    KAI_ASM_INST(0xf9005fe0)  // str	x0, [sp, #184]
    KAI_ASM_INST(0x8b0f0a00)  // add	x0, x16, x15, lsl #2
    KAI_ASM_INST(0x910010c6)  // add	x6, x6, #0x4
    KAI_ASM_INST(0x8b090108)  // add	x8, x8, x9
    KAI_ASM_INST(0xf9005be0)  // str	x0, [sp, #176]
    KAI_ASM_INST(0x8b0b0a00)  // add	x0, x16, x11, lsl #2
    KAI_ASM_INST(0xcb0201e4)  // sub	x4, x15, x2
    KAI_ASM_INST(0x8b0900ca)  // add	x10, x6, x9
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0x8b050a00)  // add	x0, x16, x5, lsl #2
    KAI_ASM_INST(0xcb020082)  // sub	x2, x4, x2
    KAI_ASM_INST(0xf90063e0)  // str	x0, [sp, #192]
    KAI_ASM_INST(0x8b080a00)  // add	x0, x16, x8, lsl #2
    KAI_ASM_INST(0x8b07020c)  // add	x12, x16, x7
    KAI_ASM_INST(0xb940fbf4)  // ldr	w20, [sp, #248]
    KAI_ASM_INST(0xd1000626)  // sub	x6, x17, #0x1
    KAI_ASM_INST(0x8b040a19)  // add	x25, x16, x4, lsl #2
    KAI_ASM_INST(0xf9006fe0)  // str	x0, [sp, #216]
    KAI_ASM_INST(0xf940a3e4)  // ldr	x4, [sp, #320]
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0x8b0a0a00)  // add	x0, x16, x10, lsl #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0x8b170a17)  // add	x23, x16, x23, lsl #2
    KAI_ASM_INST(0x8b120a1e)  // add	x30, x16, x18, lsl #2
    KAI_ASM_INST(0x8b020a1a)  // add	x26, x16, x2, lsl #2
    KAI_ASM_INST(0xcb090236)  // sub	x22, x17, x9
    KAI_ASM_INST(0x91001135)  // add	x21, x9, #0x4
    KAI_ASM_INST(0x927ff8cf)  // and	x15, x6, #0xfffffffffffffffe
    KAI_ASM_INST(0x910081ab)  // add	x11, x13, #0x20
    KAI_ASM_INST(0x9100c1aa)  // add	x10, x13, #0x30
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0x8b070180)  // add	x0, x12, x7
    KAI_ASM_INST(0xf9007bec)  // str	x12, [sp, #240]
    KAI_ASM_INST(0x910041ac)  // add	x12, x13, #0x10
    KAI_ASM_INST(0x12000294)  // and	w20, w20, #0x1
    KAI_ASM_INST(0xb900abf4)  // str	w20, [sp, #168]
    KAI_ASM_INST(0xa90e47f0)  // stp	x16, x17, [sp, #224]
    KAI_ASM_INST(0xf9007fe0)  // str	x0, [sp, #248]
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xf10022df)  // cmp	x22, #0x8
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0x4d40ca63)  // ld1r	{v3.4s}, [x19]
    KAI_ASM_INST(0x91000922)  // add	x2, x9, #0x2
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0x9a8092c7)  // csel	x7, x22, x0, ls
    KAI_ASM_INST(0xeb0901ff)  // cmp	x15, x9
    KAI_ASM_INST(0x91001920)  // add	x0, x9, #0x6
    KAI_ASM_INST(0x6e140403)  // mov	v3.s[2], v0.s[0]
    KAI_ASM_INST(0x9a8991e6)  // csel	x6, x15, x9, ls
    KAI_ASM_INST(0xeb0f005f)  // cmp	x2, x15
    KAI_ASM_INST(0x9a8f9042)  // csel	x2, x2, x15, ls
    KAI_ASM_INST(0xeb1501ff)  // cmp	x15, x21
    KAI_ASM_INST(0x9a9591e5)  // csel	x5, x15, x21, ls
    KAI_ASM_INST(0xeb0f001f)  // cmp	x0, x15
    KAI_ASM_INST(0x9a8f9000)  // csel	x0, x0, x15, ls
    KAI_ASM_INST(0xd37d0858)  // ubfiz	x24, x2, #3, #3
    KAI_ASM_INST(0xd343fc5b)  // lsr	x27, x2, #3
    KAI_ASM_INST(0xd343fcd1)  // lsr	x17, x6, #3
    KAI_ASM_INST(0xd37d0812)  // ubfiz	x18, x0, #3, #3
    KAI_ASM_INST(0xd343fc14)  // lsr	x20, x0, #3
    KAI_ASM_INST(0x6e1c0403)  // mov	v3.s[3], v0.s[0]
    KAI_ASM_INST(0xd37d08d0)  // ubfiz	x16, x6, #3, #3
    KAI_ASM_INST(0x9b0e637b)  // madd	x27, x27, x14, x24
    KAI_ASM_INST(0xd343fca8)  // lsr	x8, x5, #3
    KAI_ASM_INST(0xd37d08b8)  // ubfiz	x24, x5, #3, #3
    KAI_ASM_INST(0x9b0e4a94)  // madd	x20, x20, x14, x18
    KAI_ASM_INST(0xd37ef4d2)  // lsl	x18, x6, #2
    KAI_ASM_INST(0x9b0e4230)  // madd	x16, x17, x14, x16
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x91001252)  // add	x18, x18, #0x4
    KAI_ASM_INST(0x9b0e6111)  // madd	x17, x8, x14, x24
    KAI_ASM_INST(0xb4008abc)  // cbz	x28, 1cf4 <MlasGemmS8S8KernelSmmlaSveImpl+0x1cf4>
    KAI_ASM_INST(0xbc667b87)  // ldr	s7, [x28, x6, lsl #2]
    KAI_ASM_INST(0xbc726b80)  // ldr	s0, [x28, x18]
    KAI_ASM_INST(0x4e0404e7)  // dup	v7.4s, v7.s[0]
    KAI_ASM_INST(0x6e0c0407)  // mov	v7.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0407)  // mov	v7.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea39ce0)  // mul	v0.4s, v7.4s, v3.4s
    KAI_ASM_INST(0xbc667884)  // ldr	s4, [x4, x6, lsl #2]
    KAI_ASM_INST(0xd37ef446)  // lsl	x6, x2, #2
    KAI_ASM_INST(0xbc726881)  // ldr	s1, [x4, x18]
    KAI_ASM_INST(0x9104c3e8)  // add	x8, sp, #0x130
    KAI_ASM_INST(0x910010c6)  // add	x6, x6, #0x4
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x6e0c0424)  // mov	v4.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0424)  // mov	v4.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a50b)  // ld1w	{z11.s}, p1/z, [x8]
    KAI_ASM_INST(0xb40088dc)  // cbz	x28, 1d00 <MlasGemmS8S8KernelSmmlaSveImpl+0x1d00>
    KAI_ASM_INST(0xbc627b80)  // ldr	s0, [x28, x2, lsl #2]
    KAI_ASM_INST(0xbc666b81)  // ldr	s1, [x28, x6]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c08)  // mov	v8.16b, v0.16b
    KAI_ASM_INST(0x4ea69c01)  // mul	v1.4s, v0.4s, v6.4s
    KAI_ASM_INST(0xbc627880)  // ldr	s0, [x4, x2, lsl #2]
    KAI_ASM_INST(0xd37ef4a2)  // lsl	x2, x5, #2
    KAI_ASM_INST(0xbc666882)  // ldr	s2, [x4, x6]
    KAI_ASM_INST(0x91001042)  // add	x2, x2, #0x4
    KAI_ASM_INST(0x9104c3e6)  // add	x6, sp, #0x130
    KAI_ASM_INST(0x4e040403)  // dup	v3.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0443)  // mov	v3.s[1], v2.s[0]
    KAI_ASM_INST(0x6e1c0443)  // mov	v3.s[3], v2.s[0]
    KAI_ASM_INST(0x4ea38421)  // add	v1.4s, v1.4s, v3.4s
    KAI_ASM_INST(0x4ea31c69)  // mov	v9.16b, v3.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4d1)  // ld1w	{z17.s}, p1/z, [x6]
    KAI_ASM_INST(0xb400859c)  // cbz	x28, 1ce8 <MlasGemmS8S8KernelSmmlaSveImpl+0x1ce8>
    KAI_ASM_INST(0xbc657b80)  // ldr	s0, [x28, x5, lsl #2]
    KAI_ASM_INST(0xbc626b81)  // ldr	s1, [x28, x2]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c0e)  // mov	v14.16b, v0.16b
    KAI_ASM_INST(0x4ea69c01)  // mul	v1.4s, v0.4s, v6.4s
    KAI_ASM_INST(0xbc657882)  // ldr	s2, [x4, x5, lsl #2]
    KAI_ASM_INST(0xd37ef405)  // lsl	x5, x0, #2
    KAI_ASM_INST(0x4f00042d)  // movi	v13.4s, #0x1
    KAI_ASM_INST(0xbc626880)  // ldr	s0, [x4, x2]
    KAI_ASM_INST(0x910010a2)  // add	x2, x5, #0x4
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c0402)  // mov	v2.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0402)  // mov	v2.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea28421)  // add	v1.4s, v1.4s, v2.4s
    KAI_ASM_INST(0x4ea21c4f)  // mov	v15.16b, v2.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0xb400011c)  // cbz	x28, cac <MlasGemmS8S8KernelSmmlaSveImpl+0xcac>
    KAI_ASM_INST(0xbc607b80)  // ldr	s0, [x28, x0, lsl #2]
    KAI_ASM_INST(0xbc626b81)  // ldr	s1, [x28, x2]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c0d)  // mov	v13.16b, v0.16b
    KAI_ASM_INST(0x4ea09cc6)  // mul	v6.4s, v6.4s, v0.4s
    KAI_ASM_INST(0xbc607880)  // ldr	s0, [x4, x0, lsl #2]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0xbc626883)  // ldr	s3, [x4, x2]
    KAI_ASM_INST(0x3cc08262)  // ldur	q2, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x2d432a61)  // ldp	s1, s10, [x19, #24]
    KAI_ASM_INST(0x4e823845)  // zip1	v5.4s, v2.4s, v2.4s
    KAI_ASM_INST(0x6e0c0460)  // mov	v0.s[1], v3.s[0]
    KAI_ASM_INST(0x4ea41c93)  // mov	v19.16b, v4.16b
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0x4ea91d32)  // mov	v18.16b, v9.16b
    KAI_ASM_INST(0x6e1c0460)  // mov	v0.s[3], v3.s[0]
    KAI_ASM_INST(0x4ea794b3)  // mla	v19.4s, v5.4s, v7.4s
    KAI_ASM_INST(0x4eaf1df6)  // mov	v22.16b, v15.16b
    KAI_ASM_INST(0x4e827842)  // zip2	v2.4s, v2.4s, v2.4s
    KAI_ASM_INST(0x4ea084c3)  // add	v3.4s, v6.4s, v0.4s
    KAI_ASM_INST(0x4ea01c14)  // mov	v20.16b, v0.16b
    KAI_ASM_INST(0x4ea41c97)  // mov	v23.16b, v4.16b
    KAI_ASM_INST(0x6e140541)  // mov	v1.s[2], v10.s[0]
    KAI_ASM_INST(0x4ea59512)  // mla	v18.4s, v8.4s, v5.4s
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a40c)  // ld1w	{z12.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea91d35)  // mov	v21.16b, v9.16b
    KAI_ASM_INST(0x3d804ff3)  // str	q19, [sp, #304]
    KAI_ASM_INST(0xa540a406)  // ld1w	{z6.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea595b4)  // mla	v20.4s, v13.4s, v5.4s
    KAI_ASM_INST(0x3d804ff2)  // str	q18, [sp, #304]
    KAI_ASM_INST(0xa540a412)  // ld1w	{z18.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea595d6)  // mla	v22.4s, v14.4s, v5.4s
    KAI_ASM_INST(0x4eaf1df3)  // mov	v19.16b, v15.16b
    KAI_ASM_INST(0x4ea01c05)  // mov	v5.16b, v0.16b
    KAI_ASM_INST(0x6e1c0541)  // mov	v1.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea79457)  // mla	v23.4s, v2.4s, v7.4s
    KAI_ASM_INST(0x3d804ff6)  // str	q22, [sp, #304]
    KAI_ASM_INST(0xa540a416)  // ld1w	{z22.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea89455)  // mla	v21.4s, v2.4s, v8.4s
    KAI_ASM_INST(0x3d804ff4)  // str	q20, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea91d23)  // mov	v3.16b, v9.16b
    KAI_ASM_INST(0x3d804ff7)  // str	q23, [sp, #304]
    KAI_ASM_INST(0xa540a40a)  // ld1w	{z10.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ead9445)  // mla	v5.4s, v2.4s, v13.4s
    KAI_ASM_INST(0x3d804ff5)  // str	q21, [sp, #304]
    KAI_ASM_INST(0x4eae9453)  // mla	v19.4s, v2.4s, v14.4s
    KAI_ASM_INST(0x4eaf1de2)  // mov	v2.16b, v15.16b
    KAI_ASM_INST(0x4ea79424)  // mla	v4.4s, v1.4s, v7.4s
    KAI_ASM_INST(0x4ea89423)  // mla	v3.4s, v1.4s, v8.4s
    KAI_ASM_INST(0xa540a408)  // ld1w	{z8.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ead9420)  // mla	v0.4s, v1.4s, v13.4s
    KAI_ASM_INST(0x3d804ff3)  // str	q19, [sp, #304]
    KAI_ASM_INST(0xa540a415)  // ld1w	{z21.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eae9422)  // mla	v2.4s, v1.4s, v14.4s
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a409)  // ld1w	{z9.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a407)  // ld1w	{z7.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a40d)  // ld1w	{z13.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a417)  // ld1w	{z23.s}, p1/z, [x0]
    KAI_ASM_INST(0xb4000463)  // cbz	x3, e30 <MlasGemmS8S8KernelSmmlaSveImpl+0xe30>
    KAI_ASM_INST(0x8b100030)  // add	x16, x1, x16
    KAI_ASM_INST(0x8b1b0028)  // add	x8, x1, x27
    KAI_ASM_INST(0x8b110026)  // add	x6, x1, x17
    KAI_ASM_INST(0x8b140025)  // add	x5, x1, x20
    KAI_ASM_INST(0xd2800000)  // mov	x0, #0x0
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xa40041a0)  // ld1b	{z0.b}, p0/z, [x13, x0]
    KAI_ASM_INST(0xa4004204)  // ld1b	{z4.b}, p0/z, [x16, x0]
    KAI_ASM_INST(0xa4004103)  // ld1b	{z3.b}, p0/z, [x8, x0]
    KAI_ASM_INST(0xa40040c2)  // ld1b	{z2.b}, p0/z, [x6, x0]
    KAI_ASM_INST(0xa40040a1)  // ld1b	{z1.b}, p0/z, [x5, x0]
    KAI_ASM_INST(0x4504980b)  // smmla	z11.s, z0.b, z4.b
    KAI_ASM_INST(0x45039811)  // smmla	z17.s, z0.b, z3.b
    KAI_ASM_INST(0x45029810)  // smmla	z16.s, z0.b, z2.b
    KAI_ASM_INST(0x4501980c)  // smmla	z12.s, z0.b, z1.b
    KAI_ASM_INST(0x91000442)  // add	x2, x2, #0x1
    KAI_ASM_INST(0xa4004160)  // ld1b	{z0.b}, p0/z, [x11, x0]
    KAI_ASM_INST(0xa4004185)  // ld1b	{z5.b}, p0/z, [x12, x0]
    KAI_ASM_INST(0x4504980a)  // smmla	z10.s, z0.b, z4.b
    KAI_ASM_INST(0x45039808)  // smmla	z8.s, z0.b, z3.b
    KAI_ASM_INST(0x45029815)  // smmla	z21.s, z0.b, z2.b
    KAI_ASM_INST(0x45019813)  // smmla	z19.s, z0.b, z1.b
    KAI_ASM_INST(0x450498a6)  // smmla	z6.s, z5.b, z4.b
    KAI_ASM_INST(0xa4004140)  // ld1b	{z0.b}, p0/z, [x10, x0]
    KAI_ASM_INST(0x450398b2)  // smmla	z18.s, z5.b, z3.b
    KAI_ASM_INST(0x91010000)  // add	x0, x0, #0x40
    KAI_ASM_INST(0x450298b6)  // smmla	z22.s, z5.b, z2.b
    KAI_ASM_INST(0x450198b4)  // smmla	z20.s, z5.b, z1.b
    KAI_ASM_INST(0x45049809)  // smmla	z9.s, z0.b, z4.b
    KAI_ASM_INST(0x45039807)  // smmla	z7.s, z0.b, z3.b
    KAI_ASM_INST(0x4502980d)  // smmla	z13.s, z0.b, z2.b
    KAI_ASM_INST(0x45019817)  // smmla	z23.s, z0.b, z1.b
    KAI_ASM_INST(0xeb02007f)  // cmp	x3, x2
    KAI_ASM_INST(0x54fffca1)  // b.ne	dc0 <MlasGemmS8S8KernelSmmlaSveImpl+0xdc0>
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x9a8090e0)  // csel	x0, x7, x0, ls
    KAI_ASM_INST(0x05f16960)  // uzp1	z0.d, z11.d, z17.d
    KAI_ASM_INST(0x05ec6a02)  // uzp1	z2.d, z16.d, z12.d
    KAI_ASM_INST(0x05f16d6b)  // uzp2	z11.d, z11.d, z17.d
    KAI_ASM_INST(0x25a01fe2)  // whilelo	p2.s, xzr, x0
    KAI_ASM_INST(0xb940abe0)  // ldr	w0, [sp, #168]
    KAI_ASM_INST(0x05ec6e0c)  // uzp2	z12.d, z16.d, z12.d
    KAI_ASM_INST(0x35006f80)  // cbnz	w0, 1c44 <MlasGemmS8S8KernelSmmlaSveImpl+0x1c44>
    KAI_ASM_INST(0xf94073e6)  // ldr	x6, [sp, #224]
    KAI_ASM_INST(0xd37ef520)  // lsl	x0, x9, #2
    KAI_ASM_INST(0x8b0000c2)  // add	x2, x6, x0
    KAI_ASM_INST(0xa540a841)  // ld1w	{z1.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a00020)  // add	z0.s, z1.s, z0.s
    KAI_ASM_INST(0xe540e840)  // st1w	{z0.s}, p2, [x2]
    KAI_ASM_INST(0xf10012df)  // cmp	x22, #0x4
    KAI_ASM_INST(0x540074c9)  // b.ls	1d0c <MlasGemmS8S8KernelSmmlaSveImpl+0x1d0c>
    KAI_ASM_INST(0xd10010e7)  // sub	x7, x7, #0x4
    KAI_ASM_INST(0x25a71fe3)  // whilelo	p3.s, xzr, x7
    KAI_ASM_INST(0xa94f23e7)  // ldp	x7, x8, [sp, #240]
    KAI_ASM_INST(0x05e76d20)  // uzp2	z0.d, z9.d, z7.d
    KAI_ASM_INST(0xd37ef6a2)  // lsl	x2, x21, #2
    KAI_ASM_INST(0x05f268d0)  // uzp1	z16.d, z6.d, z18.d
    KAI_ASM_INST(0xaa0203e5)  // mov	x5, x2
    KAI_ASM_INST(0x8b0200c2)  // add	x2, x6, x2
    KAI_ASM_INST(0xa540ac51)  // ld1w	{z17.s}, p3/z, [x2]
    KAI_ASM_INST(0x04a20231)  // add	z17.s, z17.s, z2.s
    KAI_ASM_INST(0x05e76922)  // uzp1	z2.d, z9.d, z7.d
    KAI_ASM_INST(0xe540ec51)  // st1w	{z17.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab47)  // ld1w	{z7.s}, p2/z, [x26]
    KAI_ASM_INST(0x04ab00e7)  // add	z7.s, z7.s, z11.s
    KAI_ASM_INST(0x8b0000e2)  // add	x2, x7, x0
    KAI_ASM_INST(0x8b000106)  // add	x6, x8, x0
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0xe540eb47)  // st1w	{z7.s}, p2, [x26]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04ac00e7)  // add	z7.s, z7.s, z12.s
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540a847)  // ld1w	{z7.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a70207)  // add	z7.s, z16.s, z7.s
    KAI_ASM_INST(0x8b0500e0)  // add	x0, x7, x5
    KAI_ASM_INST(0x05f46acf)  // uzp1	z15.d, z22.d, z20.d
    KAI_ASM_INST(0xe540e847)  // st1w	{z7.s}, p2, [x2]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a701e7)  // add	z7.s, z15.s, z7.s
    KAI_ASM_INST(0x05f26cce)  // uzp2	z14.d, z6.d, z18.d
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab27)  // ld1w	{z7.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a701c7)  // add	z7.s, z14.s, z7.s
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x05f46ed4)  // uzp2	z20.d, z22.d, z20.d
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a70287)  // add	z7.s, z20.s, z7.s
    KAI_ASM_INST(0x05e86946)  // uzp1	z6.d, z10.d, z8.d
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540a8c7)  // ld1w	{z7.s}, p2/z, [x6]
    KAI_ASM_INST(0x04a700c6)  // add	z6.s, z6.s, z7.s
    KAI_ASM_INST(0x8b050105)  // add	x5, x8, x5
    KAI_ASM_INST(0x05f36aa5)  // uzp1	z5.d, z21.d, z19.d
    KAI_ASM_INST(0xe540e8c6)  // st1w	{z6.s}, p2, [x6]
    KAI_ASM_INST(0xa540aca6)  // ld1w	{z6.s}, p3/z, [x5]
    KAI_ASM_INST(0x04a600a5)  // add	z5.s, z5.s, z6.s
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86d44)  // uzp2	z4.d, z10.d, z8.d
    KAI_ASM_INST(0xe540eca5)  // st1w	{z5.s}, p3, [x5]
    KAI_ASM_INST(0xa540a805)  // ld1w	{z5.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a50084)  // add	z4.s, z4.s, z5.s
    KAI_ASM_INST(0x05f36ea3)  // uzp2	z3.d, z21.d, z19.d
    KAI_ASM_INST(0xe540e804)  // st1w	{z4.s}, p2, [x0]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0xa540ac04)  // ld1w	{z4.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a40063)  // add	z3.s, z3.s, z4.s
    KAI_ASM_INST(0xe540ec03)  // st1w	{z3.s}, p3, [x0]
    KAI_ASM_INST(0xa540abc3)  // ld1w	{z3.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0x05f769a1)  // uzp1	z1.d, z13.d, z23.d
    KAI_ASM_INST(0xe540ebc2)  // st1w	{z2.s}, p2, [x30]
    KAI_ASM_INST(0xa540ac02)  // ld1w	{z2.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540ec01)  // st1w	{z1.s}, p3, [x0]
    KAI_ASM_INST(0xa540aae1)  // ld1w	{z1.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0x05f76dad)  // uzp2	z13.d, z13.d, z23.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xa540ac00)  // ld1w	{z0.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a001ad)  // add	z13.s, z13.s, z0.s
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540f00d)  // st1w	{z13.s}, p4, [x0]
    KAI_ASM_INST(0x91002129)  // add	x9, x9, #0x8
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xd10022d6)  // sub	x22, x22, #0x8
    KAI_ASM_INST(0x910022b5)  // add	x21, x21, #0x8
    KAI_ASM_INST(0x910082f7)  // add	x23, x23, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005fe0)  // str	x0, [sp, #184]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x910083de)  // add	x30, x30, #0x20
    KAI_ASM_INST(0x91008339)  // add	x25, x25, #0x20
    KAI_ASM_INST(0x9100835a)  // add	x26, x26, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005be0)  // str	x0, [sp, #176]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9006fe0)  // str	x0, [sp, #216]
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf90063e0)  // str	x0, [sp, #192]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0xeb09001f)  // cmp	x0, x9
    KAI_ASM_INST(0x54ffd888)  // b.hi	b20 <MlasGemmS8S8KernelSmmlaSveImpl+0xb20>
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x6d493fee)  // ldp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0xf9408fe4)  // ldr	x4, [sp, #280]
    KAI_ASM_INST(0xe5404600)  // st1w	{z0.s}, p1, [x16, x0, lsl #2]
    KAI_ASM_INST(0xe5404481)  // st1w	{z1.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e590)  // st1w	{z16.s}, p1, [x12]
    KAI_ASM_INST(0xf94063e4)  // ldr	x4, [sp, #192]
    KAI_ASM_INST(0xe540e48d)  // st1w	{z13.s}, p1, [x4]
    KAI_ASM_INST(0x05fe6965)  // uzp1	z5.d, z11.d, z30.d
    KAI_ASM_INST(0x05fd694e)  // uzp1	z14.d, z10.d, z29.d
    KAI_ASM_INST(0xf94093e4)  // ldr	x4, [sp, #288]
    KAI_ASM_INST(0xe5404482)  // st1w	{z2.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0x8b180202)  // add	x2, x16, x24
    KAI_ASM_INST(0x05fe6d6b)  // uzp2	z11.d, z11.d, z30.d
    KAI_ASM_INST(0xf94087e4)  // ldr	x4, [sp, #264]
    KAI_ASM_INST(0xe540448c)  // st1w	{z12.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e765)  // st1w	{z5.s}, p1, [x27]
    KAI_ASM_INST(0x05fd6d4a)  // uzp2	z10.d, z10.d, z29.d
    KAI_ASM_INST(0xf94067e4)  // ldr	x4, [sp, #200]
    KAI_ASM_INST(0xe540e48e)  // st1w	{z14.s}, p1, [x4]
    KAI_ASM_INST(0xe540e6cb)  // st1w	{z11.s}, p1, [x22]
    KAI_ASM_INST(0x05fa68e0)  // uzp1	z0.d, z7.d, z26.d
    KAI_ASM_INST(0xf9406be4)  // ldr	x4, [sp, #208]
    KAI_ASM_INST(0xe540e48a)  // st1w	{z10.s}, p1, [x4]
    KAI_ASM_INST(0x05fc6924)  // uzp1	z4.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6903)  // uzp1	z3.d, z8.d, z27.d
    KAI_ASM_INST(0xf9408be4)  // ldr	x4, [sp, #272]
    KAI_ASM_INST(0x05fc6d29)  // uzp2	z9.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6d08)  // uzp2	z8.d, z8.d, z27.d
    KAI_ASM_INST(0xe540e524)  // st1w	{z4.s}, p1, [x9]
    KAI_ASM_INST(0xe5404489)  // st1w	{z9.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e543)  // st1w	{z3.s}, p1, [x10]
    KAI_ASM_INST(0xf9406fe4)  // ldr	x4, [sp, #216]
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xe540e448)  // st1w	{z8.s}, p1, [x2]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x05f968c0)  // uzp1	z0.d, z6.d, z25.d
    KAI_ASM_INST(0x05fa6ce7)  // uzp2	z7.d, z7.d, z26.d
    KAI_ASM_INST(0x05f96cc6)  // uzp2	z6.d, z6.d, z25.d
    KAI_ASM_INST(0xe540e447)  // st1w	{z7.s}, p1, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0x17fffe0d)  // b	918 <MlasGemmS8S8KernelSmmlaSveImpl+0x918>
    KAI_ASM_INST(0x8b170202)  // add	x2, x16, x23
    KAI_ASM_INST(0xe540e7c3)  // st1w	{z3.s}, p1, [x30]
    KAI_ASM_INST(0xe540e4e2)  // st1w	{z2.s}, p1, [x7]
    KAI_ASM_INST(0xe540e453)  // st1w	{z19.s}, p1, [x2]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0xe540e452)  // st1w	{z18.s}, p1, [x2]
    KAI_ASM_INST(0xf9405fe2)  // ldr	x2, [sp, #184]
    KAI_ASM_INST(0xe540e451)  // st1w	{z17.s}, p1, [x2]
    KAI_ASM_INST(0x17fffe25)  // b	99c <MlasGemmS8S8KernelSmmlaSveImpl+0x99c>
    KAI_ASM_INST(0x2d40910a)  // ldp	s10, s4, [x8, #4]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d419509)  // ldp	s9, s5, [x8, #12]
    KAI_ASM_INST(0x2d428d07)  // ldp	s7, s3, [x8, #20]
    KAI_ASM_INST(0x2d438906)  // ldp	s6, s2, [x8, #28]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x4d40c901)  // ld1r	{v1.4s}, [x8]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0xbd402500)  // ldr	s0, [x8, #36]
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c0541)  // mov	v1.s[1], v10.s[0]
    KAI_ASM_INST(0x6e0c0524)  // mov	v4.s[1], v9.s[0]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c04e5)  // mov	v5.s[1], v7.s[0]
    KAI_ASM_INST(0x6e1c0541)  // mov	v1.s[3], v10.s[0]
    KAI_ASM_INST(0x6e0c04c3)  // mov	v3.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c0402)  // mov	v2.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0524)  // mov	v4.s[3], v9.s[0]
    KAI_ASM_INST(0x6e1c04e5)  // mov	v5.s[3], v7.s[0]
    KAI_ASM_INST(0x6e1c04c3)  // mov	v3.s[3], v6.s[0]
    KAI_ASM_INST(0x4ea88429)  // add	v9.4s, v1.4s, v8.4s
    KAI_ASM_INST(0x4ea88486)  // add	v6.4s, v4.4s, v8.4s
    KAI_ASM_INST(0x6e1c0402)  // mov	v2.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea884a7)  // add	v7.4s, v5.4s, v8.4s
    KAI_ASM_INST(0x4ea11c3a)  // mov	v26.16b, v1.16b
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4b5)  // ld1w	{z21.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea38501)  // add	v1.4s, v8.4s, v3.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0x4f000426)  // movi	v6.4s, #0x1
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a4b4)  // ld1w	{z20.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea28500)  // add	v0.4s, v8.4s, v2.4s
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4ad)  // ld1w	{z13.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea61cd9)  // mov	v25.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd7)  // mov	v23.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd3)  // mov	v19.16b, v6.16b
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4ac)  // ld1w	{z12.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea61cd6)  // mov	v22.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd2)  // mov	v18.16b, v6.16b
    KAI_ASM_INST(0x17fffcfc)  // b	5a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x5a8>
    KAI_ASM_INST(0x8b0e01a9)  // add	x9, x13, x14
    KAI_ASM_INST(0xb4003785)  // cbz	x5, 18b0 <MlasGemmS8S8KernelSmmlaSveImpl+0x18b0>
    KAI_ASM_INST(0x8b0604c7)  // add	x7, x6, x6, lsl #1
    KAI_ASM_INST(0xcb060be0)  // neg	x0, x6, lsl #2
    KAI_ASM_INST(0xa90363f7)  // stp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xd37df0c5)  // lsl	x5, x6, #3
    KAI_ASM_INST(0xd37ef4e7)  // lsl	x7, x7, #2
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0xcb0600e2)  // sub	x2, x7, x6
    KAI_ASM_INST(0xa9046bf9)  // stp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xd37ef4c8)  // lsl	x8, x6, #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xd37ef442)  // lsl	x2, x2, #2
    KAI_ASM_INST(0xa9025bf5)  // stp	x21, x22, [sp, #32]
    KAI_ASM_INST(0x8b02020c)  // add	x12, x16, x2
    KAI_ASM_INST(0x8b000197)  // add	x23, x12, x0
    KAI_ASM_INST(0x8b060104)  // add	x4, x8, x6
    KAI_ASM_INST(0x8b0002fb)  // add	x27, x23, x0
    KAI_ASM_INST(0x8b0600ab)  // add	x11, x5, x6
    KAI_ASM_INST(0x8b00037a)  // add	x26, x27, x0
    KAI_ASM_INST(0xcb0600aa)  // sub	x10, x5, x6
    KAI_ASM_INST(0x8b000359)  // add	x25, x26, x0
    KAI_ASM_INST(0x910040af)  // add	x15, x5, #0x10
    KAI_ASM_INST(0x8b000336)  // add	x22, x25, x0
    KAI_ASM_INST(0x8b040a04)  // add	x4, x16, x4, lsl #2
    KAI_ASM_INST(0x8b0002d8)  // add	x24, x22, x0
    KAI_ASM_INST(0x91004108)  // add	x8, x8, #0x10
    KAI_ASM_INST(0x8b00031e)  // add	x30, x24, x0
    KAI_ASM_INST(0x8b0b0a0b)  // add	x11, x16, x11, lsl #2
    KAI_ASM_INST(0x8b0003c5)  // add	x5, x30, x0
    KAI_ASM_INST(0x8b0a0a0a)  // add	x10, x16, x10, lsl #2
    KAI_ASM_INST(0x8b0000b4)  // add	x20, x5, x0
    KAI_ASM_INST(0xaa1003f5)  // mov	x21, x16
    KAI_ASM_INST(0x8b000280)  // add	x0, x20, x0
    KAI_ASM_INST(0xa90b33e0)  // stp	x0, x12, [sp, #176]
    KAI_ASM_INST(0x91004080)  // add	x0, x4, #0x10
    KAI_ASM_INST(0xf90073e0)  // str	x0, [sp, #224]
    KAI_ASM_INST(0x8b080200)  // add	x0, x16, x8
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0x91004160)  // add	x0, x11, #0x10
    KAI_ASM_INST(0xf9007be0)  // str	x0, [sp, #240]
    KAI_ASM_INST(0x91004140)  // add	x0, x10, #0x10
    KAI_ASM_INST(0x910004d2)  // add	x18, x6, #0x1
    KAI_ASM_INST(0x8b061606)  // add	x6, x16, x6, lsl #5
    KAI_ASM_INST(0xf90077e0)  // str	x0, [sp, #232]
    KAI_ASM_INST(0x8b0f02a0)  // add	x0, x21, x15
    KAI_ASM_INST(0xb940fbec)  // ldr	w12, [sp, #248]
    KAI_ASM_INST(0x910040e7)  // add	x7, x7, #0x10
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0x910040c0)  // add	x0, x6, #0x10
    KAI_ASM_INST(0xa91117f1)  // stp	x17, x5, [sp, #272]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x8b070207)  // add	x7, x16, x7
    KAI_ASM_INST(0xf9007fe0)  // str	x0, [sp, #248]
    KAI_ASM_INST(0x8b1212a0)  // add	x0, x21, x18, lsl #4
    KAI_ASM_INST(0xf940a3e5)  // ldr	x5, [sp, #320]
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0x1200018c)  // and	w12, w12, #0x1
    KAI_ASM_INST(0xaa1c03f0)  // mov	x16, x28
    KAI_ASM_INST(0x6d0627e8)  // stp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x91004028)  // add	x8, x1, #0x10
    KAI_ASM_INST(0x9100c1ab)  // add	x11, x13, #0x30
    KAI_ASM_INST(0x6d072fea)  // stp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x9100412a)  // add	x10, x9, #0x10
    KAI_ASM_INST(0x910041af)  // add	x15, x13, #0x10
    KAI_ASM_INST(0x6d0837ec)  // stp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x9100c026)  // add	x6, x1, #0x30
    KAI_ASM_INST(0xaa1103f2)  // mov	x18, x17
    KAI_ASM_INST(0x6d093fee)  // stp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xb900abec)  // str	w12, [sp, #168]
    KAI_ASM_INST(0x910081ac)  // add	x12, x13, #0x20
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0xf9006fe7)  // str	x7, [sp, #216]
    KAI_ASM_INST(0x91008027)  // add	x7, x1, #0x20
    KAI_ASM_INST(0xf90083e0)  // str	x0, [sp, #256]
    KAI_ASM_INST(0x910042a0)  // add	x0, x21, #0x10
    KAI_ASM_INST(0xf90087e0)  // str	x0, [sp, #264]
    KAI_ASM_INST(0xf900a7fc)  // str	x28, [sp, #328]
    KAI_ASM_INST(0xf9408ffc)  // ldr	x28, [sp, #280]
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0xf100225f)  // cmp	x18, #0x8
    KAI_ASM_INST(0x4d40ca64)  // ld1r	{v4.4s}, [x19]
    KAI_ASM_INST(0x9a809251)  // csel	x17, x18, x0, ls
    KAI_ASM_INST(0xf940a7e0)  // ldr	x0, [sp, #328]
    KAI_ASM_INST(0x6e140404)  // mov	v4.s[2], v0.s[0]
    KAI_ASM_INST(0x6e1c0404)  // mov	v4.s[3], v0.s[0]
    KAI_ASM_INST(0xb40030c0)  // cbz	x0, 1930 <MlasGemmS8S8KernelSmmlaSveImpl+0x1930>
    KAI_ASM_INST(0x2d4088ab)  // ldp	s11, s2, [x5, #4]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4184a8)  // ldp	s8, s1, [x5, #12]
    KAI_ASM_INST(0x2d41a60a)  // ldp	s10, s9, [x16, #12]
    KAI_ASM_INST(0x2d40ca0c)  // ldp	s12, s18, [x16, #4]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x4d40c8a3)  // ld1r	{v3.4s}, [x5]
    KAI_ASM_INST(0x2d429607)  // ldp	s7, s5, [x16, #20]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd4014a6)  // ldr	s6, [x5, #20]
    KAI_ASM_INST(0x6e0c0563)  // mov	v3.s[1], v11.s[0]
    KAI_ASM_INST(0x4d40ca15)  // ld1r	{v21.4s}, [x16]
    KAI_ASM_INST(0x4e040529)  // dup	v9.4s, v9.s[0]
    KAI_ASM_INST(0xbd401e00)  // ldr	s0, [x16, #28]
    KAI_ASM_INST(0x6e0c0502)  // mov	v2.s[1], v8.s[0]
    KAI_ASM_INST(0x4e040652)  // dup	v18.4s, v18.s[0]
    KAI_ASM_INST(0x6e0c04c1)  // mov	v1.s[1], v6.s[0]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0x6e0c04e9)  // mov	v9.s[1], v7.s[0]
    KAI_ASM_INST(0x6e1c0563)  // mov	v3.s[3], v11.s[0]
    KAI_ASM_INST(0x6e0c0595)  // mov	v21.s[1], v12.s[0]
    KAI_ASM_INST(0x6e1c0502)  // mov	v2.s[3], v8.s[0]
    KAI_ASM_INST(0x6e0c0552)  // mov	v18.s[1], v10.s[0]
    KAI_ASM_INST(0x6e1c04c1)  // mov	v1.s[3], v6.s[0]
    KAI_ASM_INST(0x6e0c0405)  // mov	v5.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c04e9)  // mov	v9.s[3], v7.s[0]
    KAI_ASM_INST(0x6e1c0595)  // mov	v21.s[3], v12.s[0]
    KAI_ASM_INST(0x4ea31c67)  // mov	v7.16b, v3.16b
    KAI_ASM_INST(0x4ea21c46)  // mov	v6.16b, v2.16b
    KAI_ASM_INST(0x6e1c0552)  // mov	v18.s[3], v10.s[0]
    KAI_ASM_INST(0x6e1c0405)  // mov	v5.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea11c20)  // mov	v0.16b, v1.16b
    KAI_ASM_INST(0x4ea496a7)  // mla	v7.4s, v21.4s, v4.4s
    KAI_ASM_INST(0x4ea49646)  // mla	v6.4s, v18.4s, v4.4s
    KAI_ASM_INST(0x4ea49520)  // mla	v0.4s, v9.4s, v4.4s
    KAI_ASM_INST(0x4ea59c84)  // mul	v4.4s, v4.4s, v5.4s
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a41b)  // ld1w	{z27.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4328a0)  // ldp	s0, s10, [x5, #24]
    KAI_ASM_INST(0x3cc08266)  // ldur	q6, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x4ea31c71)  // mov	v17.16b, v3.16b
    KAI_ASM_INST(0x4e8638c8)  // zip1	v8.4s, v6.4s, v6.4s
    KAI_ASM_INST(0x4ea21c4d)  // mov	v13.16b, v2.16b
    KAI_ASM_INST(0x6e0c0540)  // mov	v0.s[1], v10.s[0]
    KAI_ASM_INST(0x4ea11c2c)  // mov	v12.16b, v1.16b
    KAI_ASM_INST(0x4eb59511)  // mla	v17.4s, v8.4s, v21.4s
    KAI_ASM_INST(0x4eb2950d)  // mla	v13.4s, v8.4s, v18.4s
    KAI_ASM_INST(0x6e1c0540)  // mov	v0.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea9950c)  // mla	v12.4s, v8.4s, v9.4s
    KAI_ASM_INST(0x3cc18267)  // ldur	q7, [x19, #24]
    KAI_ASM_INST(0x4ea08484)  // add	v4.4s, v4.4s, v0.4s
    KAI_ASM_INST(0x4ea01c0a)  // mov	v10.16b, v0.16b
    KAI_ASM_INST(0x4e8678c6)  // zip2	v6.4s, v6.4s, v6.4s
    KAI_ASM_INST(0x4ea31c70)  // mov	v16.16b, v3.16b
    KAI_ASM_INST(0x4ea21c4f)  // mov	v15.16b, v2.16b
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41a)  // ld1w	{z26.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea894aa)  // mla	v10.4s, v5.4s, v8.4s
    KAI_ASM_INST(0x3d804ff1)  // str	q17, [sp, #304]
    KAI_ASM_INST(0xa540a411)  // ld1w	{z17.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea11c2e)  // mov	v14.16b, v1.16b
    KAI_ASM_INST(0x3d804fed)  // str	q13, [sp, #304]
    KAI_ASM_INST(0xa540a408)  // ld1w	{z8.s}, p1/z, [x0]
    KAI_ASM_INST(0x4e8738eb)  // zip1	v11.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x3d804fec)  // str	q12, [sp, #304]
    KAI_ASM_INST(0xa540a419)  // ld1w	{z25.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea01c0c)  // mov	v12.16b, v0.16b
    KAI_ASM_INST(0x3d804fea)  // str	q10, [sp, #304]
    KAI_ASM_INST(0xa540a40d)  // ld1w	{z13.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb594d0)  // mla	v16.4s, v6.4s, v21.4s
    KAI_ASM_INST(0x4eb294cf)  // mla	v15.4s, v6.4s, v18.4s
    KAI_ASM_INST(0x4ea994ce)  // mla	v14.4s, v6.4s, v9.4s
    KAI_ASM_INST(0x4ea594cc)  // mla	v12.4s, v6.4s, v5.4s
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x3d804ff0)  // str	q16, [sp, #304]
    KAI_ASM_INST(0xa540a410)  // ld1w	{z16.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea21c44)  // mov	v4.16b, v2.16b
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a41f)  // ld1w	{z31.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb59566)  // mla	v6.4s, v11.4s, v21.4s
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a418)  // ld1w	{z24.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb29564)  // mla	v4.4s, v11.4s, v18.4s
    KAI_ASM_INST(0x3d804fec)  // str	q12, [sp, #304]
    KAI_ASM_INST(0xa540a40c)  // ld1w	{z12.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea11c2f)  // mov	v15.16b, v1.16b
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0x4ea01c0e)  // mov	v14.16b, v0.16b
    KAI_ASM_INST(0x4e8778ea)  // zip2	v10.4s, v7.4s, v7.4s
    KAI_ASM_INST(0xa540a407)  // ld1w	{z7.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41e)  // ld1w	{z30.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea21c44)  // mov	v4.16b, v2.16b
    KAI_ASM_INST(0x4ea9956f)  // mla	v15.4s, v11.4s, v9.4s
    KAI_ASM_INST(0x4ea5956e)  // mla	v14.4s, v11.4s, v5.4s
    KAI_ASM_INST(0x4eb59546)  // mla	v6.4s, v10.4s, v21.4s
    KAI_ASM_INST(0x4eb29544)  // mla	v4.4s, v10.4s, v18.4s
    KAI_ASM_INST(0x4ea11c36)  // mov	v22.16b, v1.16b
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a417)  // ld1w	{z23.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea01c0f)  // mov	v15.16b, v0.16b
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a40b)  // ld1w	{z11.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea99556)  // mla	v22.4s, v10.4s, v9.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a406)  // ld1w	{z6.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea5954f)  // mla	v15.4s, v10.4s, v5.4s
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41d)  // ld1w	{z29.s}, p1/z, [x0]
    KAI_ASM_INST(0x2d453a64)  // ldp	s4, s14, [x19, #40]
    KAI_ASM_INST(0x3d804ff6)  // str	q22, [sp, #304]
    KAI_ASM_INST(0xa540a416)  // ld1w	{z22.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a40a)  // ld1w	{z10.s}, p1/z, [x0]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x6e1405c4)  // mov	v4.s[2], v14.s[0]
    KAI_ASM_INST(0x6e1c05c4)  // mov	v4.s[3], v14.s[0]
    KAI_ASM_INST(0x4eb59483)  // mla	v3.4s, v4.4s, v21.4s
    KAI_ASM_INST(0x4eb29482)  // mla	v2.4s, v4.4s, v18.4s
    KAI_ASM_INST(0x4ea99481)  // mla	v1.4s, v4.4s, v9.4s
    KAI_ASM_INST(0x4ea59480)  // mla	v0.4s, v4.4s, v5.4s
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a412)  // ld1w	{z18.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a41c)  // ld1w	{z28.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a415)  // ld1w	{z21.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a409)  // ld1w	{z9.s}, p1/z, [x0]
    KAI_ASM_INST(0xb4000583)  // cbz	x3, 15ec <MlasGemmS8S8KernelSmmlaSveImpl+0x15ec>
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xd2800000)  // mov	x0, #0x0
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xa40041a4)  // ld1b	{z4.b}, p0/z, [x13, x0]
    KAI_ASM_INST(0xa4004023)  // ld1b	{z3.b}, p0/z, [x1, x0]
    KAI_ASM_INST(0xa4004102)  // ld1b	{z2.b}, p0/z, [x8, x0]
    KAI_ASM_INST(0xa40040e1)  // ld1b	{z1.b}, p0/z, [x7, x0]
    KAI_ASM_INST(0xa40040c0)  // ld1b	{z0.b}, p0/z, [x6, x0]
    KAI_ASM_INST(0x4503989b)  // smmla	z27.s, z4.b, z3.b
    KAI_ASM_INST(0x45029893)  // smmla	z19.s, z4.b, z2.b
    KAI_ASM_INST(0x45019894)  // smmla	z20.s, z4.b, z1.b
    KAI_ASM_INST(0x4500989a)  // smmla	z26.s, z4.b, z0.b
    KAI_ASM_INST(0xa4004184)  // ld1b	{z4.b}, p0/z, [x12, x0]
    KAI_ASM_INST(0xa40041e5)  // ld1b	{z5.b}, p0/z, [x15, x0]
    KAI_ASM_INST(0x45039890)  // smmla	z16.s, z4.b, z3.b
    KAI_ASM_INST(0x4502989f)  // smmla	z31.s, z4.b, z2.b
    KAI_ASM_INST(0x45019898)  // smmla	z24.s, z4.b, z1.b
    KAI_ASM_INST(0x4500988c)  // smmla	z12.s, z4.b, z0.b
    KAI_ASM_INST(0x91000484)  // add	x4, x4, #0x1
    KAI_ASM_INST(0xa4024124)  // ld1b	{z4.b}, p0/z, [x9, x2]
    KAI_ASM_INST(0x450398b1)  // smmla	z17.s, z5.b, z3.b
    KAI_ASM_INST(0x450298a8)  // smmla	z8.s, z5.b, z2.b
    KAI_ASM_INST(0x450198b9)  // smmla	z25.s, z5.b, z1.b
    KAI_ASM_INST(0x450098ad)  // smmla	z13.s, z5.b, z0.b
    KAI_ASM_INST(0x45039886)  // smmla	z6.s, z4.b, z3.b
    KAI_ASM_INST(0xa4004165)  // ld1b	{z5.b}, p0/z, [x11, x0]
    KAI_ASM_INST(0x4502989d)  // smmla	z29.s, z4.b, z2.b
    KAI_ASM_INST(0x45019896)  // smmla	z22.s, z4.b, z1.b
    KAI_ASM_INST(0x4500988a)  // smmla	z10.s, z4.b, z0.b
    KAI_ASM_INST(0x91010000)  // add	x0, x0, #0x40
    KAI_ASM_INST(0xa4024144)  // ld1b	{z4.b}, p0/z, [x10, x2]
    KAI_ASM_INST(0x450398a7)  // smmla	z7.s, z5.b, z3.b
    KAI_ASM_INST(0x91008042)  // add	x2, x2, #0x20
    KAI_ASM_INST(0x450298be)  // smmla	z30.s, z5.b, z2.b
    KAI_ASM_INST(0x450198b7)  // smmla	z23.s, z5.b, z1.b
    KAI_ASM_INST(0x450098ab)  // smmla	z11.s, z5.b, z0.b
    KAI_ASM_INST(0x45039892)  // smmla	z18.s, z4.b, z3.b
    KAI_ASM_INST(0x4502989c)  // smmla	z28.s, z4.b, z2.b
    KAI_ASM_INST(0x45019895)  // smmla	z21.s, z4.b, z1.b
    KAI_ASM_INST(0x45009889)  // smmla	z9.s, z4.b, z0.b
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54fffb41)  // b.ne	1550 <MlasGemmS8S8KernelSmmlaSveImpl+0x1550>
    KAI_ASM_INST(0xf100123f)  // cmp	x17, #0x4
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x9a809220)  // csel	x0, x17, x0, ls
    KAI_ASM_INST(0x05f36b60)  // uzp1	z0.d, z27.d, z19.d
    KAI_ASM_INST(0x05fa6a82)  // uzp1	z2.d, z20.d, z26.d
    KAI_ASM_INST(0x05f36f73)  // uzp2	z19.d, z27.d, z19.d
    KAI_ASM_INST(0x25a01fe2)  // whilelo	p2.s, xzr, x0
    KAI_ASM_INST(0xb940abe0)  // ldr	w0, [sp, #168]
    KAI_ASM_INST(0x05fa6e94)  // uzp2	z20.d, z20.d, z26.d
    KAI_ASM_INST(0x350015a0)  // cbnz	w0, 18c4 <MlasGemmS8S8KernelSmmlaSveImpl+0x18c4>
    KAI_ASM_INST(0xa540aaa1)  // ld1w	{z1.s}, p2/z, [x21]
    KAI_ASM_INST(0x04a00020)  // add	z0.s, z1.s, z0.s
    KAI_ASM_INST(0xe540eaa0)  // st1w	{z0.s}, p2, [x21]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54002529)  // b.ls	1ac8 <MlasGemmS8S8KernelSmmlaSveImpl+0x1ac8>
    KAI_ASM_INST(0xf94087e2)  // ldr	x2, [sp, #264]
    KAI_ASM_INST(0x05fd68c3)  // uzp1	z3.d, z6.d, z29.d
    KAI_ASM_INST(0xd1001220)  // sub	x0, x17, #0x4
    KAI_ASM_INST(0x05fd6cc1)  // uzp2	z1.d, z6.d, z29.d
    KAI_ASM_INST(0x25a01fe3)  // whilelo	p3.s, xzr, x0
    KAI_ASM_INST(0xa540ac4f)  // ld1w	{z15.s}, p3/z, [x2]
    KAI_ASM_INST(0x04a201ef)  // add	z15.s, z15.s, z2.s
    KAI_ASM_INST(0xe540ec4f)  // st1w	{z15.s}, p3, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xa540a846)  // ld1w	{z6.s}, p2/z, [x2]
    KAI_ASM_INST(0x04b300c6)  // add	z6.s, z6.s, z19.s
    KAI_ASM_INST(0xe540e846)  // st1w	{z6.s}, p2, [x2]
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x04b400c6)  // add	z6.s, z6.s, z20.s
    KAI_ASM_INST(0xe540ec46)  // st1w	{z6.s}, p3, [x2]
    KAI_ASM_INST(0xa540aa86)  // ld1w	{z6.s}, p2/z, [x20]
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0x05e86a2e)  // uzp1	z14.d, z17.d, z8.d
    KAI_ASM_INST(0x04a601ce)  // add	z14.s, z14.s, z6.s
    KAI_ASM_INST(0xe540ea8e)  // st1w	{z14.s}, p2, [x20]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ed6b28)  // uzp1	z8.d, z25.d, z13.d
    KAI_ASM_INST(0x04a60108)  // add	z8.s, z8.s, z6.s
    KAI_ASM_INST(0xe540ec48)  // st1w	{z8.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab86)  // ld1w	{z6.s}, p2/z, [x28]
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0x04a60231)  // add	z17.s, z17.s, z6.s
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ed6f39)  // uzp2	z25.d, z25.d, z13.d
    KAI_ASM_INST(0x04a60339)  // add	z25.s, z25.s, z6.s
    KAI_ASM_INST(0xe540ec59)  // st1w	{z25.s}, p3, [x2]
    KAI_ASM_INST(0xa540abc6)  // ld1w	{z6.s}, p2/z, [x30]
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0x05ff6a0d)  // uzp1	z13.d, z16.d, z31.d
    KAI_ASM_INST(0x04a601ad)  // add	z13.s, z13.s, z6.s
    KAI_ASM_INST(0xe540ebcd)  // st1w	{z13.s}, p2, [x30]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ec6b1a)  // uzp1	z26.d, z24.d, z12.d
    KAI_ASM_INST(0x04a6035a)  // add	z26.s, z26.s, z6.s
    KAI_ASM_INST(0xe540ec5a)  // st1w	{z26.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab06)  // ld1w	{z6.s}, p2/z, [x24]
    KAI_ASM_INST(0xf94073e2)  // ldr	x2, [sp, #224]
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x04a60210)  // add	z16.s, z16.s, z6.s
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ec6f18)  // uzp2	z24.d, z24.d, z12.d
    KAI_ASM_INST(0x04a60318)  // add	z24.s, z24.s, z6.s
    KAI_ASM_INST(0xe540ec58)  // st1w	{z24.s}, p3, [x2]
    KAI_ASM_INST(0xa540aac6)  // ld1w	{z6.s}, p2/z, [x22]
    KAI_ASM_INST(0x910042c0)  // add	x0, x22, #0x10
    KAI_ASM_INST(0x05fe68ec)  // uzp1	z12.d, z7.d, z30.d
    KAI_ASM_INST(0x04a6018c)  // add	z12.s, z12.s, z6.s
    KAI_ASM_INST(0xe540eacc)  // st1w	{z12.s}, p2, [x22]
    KAI_ASM_INST(0xa540ac06)  // ld1w	{z6.s}, p3/z, [x0]
    KAI_ASM_INST(0x05eb6afb)  // uzp1	z27.d, z23.d, z11.d
    KAI_ASM_INST(0x04a6037b)  // add	z27.s, z27.s, z6.s
    KAI_ASM_INST(0xe540ec1b)  // st1w	{z27.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab26)  // ld1w	{z6.s}, p2/z, [x25]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0x05eb6ef7)  // uzp2	z23.d, z23.d, z11.d
    KAI_ASM_INST(0x04a600e7)  // add	z7.s, z7.s, z6.s
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xa540ac06)  // ld1w	{z6.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a602e6)  // add	z6.s, z23.s, z6.s
    KAI_ASM_INST(0xe540ec06)  // st1w	{z6.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab46)  // ld1w	{z6.s}, p2/z, [x26]
    KAI_ASM_INST(0x04a60063)  // add	z3.s, z3.s, z6.s
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05ea6ac2)  // uzp1	z2.d, z22.d, z10.d
    KAI_ASM_INST(0xe540eb43)  // st1w	{z3.s}, p2, [x26]
    KAI_ASM_INST(0xa540ac03)  // ld1w	{z3.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xe540ec02)  // st1w	{z2.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab62)  // ld1w	{z2.s}, p2/z, [x27]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05ea6ec0)  // uzp2	z0.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6a44)  // uzp1	z4.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96aa5)  // uzp1	z5.d, z21.d, z9.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540eb61)  // st1w	{z1.s}, p2, [x27]
    KAI_ASM_INST(0xa540ac01)  // ld1w	{z1.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xa540aae0)  // ld1w	{z0.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a00080)  // add	z0.s, z4.s, z0.s
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54001788)  // b.hi	1a90 <MlasGemmS8S8KernelSmmlaSveImpl+0x1a90>
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xa540a800)  // ld1w	{z0.s}, p2/z, [x0]
    KAI_ASM_INST(0x540031c0)  // b.eq	1de4 <MlasGemmS8S8KernelSmmlaSveImpl+0x1de4>
    KAI_ASM_INST(0x04a00252)  // add	z18.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0xa94b93e0)  // ldp	x0, x4, [sp, #184]
    KAI_ASM_INST(0xd1002252)  // sub	x18, x18, #0x8
    KAI_ASM_INST(0xf9408be2)  // ldr	x2, [sp, #272]
    KAI_ASM_INST(0x910082f7)  // add	x23, x23, #0x20
    KAI_ASM_INST(0x9100837b)  // add	x27, x27, #0x20
    KAI_ASM_INST(0x9100835a)  // add	x26, x26, #0x20
    KAI_ASM_INST(0x91008339)  // add	x25, x25, #0x20
    KAI_ASM_INST(0x910082d6)  // add	x22, x22, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xa90b93e0)  // stp	x0, x4, [sp, #184]
    KAI_ASM_INST(0xcb120040)  // sub	x0, x2, x18
    KAI_ASM_INST(0x91008318)  // add	x24, x24, #0x20
    KAI_ASM_INST(0xf9407be4)  // ldr	x4, [sp, #240]
    KAI_ASM_INST(0x910083de)  // add	x30, x30, #0x20
    KAI_ASM_INST(0x9100839c)  // add	x28, x28, #0x20
    KAI_ASM_INST(0x91008294)  // add	x20, x20, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9007be4)  // str	x4, [sp, #240]
    KAI_ASM_INST(0xf94077e4)  // ldr	x4, [sp, #232]
    KAI_ASM_INST(0x910082b5)  // add	x21, x21, #0x20
    KAI_ASM_INST(0x910080a5)  // add	x5, x5, #0x20
    KAI_ASM_INST(0x91008210)  // add	x16, x16, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90077e4)  // str	x4, [sp, #232]
    KAI_ASM_INST(0xf94073e4)  // ldr	x4, [sp, #224]
    KAI_ASM_INST(0x8b0e0021)  // add	x1, x1, x14
    KAI_ASM_INST(0x8b0e0108)  // add	x8, x8, x14
    KAI_ASM_INST(0x8b0e00e7)  // add	x7, x7, x14
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90073e4)  // str	x4, [sp, #224]
    KAI_ASM_INST(0xf9406fe4)  // ldr	x4, [sp, #216]
    KAI_ASM_INST(0x8b0e00c6)  // add	x6, x6, x14
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9006fe4)  // str	x4, [sp, #216]
    KAI_ASM_INST(0xf9405be4)  // ldr	x4, [sp, #176]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9005be4)  // str	x4, [sp, #176]
    KAI_ASM_INST(0xf9406be4)  // ldr	x4, [sp, #208]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9006be4)  // str	x4, [sp, #208]
    KAI_ASM_INST(0xf9407fe4)  // ldr	x4, [sp, #248]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9007fe4)  // str	x4, [sp, #248]
    KAI_ASM_INST(0xf94083e4)  // ldr	x4, [sp, #256]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90083e4)  // str	x4, [sp, #256]
    KAI_ASM_INST(0xf94067e4)  // ldr	x4, [sp, #200]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90067e4)  // str	x4, [sp, #200]
    KAI_ASM_INST(0xf94087e4)  // ldr	x4, [sp, #264]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90087e4)  // str	x4, [sp, #264]
    KAI_ASM_INST(0xeb00005f)  // cmp	x2, x0
    KAI_ASM_INST(0x54ffd348)  // b.hi	12f8 <MlasGemmS8S8KernelSmmlaSveImpl+0x12f8>
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x6d493fee)  // ldp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xd2800180)  // mov	x0, #0xc
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0xe540eaa0)  // st1w	{z0.s}, p2, [x21]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54000688)  // b.hi	199c <MlasGemmS8S8KernelSmmlaSveImpl+0x199c>
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86a24)  // uzp1	z4.d, z17.d, z8.d
    KAI_ASM_INST(0x05ff6a03)  // uzp1	z3.d, z16.d, z31.d
    KAI_ASM_INST(0x05fe68e2)  // uzp1	z2.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd68c1)  // uzp1	z1.d, z6.d, z29.d
    KAI_ASM_INST(0x05fc6a40)  // uzp1	z0.d, z18.d, z28.d
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd6cc6)  // uzp2	z6.d, z6.d, z29.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0xe540e813)  // st1w	{z19.s}, p2, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540ea84)  // st1w	{z4.s}, p2, [x20]
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xe540ebc3)  // st1w	{z3.s}, p2, [x30]
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xe540eac2)  // st1w	{z2.s}, p2, [x22]
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xe540eb41)  // st1w	{z1.s}, p2, [x26]
    KAI_ASM_INST(0xe540eb66)  // st1w	{z6.s}, p2, [x27]
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0x17ffffa3)  // b	17b8 <MlasGemmS8S8KernelSmmlaSveImpl+0x17b8>
    KAI_ASM_INST(0x2d4088a6)  // ldp	s6, s2, [x5, #4]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4184a5)  // ldp	s5, s1, [x5, #12]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x4d40c8a3)  // ld1r	{v3.4s}, [x5]
    KAI_ASM_INST(0x4f000435)  // movi	v21.4s, #0x1
    KAI_ASM_INST(0xbd4014a0)  // ldr	s0, [x5, #20]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0x6e0c04c3)  // mov	v3.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c04a2)  // mov	v2.s[1], v5.s[0]
    KAI_ASM_INST(0x4eb51eb2)  // mov	v18.16b, v21.16b
    KAI_ASM_INST(0x6e0c0401)  // mov	v1.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c04c3)  // mov	v3.s[3], v6.s[0]
    KAI_ASM_INST(0x6e1c04a2)  // mov	v2.s[3], v5.s[0]
    KAI_ASM_INST(0x4eb51ea9)  // mov	v9.16b, v21.16b
    KAI_ASM_INST(0x6e1c0401)  // mov	v1.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea38486)  // add	v6.4s, v4.4s, v3.4s
    KAI_ASM_INST(0x4ea48445)  // add	v5.4s, v2.4s, v4.4s
    KAI_ASM_INST(0x4ea48420)  // add	v0.4s, v1.4s, v4.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a41b)  // ld1w	{z27.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb51ea5)  // mov	v5.16b, v21.16b
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x17fffe8b)  // b	13c4 <MlasGemmS8S8KernelSmmlaSveImpl+0x13c4>
    KAI_ASM_INST(0xf94087e2)  // ldr	x2, [sp, #264]
    KAI_ASM_INST(0x05e86a24)  // uzp1	z4.d, z17.d, z8.d
    KAI_ASM_INST(0xd1001231)  // sub	x17, x17, #0x4
    KAI_ASM_INST(0x25b11fe3)  // whilelo	p3.s, xzr, x17
    KAI_ASM_INST(0xe540ec42)  // st1w	{z2.s}, p3, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xe540e853)  // st1w	{z19.s}, p2, [x2]
    KAI_ASM_INST(0x05ed6b25)  // uzp1	z5.d, z25.d, z13.d
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0xe540ec54)  // st1w	{z20.s}, p3, [x2]
    KAI_ASM_INST(0xe540ea84)  // st1w	{z4.s}, p2, [x20]
    KAI_ASM_INST(0x910042c0)  // add	x0, x22, #0x10
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0xe540ec45)  // st1w	{z5.s}, p3, [x2]
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ed6f39)  // uzp2	z25.d, z25.d, z13.d
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xe540ec59)  // st1w	{z25.s}, p3, [x2]
    KAI_ASM_INST(0x05eb6ae1)  // uzp1	z1.d, z23.d, z11.d
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0x05ff6a0d)  // uzp1	z13.d, z16.d, z31.d
    KAI_ASM_INST(0x05ec6b08)  // uzp1	z8.d, z24.d, z12.d
    KAI_ASM_INST(0x05fe68e3)  // uzp1	z3.d, z7.d, z30.d
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x05ec6f18)  // uzp2	z24.d, z24.d, z12.d
    KAI_ASM_INST(0xe540ebcd)  // st1w	{z13.s}, p2, [x30]
    KAI_ASM_INST(0xe540ec48)  // st1w	{z8.s}, p3, [x2]
    KAI_ASM_INST(0xf94073e2)  // ldr	x2, [sp, #224]
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xe540ec58)  // st1w	{z24.s}, p3, [x2]
    KAI_ASM_INST(0xe540eac3)  // st1w	{z3.s}, p2, [x22]
    KAI_ASM_INST(0xe540ec01)  // st1w	{z1.s}, p3, [x0]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0x05fd68c0)  // uzp1	z0.d, z6.d, z29.d
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0xf9407fe2)  // ldr	x2, [sp, #248]
    KAI_ASM_INST(0x05eb6ef7)  // uzp2	z23.d, z23.d, z11.d
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xe540ec17)  // st1w	{z23.s}, p3, [x0]
    KAI_ASM_INST(0x910042e0)  // add	x0, x23, #0x10
    KAI_ASM_INST(0x05ea6ac2)  // uzp1	z2.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6a41)  // uzp1	z1.d, z18.d, z28.d
    KAI_ASM_INST(0x05fd6cc6)  // uzp2	z6.d, z6.d, z29.d
    KAI_ASM_INST(0x05ea6ed6)  // uzp2	z22.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0xe540eb40)  // st1w	{z0.s}, p2, [x26]
    KAI_ASM_INST(0x05e96aa0)  // uzp1	z0.d, z21.d, z9.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540ec42)  // st1w	{z2.s}, p3, [x2]
    KAI_ASM_INST(0xf9407be2)  // ldr	x2, [sp, #240]
    KAI_ASM_INST(0xe540eb66)  // st1w	{z6.s}, p2, [x27]
    KAI_ASM_INST(0xe540ec56)  // st1w	{z22.s}, p3, [x2]
    KAI_ASM_INST(0xe540eae1)  // st1w	{z1.s}, p2, [x23]
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xe540f015)  // st1w	{z21.s}, p4, [x0]
    KAI_ASM_INST(0x17ffff4b)  // b	17b8 <MlasGemmS8S8KernelSmmlaSveImpl+0x17b8>
    KAI_ASM_INST(0x910042e0)  // add	x0, x23, #0x10
    KAI_ASM_INST(0xd1001231)  // sub	x17, x17, #0x4
    KAI_ASM_INST(0x25b11fe4)  // whilelo	p4.s, xzr, x17
    KAI_ASM_INST(0xa540b000)  // ld1w	{z0.s}, p4/z, [x0]
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0xe540f005)  // st1w	{z5.s}, p4, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xa540a800)  // ld1w	{z0.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a00240)  // add	z0.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e800)  // st1w	{z0.s}, p2, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xa540b000)  // ld1w	{z0.s}, p4/z, [x0]
    KAI_ASM_INST(0x04a002b5)  // add	z21.s, z21.s, z0.s
    KAI_ASM_INST(0x17fffff0)  // b	1a84 <MlasGemmS8S8KernelSmmlaSveImpl+0x1a84>
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86a2b)  // uzp1	z11.d, z17.d, z8.d
    KAI_ASM_INST(0xa540a80c)  // ld1w	{z12.s}, p2/z, [x0]
    KAI_ASM_INST(0x05e86e23)  // uzp2	z3.d, z17.d, z8.d
    KAI_ASM_INST(0x05fe6ce1)  // uzp2	z1.d, z7.d, z30.d
    KAI_ASM_INST(0x05fe68e8)  // uzp1	z8.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd6cc0)  // uzp2	z0.d, z6.d, z29.d
    KAI_ASM_INST(0x05fd68c7)  // uzp1	z7.d, z6.d, z29.d
    KAI_ASM_INST(0x04b3018c)  // add	z12.s, z12.s, z19.s
    KAI_ASM_INST(0xe540e80c)  // st1w	{z12.s}, p2, [x0]
    KAI_ASM_INST(0xa540aa86)  // ld1w	{z6.s}, p2/z, [x20]
    KAI_ASM_INST(0x04a60166)  // add	z6.s, z11.s, z6.s
    KAI_ASM_INST(0xe540ea86)  // st1w	{z6.s}, p2, [x20]
    KAI_ASM_INST(0xa540ab86)  // ld1w	{z6.s}, p2/z, [x28]
    KAI_ASM_INST(0x04a60063)  // add	z3.s, z3.s, z6.s
    KAI_ASM_INST(0x05ff6a0a)  // uzp1	z10.d, z16.d, z31.d
    KAI_ASM_INST(0xe540eb83)  // st1w	{z3.s}, p2, [x28]
    KAI_ASM_INST(0xa540abc3)  // ld1w	{z3.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a30143)  // add	z3.s, z10.s, z3.s
    KAI_ASM_INST(0x05ff6e02)  // uzp2	z2.d, z16.d, z31.d
    KAI_ASM_INST(0xe540ebc3)  // st1w	{z3.s}, p2, [x30]
    KAI_ASM_INST(0xa540ab03)  // ld1w	{z3.s}, p2/z, [x24]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xe540eb02)  // st1w	{z2.s}, p2, [x24]
    KAI_ASM_INST(0xa540aac2)  // ld1w	{z2.s}, p2/z, [x22]
    KAI_ASM_INST(0x04a20102)  // add	z2.s, z8.s, z2.s
    KAI_ASM_INST(0xe540eac2)  // st1w	{z2.s}, p2, [x22]
    KAI_ASM_INST(0xa540ab22)  // ld1w	{z2.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540eb21)  // st1w	{z1.s}, p2, [x25]
    KAI_ASM_INST(0xa540ab41)  // ld1w	{z1.s}, p2/z, [x26]
    KAI_ASM_INST(0x04a100e1)  // add	z1.s, z7.s, z1.s
    KAI_ASM_INST(0x05fc6a44)  // uzp1	z4.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96aa5)  // uzp1	z5.d, z21.d, z9.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540eb41)  // st1w	{z1.s}, p2, [x26]
    KAI_ASM_INST(0xa540ab61)  // ld1w	{z1.s}, p2/z, [x27]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540eb60)  // st1w	{z0.s}, p2, [x27]
    KAI_ASM_INST(0x17ffff0a)  // b	1790 <MlasGemmS8S8KernelSmmlaSveImpl+0x1790>
    KAI_ASM_INST(0xd2800208)  // mov	x8, #0x10
    KAI_ASM_INST(0xd2800022)  // mov	x2, #0x1
    KAI_ASM_INST(0xd2800040)  // mov	x0, #0x2
    KAI_ASM_INST(0x2518e121)  // ptrue	p1.b, vl16
    KAI_ASM_INST(0x17fff940)  // b	7c <MlasGemmS8S8KernelSmmlaSveImpl+0x7c>
    KAI_ASM_INST(0xe540ed64)  // st1w	{z4.s}, p3, [x11]
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0x54000369)  // b.ls	1bf4 <MlasGemmS8S8KernelSmmlaSveImpl+0x1bf4>
    KAI_ASM_INST(0xd10011cc)  // sub	x12, x14, #0x4
    KAI_ASM_INST(0x25ac1fe4)  // whilelo	p4.s, xzr, x12
    KAI_ASM_INST(0x9100416c)  // add	x12, x11, #0x10
    KAI_ASM_INST(0xe540f181)  // st1w	{z1.s}, p4, [x12]
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff3849)  // b.ls	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xe540ed82)  // st1w	{z2.s}, p3, [x12]
    KAI_ASM_INST(0xe540f2a5)  // st1w	{z5.s}, p4, [x21]
    KAI_ASM_INST(0x17fff9be)  // b	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0x2d408486)  // ldp	s6, s1, [x4, #4]
    KAI_ASM_INST(0x2d418085)  // ldp	s5, s0, [x4, #12]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd401483)  // ldr	s3, [x4, #20]
    KAI_ASM_INST(0x4d40c882)  // ld1r	{v2.4s}, [x4]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c04c2)  // mov	v2.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c04a1)  // mov	v1.s[1], v5.s[0]
    KAI_ASM_INST(0x6e0c0460)  // mov	v0.s[1], v3.s[0]
    KAI_ASM_INST(0x6e1c04c2)  // mov	v2.s[3], v6.s[0]
    KAI_ASM_INST(0x6e1c04a1)  // mov	v1.s[3], v5.s[0]
    KAI_ASM_INST(0x6e1c0460)  // mov	v0.s[3], v3.s[0]
    KAI_ASM_INST(0x4ea48442)  // add	v2.4s, v2.4s, v4.4s
    KAI_ASM_INST(0x4ea48421)  // add	v1.4s, v1.4s, v4.4s
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x17fff976)  // b	1c8 <MlasGemmS8S8KernelSmmlaSveImpl+0x1c8>
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff3589)  // b.ls	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xe540ed82)  // st1w	{z2.s}, p3, [x12]
    KAI_ASM_INST(0x17fff9a9)  // b	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff34e9)  // b.ls	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xa540ad80)  // ld1w	{z0.s}, p3/z, [x12]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0xe540ed80)  // st1w	{z0.s}, p3, [x12]
    KAI_ASM_INST(0x17fff9a2)  // b	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xa540ad81)  // ld1w	{z1.s}, p3/z, [x12]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540ed81)  // st1w	{z1.s}, p3, [x12]
    KAI_ASM_INST(0xa540b2a5)  // ld1w	{z5.s}, p4/z, [x21]
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0xe540f2a5)  // st1w	{z5.s}, p4, [x21]
    KAI_ASM_INST(0x17fff99a)  // b	2a8 <MlasGemmS8S8KernelSmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xf94073e0)  // ldr	x0, [sp, #224]
    KAI_ASM_INST(0xe5494800)  // st1w	{z0.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf10012df)  // cmp	x22, #0x4
    KAI_ASM_INST(0x540009e9)  // b.ls	1d8c <MlasGemmS8S8KernelSmmlaSveImpl+0x1d8c>
    KAI_ASM_INST(0xd10010e7)  // sub	x7, x7, #0x4
    KAI_ASM_INST(0x25a71fe3)  // whilelo	p3.s, xzr, x7
    KAI_ASM_INST(0xe5554c02)  // st1w	{z2.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0xe540ec0c)  // st1w	{z12.s}, p3, [x0]
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05f268ce)  // uzp1	z14.d, z6.d, z18.d
    KAI_ASM_INST(0x05f46ac5)  // uzp1	z5.d, z22.d, z20.d
    KAI_ASM_INST(0xe549480e)  // st1w	{z14.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xe5554c05)  // st1w	{z5.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x05f26cc6)  // uzp2	z6.d, z6.d, z18.d
    KAI_ASM_INST(0x05f46ed6)  // uzp2	z22.d, z22.d, z20.d
    KAI_ASM_INST(0xe540eb26)  // st1w	{z6.s}, p2, [x25]
    KAI_ASM_INST(0xe540ec16)  // st1w	{z22.s}, p3, [x0]
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05e86944)  // uzp1	z4.d, z10.d, z8.d
    KAI_ASM_INST(0x05f36aa3)  // uzp1	z3.d, z21.d, z19.d
    KAI_ASM_INST(0xe5494804)  // st1w	{z4.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xe5554c03)  // st1w	{z3.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86d48)  // uzp2	z8.d, z10.d, z8.d
    KAI_ASM_INST(0xe540e808)  // st1w	{z8.s}, p2, [x0]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0x05e76921)  // uzp1	z1.d, z9.d, z7.d
    KAI_ASM_INST(0x05f769a0)  // uzp1	z0.d, z13.d, z23.d
    KAI_ASM_INST(0x05f36eb5)  // uzp2	z21.d, z21.d, z19.d
    KAI_ASM_INST(0x05e76d27)  // uzp2	z7.d, z9.d, z7.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0x05f76dad)  // uzp2	z13.d, z13.d, z23.d
    KAI_ASM_INST(0xe540ec15)  // st1w	{z21.s}, p3, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xe540ebc1)  // st1w	{z1.s}, p2, [x30]
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xe540eae7)  // st1w	{z7.s}, p2, [x23]
    KAI_ASM_INST(0x17fffcae)  // b	f9c <MlasGemmS8S8KernelSmmlaSveImpl+0xf9c>
    KAI_ASM_INST(0x4ea61cc1)  // mov	v1.16b, v6.16b
    KAI_ASM_INST(0x4f00042e)  // movi	v14.4s, #0x1
    KAI_ASM_INST(0x17fffbda)  // b	c58 <MlasGemmS8S8KernelSmmlaSveImpl+0xc58>
    KAI_ASM_INST(0x4ea31c60)  // mov	v0.16b, v3.16b
    KAI_ASM_INST(0x4f000427)  // movi	v7.4s, #0x1
    KAI_ASM_INST(0x17fffbb0)  // b	bbc <MlasGemmS8S8KernelSmmlaSveImpl+0xbbc>
    KAI_ASM_INST(0x4ea61cc1)  // mov	v1.16b, v6.16b
    KAI_ASM_INST(0x4f000428)  // movi	v8.4s, #0x1
    KAI_ASM_INST(0x17fffbc0)  // b	c08 <MlasGemmS8S8KernelSmmlaSveImpl+0xc08>
    KAI_ASM_INST(0xa94f17e2)  // ldp	x2, x5, [sp, #240]
    KAI_ASM_INST(0xa540ab40)  // ld1w	{z0.s}, p2/z, [x26]
    KAI_ASM_INST(0x05f268c5)  // uzp1	z5.d, z6.d, z18.d
    KAI_ASM_INST(0x04ab000b)  // add	z11.s, z0.s, z11.s
    KAI_ASM_INST(0x05f26cc2)  // uzp2	z2.d, z6.d, z18.d
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0x05e86944)  // uzp1	z4.d, z10.d, z8.d
    KAI_ASM_INST(0x05e86d41)  // uzp2	z1.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76923)  // uzp1	z3.d, z9.d, z7.d
    KAI_ASM_INST(0x05e76d20)  // uzp2	z0.d, z9.d, z7.d
    KAI_ASM_INST(0x8b000042)  // add	x2, x2, x0
    KAI_ASM_INST(0xa540a846)  // ld1w	{z6.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a600a5)  // add	z5.s, z5.s, z6.s
    KAI_ASM_INST(0xe540e845)  // st1w	{z5.s}, p2, [x2]
    KAI_ASM_INST(0xa540ab25)  // ld1w	{z5.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a50042)  // add	z2.s, z2.s, z5.s
    KAI_ASM_INST(0x8b0000a0)  // add	x0, x5, x0
    KAI_ASM_INST(0xe540eb22)  // st1w	{z2.s}, p2, [x25]
    KAI_ASM_INST(0xa540a802)  // ld1w	{z2.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a20082)  // add	z2.s, z4.s, z2.s
    KAI_ASM_INST(0xe540e802)  // st1w	{z2.s}, p2, [x0]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0xa540a802)  // ld1w	{z2.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540e801)  // st1w	{z1.s}, p2, [x0]
    KAI_ASM_INST(0xa540abc1)  // ld1w	{z1.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a10061)  // add	z1.s, z3.s, z1.s
    KAI_ASM_INST(0xe540ebc1)  // st1w	{z1.s}, p2, [x30]
    KAI_ASM_INST(0xa540aae1)  // ld1w	{z1.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0x17fffc87)  // b	fa4 <MlasGemmS8S8KernelSmmlaSveImpl+0xfa4>
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05f268c2)  // uzp1	z2.d, z6.d, z18.d
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0xe5494802)  // st1w	{z2.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05e86941)  // uzp1	z1.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76920)  // uzp1	z0.d, z9.d, z7.d
    KAI_ASM_INST(0x05f26cc6)  // uzp2	z6.d, z6.d, z18.d
    KAI_ASM_INST(0x05e86d48)  // uzp2	z8.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76d27)  // uzp2	z7.d, z9.d, z7.d
    KAI_ASM_INST(0xe540eb26)  // st1w	{z6.s}, p2, [x25]
    KAI_ASM_INST(0xe5494801)  // st1w	{z1.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0xe540e808)  // st1w	{z8.s}, p2, [x0]
    KAI_ASM_INST(0xe540ebc0)  // st1w	{z0.s}, p2, [x30]
    KAI_ASM_INST(0xe540eae7)  // st1w	{z7.s}, p2, [x23]
    KAI_ASM_INST(0x17fffc76)  // b	fa4 <MlasGemmS8S8KernelSmmlaSveImpl+0xfa4>
    KAI_ASM_INST(0xd2800408)  // mov	x8, #0x20
    KAI_ASM_INST(0xd2800042)  // mov	x2, #0x2
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x2518e121)  // ptrue	p1.b, vl16
    KAI_ASM_INST(0x17fff8a7)  // b	7c <MlasGemmS8S8KernelSmmlaSveImpl+0x7c>
    KAI_ASM_INST(0x04a00240)  // add	z0.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e800)  // st1w	{z0.s}, p2, [x0]
    KAI_ASM_INST(0x17fffe73)  // b	17b8 <MlasGemmS8S8KernelSmmlaSveImpl+0x17b8>
    KAI_ASM_FUNCTION_END(MlasGemmS8S8KernelSmmlaSveImpl)

    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(MlasGemmU8X8KernelUmmlaSveImpl)
    KAI_ASM_FUNCTION_TYPE(MlasGemmU8X8KernelUmmlaSveImpl)
KAI_ASM_FUNCTION_LABEL(MlasGemmU8X8KernelUmmlaSveImpl)
    KAI_ASM_INST(0xa9ac7bfd)  // stp	x29, x30, [sp, #-320]!
    KAI_ASM_INST(0xaa0003ed)  // mov	x13, x0
    KAI_ASM_INST(0xd1003089)  // sub	x9, x4, #0xc
    KAI_ASM_INST(0x910003fd)  // mov	x29, sp
    KAI_ASM_INST(0x394543e0)  // ldrb	w0, [sp, #336]
    KAI_ASM_INST(0xaa0203f0)  // mov	x16, x2
    KAI_ASM_INST(0xa90153f3)  // stp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xaa0503f1)  // mov	x17, x5
    KAI_ASM_INST(0xaa0703f3)  // mov	x19, x7
    KAI_ASM_INST(0xa90573fb)  // stp	x27, x28, [sp, #80]
    KAI_ASM_INST(0xd37ae46e)  // lsl	x14, x3, #6
    KAI_ASM_INST(0xb900fbe0)  // str	w0, [sp, #248]
    KAI_ASM_INST(0xf940a7fc)  // ldr	x28, [sp, #328]
    KAI_ASM_INST(0xf1000d3f)  // cmp	x9, #0x3
    KAI_ASM_INST(0x54008c29)  // b.ls	11bc <MlasGemmU8X8KernelUmmlaSveImpl+0x11bc>
    KAI_ASM_INST(0xa9025bf5)  // stp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa90363f7)  // stp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9046bf9)  // stp	x25, x26, [sp, #64]
    KAI_ASM_INST(0x6d0627e8)  // stp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d072fea)  // stp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d0837ec)  // stp	d12, d13, [sp, #128]
    KAI_ASM_INST(0xf1001c9f)  // cmp	x4, #0x7
    KAI_ASM_INST(0x540015c8)  // b.hi	310 <MlasGemmU8X8KernelUmmlaSveImpl+0x310>
    KAI_ASM_INST(0xf1000c9f)  // cmp	x4, #0x3
    KAI_ASM_INST(0x5400eb88)  // b.hi	1dd0 <MlasGemmU8X8KernelUmmlaSveImpl+0x1dd0>
    KAI_ASM_INST(0xf100049f)  // cmp	x4, #0x1
    KAI_ASM_INST(0x5400d828)  // b.hi	1b6c <MlasGemmU8X8KernelUmmlaSveImpl+0x1b6c>
    KAI_ASM_INST(0xd2800020)  // mov	x0, #0x1
    KAI_ASM_INST(0xaa0003e2)  // mov	x2, x0
    KAI_ASM_INST(0xd2800108)  // mov	x8, #0x8
    KAI_ASM_INST(0x2518e101)  // ptrue	p1.b, vl8
    KAI_ASM_INST(0xb4001371)  // cbz	x17, 2e8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2e8>
    KAI_ASM_INST(0xf100045f)  // cmp	x2, #0x1
    KAI_ASM_INST(0xb940fbe2)  // ldr	w2, [sp, #248]
    KAI_ASM_INST(0xf940a3e4)  // ldr	x4, [sp, #320]
    KAI_ASM_INST(0x2598e082)  // ptrue	p2.s, vl4
    KAI_ASM_INST(0xd37ef4d9)  // lsl	x25, x6, #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xd37df0d4)  // lsl	x20, x6, #3
    KAI_ASM_INST(0x1200005e)  // and	w30, w2, #0x1
    KAI_ASM_INST(0xd2800058)  // mov	x24, #0x2
    KAI_ASM_INST(0xd280009a)  // mov	x26, #0x4
    KAI_ASM_INST(0xaa1c03e9)  // mov	x9, x28
    KAI_ASM_INST(0x9a9a0318)  // csel	x24, x24, x26, eq
    KAI_ASM_INST(0xaa1103e7)  // mov	x7, x17
    KAI_ASM_INST(0x91004026)  // add	x6, x1, #0x10
    KAI_ASM_INST(0x91008025)  // add	x5, x1, #0x20
    KAI_ASM_INST(0xaa1103ef)  // mov	x15, x17
    KAI_ASM_INST(0x9100c022)  // add	x2, x1, #0x30
    KAI_ASM_INST(0x91004339)  // add	x25, x25, #0x10
    KAI_ASM_INST(0xd2800017)  // mov	x23, #0x0
    KAI_ASM_INST(0xd280000c)  // mov	x12, #0x0
    KAI_ASM_INST(0xa90be7ee)  // stp	x14, x25, [sp, #184]
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xf10020ff)  // cmp	x7, #0x8
    KAI_ASM_INST(0xd280010a)  // mov	x10, #0x8
    KAI_ASM_INST(0x9a8a90f9)  // csel	x25, x7, x10, ls
    KAI_ASM_INST(0xf94063ea)  // ldr	x10, [sp, #192]
    KAI_ASM_INST(0xf100133f)  // cmp	x25, #0x4
    KAI_ASM_INST(0xaa1303f1)  // mov	x17, x19
    KAI_ASM_INST(0xaa1003eb)  // mov	x11, x16
    KAI_ASM_INST(0xaa1903ee)  // mov	x14, x25
    KAI_ASM_INST(0x8b100155)  // add	x21, x10, x16
    KAI_ASM_INST(0xd280000a)  // mov	x10, #0x0
    KAI_ASM_INST(0xa90ac3ec)  // stp	x12, x16, [sp, #168]
    KAI_ASM_INST(0x9a9a9336)  // csel	x22, x25, x26, ls
    KAI_ASM_INST(0x25b61fe3)  // whilelo	p3.s, xzr, x22
    KAI_ASM_INST(0x91000559)  // add	x25, x10, #0x1
    KAI_ASM_INST(0xb9400230)  // ldr	w16, [x17]
    KAI_ASM_INST(0x5280000c)  // mov	w12, #0x0
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54000049)  // b.ls	124 <MlasGemmU8X8KernelUmmlaSveImpl+0x124>
    KAI_ASM_INST(0xb940062c)  // ldr	w12, [x17, #4]
    KAI_ASM_INST(0x4e040e04)  // dup	v4.4s, w16
    KAI_ASM_INST(0x4e141d84)  // mov	v4.s[2], w12
    KAI_ASM_INST(0x4e1c1d84)  // mov	v4.s[3], w12
    KAI_ASM_INST(0xb400d43c)  // cbz	x28, 1bb4 <MlasGemmU8X8KernelUmmlaSveImpl+0x1bb4>
    KAI_ASM_INST(0x29404936)  // ldp	w22, w18, [x9]
    KAI_ASM_INST(0x2d40848d)  // ldp	s13, s1, [x4, #4]
    KAI_ASM_INST(0x2d41808b)  // ldp	s11, s0, [x4, #12]
    KAI_ASM_INST(0x1b167e1b)  // mul	w27, w16, w22
    KAI_ASM_INST(0x2d413126)  // ldp	s6, s12, [x9, #8]
    KAI_ASM_INST(0x1b127e10)  // mul	w16, w16, w18
    KAI_ASM_INST(0x2d422925)  // ldp	s5, s10, [x9, #16]
    KAI_ASM_INST(0x1e270362)  // fmov	s2, w27
    KAI_ASM_INST(0x2d432123)  // ldp	s3, s8, [x9, #24]
    KAI_ASM_INST(0x4e0c1e02)  // mov	v2.s[1], w16
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd401489)  // ldr	s9, [x4, #20]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x1b0c7ed6)  // mul	w22, w22, w12
    KAI_ASM_INST(0x4e0404c6)  // dup	v6.4s, v6.s[0]
    KAI_ASM_INST(0x4d40c887)  // ld1r	{v7.4s}, [x4]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0x1b0c7e52)  // mul	w18, w18, w12
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c05a7)  // mov	v7.s[1], v13.s[0]
    KAI_ASM_INST(0x6e0c0561)  // mov	v1.s[1], v11.s[0]
    KAI_ASM_INST(0x6e0c0520)  // mov	v0.s[1], v9.s[0]
    KAI_ASM_INST(0x4e141ec2)  // mov	v2.s[2], w22
    KAI_ASM_INST(0x6e0c0586)  // mov	v6.s[1], v12.s[0]
    KAI_ASM_INST(0x6e0c0545)  // mov	v5.s[1], v10.s[0]
    KAI_ASM_INST(0x6e0c0503)  // mov	v3.s[1], v8.s[0]
    KAI_ASM_INST(0x6e1c05a7)  // mov	v7.s[3], v13.s[0]
    KAI_ASM_INST(0x6e1c0561)  // mov	v1.s[3], v11.s[0]
    KAI_ASM_INST(0x6e1c0520)  // mov	v0.s[3], v9.s[0]
    KAI_ASM_INST(0x4e1c1e42)  // mov	v2.s[3], w18
    KAI_ASM_INST(0x6e1c0586)  // mov	v6.s[3], v12.s[0]
    KAI_ASM_INST(0x6e1c0545)  // mov	v5.s[3], v10.s[0]
    KAI_ASM_INST(0x6e1c0503)  // mov	v3.s[3], v8.s[0]
    KAI_ASM_INST(0x4ea78442)  // add	v2.4s, v2.4s, v7.4s
    KAI_ASM_INST(0x4ea494c1)  // mla	v1.4s, v6.4s, v4.4s
    KAI_ASM_INST(0x4ea494a0)  // mla	v0.4s, v5.4s, v4.4s
    KAI_ASM_INST(0x4ea39c84)  // mul	v4.4s, v4.4s, v3.4s
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0x9104c3ec)  // add	x12, sp, #0x130
    KAI_ASM_INST(0xa540a982)  // ld1w	{z2.s}, p2/z, [x12]
    KAI_ASM_INST(0xbd401c85)  // ldr	s5, [x4, #28]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a983)  // ld1w	{z3.s}, p2/z, [x12]
    KAI_ASM_INST(0xd37df150)  // lsl	x16, x10, #3
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a981)  // ld1w	{z1.s}, p2/z, [x12]
    KAI_ASM_INST(0xbd401880)  // ldr	s0, [x4, #24]
    KAI_ASM_INST(0x9104c3ec)  // add	x12, sp, #0x130
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c04a0)  // mov	v0.s[1], v5.s[0]
    KAI_ASM_INST(0x6e1c04a0)  // mov	v0.s[3], v5.s[0]
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a984)  // ld1w	{z4.s}, p2/z, [x12]
    KAI_ASM_INST(0xb4000263)  // cbz	x3, 258 <MlasGemmU8X8KernelUmmlaSveImpl+0x258>
    KAI_ASM_INST(0x8b1001b0)  // add	x16, x13, x16
    KAI_ASM_INST(0xaa1703ec)  // mov	x12, x23
    KAI_ASM_INST(0xd2800012)  // mov	x18, #0x0
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xa40c4026)  // ld1b	{z6.b}, p0/z, [x1, x12]
    KAI_ASM_INST(0xa40c40c5)  // ld1b	{z5.b}, p0/z, [x6, x12]
    KAI_ASM_INST(0x91000652)  // add	x18, x18, #0x1
    KAI_ASM_INST(0xa400a600)  // ld1b	{z0.b}, p1/z, [x16]
    KAI_ASM_INST(0x8b080210)  // add	x16, x16, x8
    KAI_ASM_INST(0x45c69802)  // ummla	z2.s, z0.b, z6.b
    KAI_ASM_INST(0x45c59803)  // ummla	z3.s, z0.b, z5.b
    KAI_ASM_INST(0xa40c40a6)  // ld1b	{z6.b}, p0/z, [x5, x12]
    KAI_ASM_INST(0xa40c4045)  // ld1b	{z5.b}, p0/z, [x2, x12]
    KAI_ASM_INST(0x45c69801)  // ummla	z1.s, z0.b, z6.b
    KAI_ASM_INST(0x9101018c)  // add	x12, x12, #0x40
    KAI_ASM_INST(0x45c59804)  // ummla	z4.s, z0.b, z5.b
    KAI_ASM_INST(0xeb12007f)  // cmp	x3, x18
    KAI_ASM_INST(0x54fffe61)  // b.ne	220 <MlasGemmU8X8KernelUmmlaSveImpl+0x220>
    KAI_ASM_INST(0x05e46c20)  // uzp2	z0.d, z1.d, z4.d
    KAI_ASM_INST(0x05e46821)  // uzp1	z1.d, z1.d, z4.d
    KAI_ASM_INST(0x04603005)  // mov	z5.d, z0.d
    KAI_ASM_INST(0x05e36844)  // uzp1	z4.d, z2.d, z3.d
    KAI_ASM_INST(0x05e36c42)  // uzp2	z2.d, z2.d, z3.d
    KAI_ASM_INST(0x3500c8be)  // cbnz	w30, 1b80 <MlasGemmU8X8KernelUmmlaSveImpl+0x1b80>
    KAI_ASM_INST(0xa540ad63)  // ld1w	{z3.s}, p3/z, [x11]
    KAI_ASM_INST(0x04a40063)  // add	z3.s, z3.s, z4.s
    KAI_ASM_INST(0xe540ed63)  // st1w	{z3.s}, p3, [x11]
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0x5400cc49)  // b.ls	1c08 <MlasGemmU8X8KernelUmmlaSveImpl+0x1c08>
    KAI_ASM_INST(0xd10011cc)  // sub	x12, x14, #0x4
    KAI_ASM_INST(0x25ac1fe4)  // whilelo	p4.s, xzr, x12
    KAI_ASM_INST(0x9100416c)  // add	x12, x11, #0x10
    KAI_ASM_INST(0xa540b183)  // ld1w	{z3.s}, p4/z, [x12]
    KAI_ASM_INST(0x04a10061)  // add	z1.s, z3.s, z1.s
    KAI_ASM_INST(0xe540f181)  // st1w	{z1.s}, p4, [x12]
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x5400cc28)  // b.hi	1c24 <MlasGemmU8X8KernelUmmlaSveImpl+0x1c24>
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0x9100094a)  // add	x10, x10, #0x2
    KAI_ASM_INST(0x91002231)  // add	x17, x17, #0x8
    KAI_ASM_INST(0x8b1402b5)  // add	x21, x21, x20
    KAI_ASM_INST(0x8b14016b)  // add	x11, x11, x20
    KAI_ASM_INST(0xeb18015f)  // cmp	x10, x24
    KAI_ASM_INST(0x54fff281)  // b.ne	10c <MlasGemmU8X8KernelUmmlaSveImpl+0x10c>
    KAI_ASM_INST(0xa94ac3ec)  // ldp	x12, x16, [sp, #168]
    KAI_ASM_INST(0xd10020e7)  // sub	x7, x7, #0x8
    KAI_ASM_INST(0xf9405fea)  // ldr	x10, [sp, #184]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0x91008129)  // add	x9, x9, #0x20
    KAI_ASM_INST(0x8b0a02f7)  // add	x23, x23, x10
    KAI_ASM_INST(0x9100218c)  // add	x12, x12, #0x8
    KAI_ASM_INST(0x91008210)  // add	x16, x16, #0x20
    KAI_ASM_INST(0xeb0c01ff)  // cmp	x15, x12
    KAI_ASM_INST(0x54ffefa8)  // b.hi	d8 <MlasGemmU8X8KernelUmmlaSveImpl+0xd8>
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0x6d093fee)  // stp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xd2800009)  // mov	x9, #0x0
    KAI_ASM_INST(0xf1002cbf)  // cmp	x5, #0xb
    KAI_ASM_INST(0x54003909)  // b.ls	a3c <MlasGemmU8X8KernelUmmlaSveImpl+0xa3c>
    KAI_ASM_INST(0x8b0604c7)  // add	x7, x6, x6, lsl #1
    KAI_ASM_INST(0xcb060bea)  // neg	x10, x6, lsl #2
    KAI_ASM_INST(0x910004c5)  // add	x5, x6, #0x1
    KAI_ASM_INST(0xd37ef4c2)  // lsl	x2, x6, #2
    KAI_ASM_INST(0xd37df0e9)  // lsl	x9, x7, #3
    KAI_ASM_INST(0x8b070a08)  // add	x8, x16, x7, lsl #2
    KAI_ASM_INST(0x8b09021e)  // add	x30, x16, x9
    KAI_ASM_INST(0xd37df0c0)  // lsl	x0, x6, #3
    KAI_ASM_INST(0xcb060fc4)  // sub	x4, x30, x6, lsl #3
    KAI_ASM_INST(0xd37ceca5)  // lsl	x5, x5, #4
    KAI_ASM_INST(0x8b0a0096)  // add	x22, x4, x10
    KAI_ASM_INST(0x8b060058)  // add	x24, x2, x6
    KAI_ASM_INST(0x8b0a02db)  // add	x27, x22, x10
    KAI_ASM_INST(0x91004108)  // add	x8, x8, #0x10
    KAI_ASM_INST(0x8b0a036c)  // add	x12, x27, x10
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x910040aa)  // add	x10, x5, #0x10
    KAI_ASM_INST(0x91004007)  // add	x7, x0, #0x10
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0x8b050205)  // add	x5, x16, x5
    KAI_ASM_INST(0xa90d17e8)  // stp	x8, x5, [sp, #208]
    KAI_ASM_INST(0x91008188)  // add	x8, x12, #0x20
    KAI_ASM_INST(0x9100412b)  // add	x11, x9, #0x10
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0x91008129)  // add	x9, x9, #0x20
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0x8b070202)  // add	x2, x16, x7
    KAI_ASM_INST(0xf90087e8)  // str	x8, [sp, #264]
    KAI_ASM_INST(0x910082c8)  // add	x8, x22, #0x20
    KAI_ASM_INST(0xf90067e2)  // str	x2, [sp, #200]
    KAI_ASM_INST(0x91004202)  // add	x2, x16, #0x10
    KAI_ASM_INST(0xa9110be8)  // stp	x8, x2, [sp, #272]
    KAI_ASM_INST(0x8b090208)  // add	x8, x16, x9
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xcb060017)  // sub	x23, x0, x6
    KAI_ASM_INST(0xb940fbe2)  // ldr	w2, [sp, #248]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005fe8)  // str	x8, [sp, #184]
    KAI_ASM_INST(0x8b0a0208)  // add	x8, x16, x10
    KAI_ASM_INST(0xd2800009)  // mov	x9, #0x0
    KAI_ASM_INST(0x8b0b020b)  // add	x11, x16, x11
    KAI_ASM_INST(0x8b000200)  // add	x0, x16, x0
    KAI_ASM_INST(0xf9005be8)  // str	x8, [sp, #176]
    KAI_ASM_INST(0x12000042)  // and	w2, w2, #0x1
    KAI_ASM_INST(0xf940a3e8)  // ldr	x8, [sp, #320]
    KAI_ASM_INST(0xd37ef718)  // lsl	x24, x24, #2
    KAI_ASM_INST(0xd37ef6f7)  // lsl	x23, x23, #2
    KAI_ASM_INST(0xaa1c03ef)  // mov	x15, x28
    KAI_ASM_INST(0x910041b5)  // add	x21, x13, #0x10
    KAI_ASM_INST(0x910081b4)  // add	x20, x13, #0x20
    KAI_ASM_INST(0xaa0b03e7)  // mov	x7, x11
    KAI_ASM_INST(0xaa0403ea)  // mov	x10, x4
    KAI_ASM_INST(0xf90073e9)  // str	x9, [sp, #224]
    KAI_ASM_INST(0xaa0003e9)  // mov	x9, x0
    KAI_ASM_INST(0x9100c1b2)  // add	x18, x13, #0x30
    KAI_ASM_INST(0xb900abe2)  // str	w2, [sp, #168]
    KAI_ASM_INST(0x91008202)  // add	x2, x16, #0x20
    KAI_ASM_INST(0xf90077f2)  // str	x18, [sp, #232]
    KAI_ASM_INST(0xf90083f1)  // str	x17, [sp, #256]
    KAI_ASM_INST(0xf90093e2)  // str	x2, [sp, #288]
    KAI_ASM_INST(0xf90097e6)  // str	x6, [sp, #296]
    KAI_ASM_INST(0xf900a7fc)  // str	x28, [sp, #328]
    KAI_ASM_INST(0xf94073f2)  // ldr	x18, [sp, #224]
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0x91001245)  // add	x5, x18, #0x4
    KAI_ASM_INST(0x91002a42)  // add	x2, x18, #0xa
    KAI_ASM_INST(0x91000a46)  // add	x6, x18, #0x2
    KAI_ASM_INST(0x9100224b)  // add	x11, x18, #0x8
    KAI_ASM_INST(0xd343fcba)  // lsr	x26, x5, #3
    KAI_ASM_INST(0xd37d08a5)  // ubfiz	x5, x5, #3, #3
    KAI_ASM_INST(0x4d40ca68)  // ld1r	{v8.4s}, [x19]
    KAI_ASM_INST(0xd343fcdc)  // lsr	x28, x6, #3
    KAI_ASM_INST(0xd37d08c6)  // ubfiz	x6, x6, #3, #3
    KAI_ASM_INST(0x91001a44)  // add	x4, x18, #0x6
    KAI_ASM_INST(0x9b0e1745)  // madd	x5, x26, x14, x5
    KAI_ASM_INST(0xd37d0a51)  // ubfiz	x17, x18, #3, #3
    KAI_ASM_INST(0xf9007be5)  // str	x5, [sp, #240]
    KAI_ASM_INST(0xd343fc45)  // lsr	x5, x2, #3
    KAI_ASM_INST(0xd37d0842)  // ubfiz	x2, x2, #3, #3
    KAI_ASM_INST(0xaa1203e0)  // mov	x0, x18
    KAI_ASM_INST(0xd343fd6b)  // lsr	x11, x11, #3
    KAI_ASM_INST(0xd343fc99)  // lsr	x25, x4, #3
    KAI_ASM_INST(0x6e140408)  // mov	v8.s[2], v0.s[0]
    KAI_ASM_INST(0x9b0e1b9c)  // madd	x28, x28, x14, x6
    KAI_ASM_INST(0x9b0e08a2)  // madd	x2, x5, x14, x2
    KAI_ASM_INST(0xd37d0884)  // ubfiz	x4, x4, #3, #3
    KAI_ASM_INST(0xf940a7e5)  // ldr	x5, [sp, #328]
    KAI_ASM_INST(0xd343fc06)  // lsr	x6, x0, #3
    KAI_ASM_INST(0x91003252)  // add	x18, x18, #0xc
    KAI_ASM_INST(0x9b0e456b)  // madd	x11, x11, x14, x17
    KAI_ASM_INST(0xf90073f2)  // str	x18, [sp, #224]
    KAI_ASM_INST(0x9b0e1324)  // madd	x4, x25, x14, x4
    KAI_ASM_INST(0x6e1c0408)  // mov	v8.s[3], v0.s[0]
    KAI_ASM_INST(0x9b0e44c6)  // madd	x6, x6, x14, x17
    KAI_ASM_INST(0xaa0b03f2)  // mov	x18, x11
    KAI_ASM_INST(0xb4006385)  // cbz	x5, 110c <MlasGemmU8X8KernelUmmlaSveImpl+0x110c>
    KAI_ASM_INST(0x2d409110)  // ldp	s16, s4, [x8, #4]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d41950e)  // ldp	s14, s5, [x8, #12]
    KAI_ASM_INST(0x2d428d0c)  // ldp	s12, s3, [x8, #20]
    KAI_ASM_INST(0x2d40e5f1)  // ldp	s17, s25, [x15, #4]
    KAI_ASM_INST(0x2d43890a)  // ldp	s10, s2, [x8, #28]
    KAI_ASM_INST(0x4d40c9e6)  // ld1r	{v6.4s}, [x15]
    KAI_ASM_INST(0x4d40c901)  // ld1r	{v1.4s}, [x8]
    KAI_ASM_INST(0x2d43d9eb)  // ldp	s11, s22, [x15, #28]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0xbd402507)  // ldr	s7, [x8, #36]
    KAI_ASM_INST(0x2d41ddef)  // ldp	s15, s23, [x15, #12]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0xbd402de0)  // ldr	s0, [x15, #44]
    KAI_ASM_INST(0x2d42cded)  // ldp	s13, s19, [x15, #20]
    KAI_ASM_INST(0x6e0c0601)  // mov	v1.s[1], v16.s[0]
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c0626)  // mov	v6.s[1], v17.s[0]
    KAI_ASM_INST(0x2d44c9e9)  // ldp	s9, s18, [x15, #36]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c05c4)  // mov	v4.s[1], v14.s[0]
    KAI_ASM_INST(0x4e040739)  // dup	v25.4s, v25.s[0]
    KAI_ASM_INST(0x4e0406d6)  // dup	v22.4s, v22.s[0]
    KAI_ASM_INST(0x6e0c0585)  // mov	v5.s[1], v12.s[0]
    KAI_ASM_INST(0x4e0406f7)  // dup	v23.4s, v23.s[0]
    KAI_ASM_INST(0x6e1c0601)  // mov	v1.s[3], v16.s[0]
    KAI_ASM_INST(0x6e0c0543)  // mov	v3.s[1], v10.s[0]
    KAI_ASM_INST(0x4e040673)  // dup	v19.4s, v19.s[0]
    KAI_ASM_INST(0x6e1c0626)  // mov	v6.s[3], v17.s[0]
    KAI_ASM_INST(0x6e0c04e2)  // mov	v2.s[1], v7.s[0]
    KAI_ASM_INST(0x4e040652)  // dup	v18.4s, v18.s[0]
    KAI_ASM_INST(0x6e0c0536)  // mov	v22.s[1], v9.s[0]
    KAI_ASM_INST(0x6e1c05c4)  // mov	v4.s[3], v14.s[0]
    KAI_ASM_INST(0x6e0c05f9)  // mov	v25.s[1], v15.s[0]
    KAI_ASM_INST(0x6e1c0585)  // mov	v5.s[3], v12.s[0]
    KAI_ASM_INST(0x6e0c05b7)  // mov	v23.s[1], v13.s[0]
    KAI_ASM_INST(0x4ea11c3a)  // mov	v26.16b, v1.16b
    KAI_ASM_INST(0x6e1c0543)  // mov	v3.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea894c1)  // mla	v1.4s, v6.4s, v8.4s
    KAI_ASM_INST(0x6e0c0573)  // mov	v19.s[1], v11.s[0]
    KAI_ASM_INST(0x6e1c04e2)  // mov	v2.s[3], v7.s[0]
    KAI_ASM_INST(0x6e0c0412)  // mov	v18.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0536)  // mov	v22.s[3], v9.s[0]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b5)  // ld1w	{z21.s}, p1/z, [x5]
    KAI_ASM_INST(0x6e1c05f9)  // mov	v25.s[3], v15.s[0]
    KAI_ASM_INST(0x4ea41c89)  // mov	v9.16b, v4.16b
    KAI_ASM_INST(0x6e1c05b7)  // mov	v23.s[3], v13.s[0]
    KAI_ASM_INST(0x4ea51ca7)  // mov	v7.16b, v5.16b
    KAI_ASM_INST(0x6e1c0573)  // mov	v19.s[3], v11.s[0]
    KAI_ASM_INST(0x4ea31c61)  // mov	v1.16b, v3.16b
    KAI_ASM_INST(0x6e1c0412)  // mov	v18.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea21c40)  // mov	v0.16b, v2.16b
    KAI_ASM_INST(0x4ea89729)  // mla	v9.4s, v25.4s, v8.4s
    KAI_ASM_INST(0x4ea896e7)  // mla	v7.4s, v23.4s, v8.4s
    KAI_ASM_INST(0x4ea89661)  // mla	v1.4s, v19.4s, v8.4s
    KAI_ASM_INST(0x4ea896c0)  // mla	v0.4s, v22.4s, v8.4s
    KAI_ASM_INST(0x4eb29d08)  // mul	v8.4s, v8.4s, v18.4s
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a4b4)  // ld1w	{z20.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4ad)  // ld1w	{z13.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4ac)  // ld1w	{z12.s}, p1/z, [x5]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d452500)  // ldp	s0, s9, [x8, #40]
    KAI_ASM_INST(0x3cc08267)  // ldur	q7, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x4eba1f4b)  // mov	v11.16b, v26.16b
    KAI_ASM_INST(0x4e8738e1)  // zip1	v1.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x4ea41c98)  // mov	v24.16b, v4.16b
    KAI_ASM_INST(0x6e0c0520)  // mov	v0.s[1], v9.s[0]
    KAI_ASM_INST(0x2d433a71)  // ldp	s17, s14, [x19, #24]
    KAI_ASM_INST(0x4ea6942b)  // mla	v11.4s, v1.4s, v6.4s
    KAI_ASM_INST(0x6e1c0520)  // mov	v0.s[3], v9.s[0]
    KAI_ASM_INST(0x4ea51caa)  // mov	v10.16b, v5.16b
    KAI_ASM_INST(0x4eb99438)  // mla	v24.4s, v1.4s, v25.4s
    KAI_ASM_INST(0x4ea31c7c)  // mov	v28.16b, v3.16b
    KAI_ASM_INST(0x4ea08508)  // add	v8.4s, v8.4s, v0.4s
    KAI_ASM_INST(0x4ea21c49)  // mov	v9.16b, v2.16b
    KAI_ASM_INST(0x4ea01c0f)  // mov	v15.16b, v0.16b
    KAI_ASM_INST(0x4e8778e7)  // zip2	v7.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x4eb7942a)  // mla	v10.4s, v1.4s, v23.4s
    KAI_ASM_INST(0x3d804fe8)  // str	q8, [sp, #304]
    KAI_ASM_INST(0xa540a4bf)  // ld1w	{z31.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eba1f5b)  // mov	v27.16b, v26.16b
    KAI_ASM_INST(0x3d804feb)  // str	q11, [sp, #304]
    KAI_ASM_INST(0xa540a4ab)  // ld1w	{z11.s}, p1/z, [x5]
    KAI_ASM_INST(0x4e040631)  // dup	v17.4s, v17.s[0]
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4be)  // ld1w	{z30.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb3943c)  // mla	v28.4s, v1.4s, v19.4s
    KAI_ASM_INST(0x3d804fea)  // str	q10, [sp, #304]
    KAI_ASM_INST(0xa540a4aa)  // ld1w	{z10.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea41c98)  // mov	v24.16b, v4.16b
    KAI_ASM_INST(0x4eb69429)  // mla	v9.4s, v1.4s, v22.4s
    KAI_ASM_INST(0x4ea1964f)  // mla	v15.4s, v18.4s, v1.4s
    KAI_ASM_INST(0x3d804ffc)  // str	q28, [sp, #304]
    KAI_ASM_INST(0xa540a4bd)  // ld1w	{z29.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea51ca1)  // mov	v1.16b, v5.16b
    KAI_ASM_INST(0x4ea694fb)  // mla	v27.4s, v7.4s, v6.4s
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4a9)  // ld1w	{z9.s}, p1/z, [x5]
    KAI_ASM_INST(0x6e1405d1)  // mov	v17.s[2], v14.s[0]
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a4bc)  // ld1w	{z28.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb994f8)  // mla	v24.4s, v7.4s, v25.4s
    KAI_ASM_INST(0x3d804ffb)  // str	q27, [sp, #304]
    KAI_ASM_INST(0xa540a4a8)  // ld1w	{z8.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb794e1)  // mla	v1.4s, v7.4s, v23.4s
    KAI_ASM_INST(0x4ea31c6f)  // mov	v15.16b, v3.16b
    KAI_ASM_INST(0x6e1c05d1)  // mov	v17.s[3], v14.s[0]
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4bb)  // ld1w	{z27.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea21c4e)  // mov	v14.16b, v2.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0x4ea01c18)  // mov	v24.16b, v0.16b
    KAI_ASM_INST(0x4eba1f41)  // mov	v1.16b, v26.16b
    KAI_ASM_INST(0x4eb394ef)  // mla	v15.4s, v7.4s, v19.4s
    KAI_ASM_INST(0x4eb694ee)  // mla	v14.4s, v7.4s, v22.4s
    KAI_ASM_INST(0x4eb294f8)  // mla	v24.4s, v7.4s, v18.4s
    KAI_ASM_INST(0xa540a4a7)  // ld1w	{z7.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb99624)  // mla	v4.4s, v17.4s, v25.4s
    KAI_ASM_INST(0x4ea69621)  // mla	v1.4s, v17.4s, v6.4s
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a4ba)  // ld1w	{z26.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb79625)  // mla	v5.4s, v17.4s, v23.4s
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a4a6)  // ld1w	{z6.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb39623)  // mla	v3.4s, v17.4s, v19.4s
    KAI_ASM_INST(0x3d804ff8)  // str	q24, [sp, #304]
    KAI_ASM_INST(0xa540a4b9)  // ld1w	{z25.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb69622)  // mla	v2.4s, v17.4s, v22.4s
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b3)  // ld1w	{z19.s}, p1/z, [x5]
    KAI_ASM_INST(0x4eb29620)  // mla	v0.4s, v17.4s, v18.4s
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a4b7)  // ld1w	{z23.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a4b2)  // ld1w	{z18.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a4b6)  // ld1w	{z22.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a4b1)  // ld1w	{z17.s}, p1/z, [x5]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4b8)  // ld1w	{z24.s}, p1/z, [x5]
    KAI_ASM_INST(0xb4000623)  // cbz	x3, 7b4 <MlasGemmU8X8KernelUmmlaSveImpl+0x7b4>
    KAI_ASM_INST(0x8b06003a)  // add	x26, x1, x6
    KAI_ASM_INST(0x8b120026)  // add	x6, x1, x18
    KAI_ASM_INST(0xa94e97f2)  // ldp	x18, x5, [sp, #232]
    KAI_ASM_INST(0x8b1c0039)  // add	x25, x1, x28
    KAI_ASM_INST(0x8b04002b)  // add	x11, x1, x4
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0x8b050031)  // add	x17, x1, x5
    KAI_ASM_INST(0x8b020025)  // add	x5, x1, x2
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xa4024344)  // ld1b	{z4.b}, p0/z, [x26, x2]
    KAI_ASM_INST(0xa40241a3)  // ld1b	{z3.b}, p0/z, [x13, x2]
    KAI_ASM_INST(0xa40242a2)  // ld1b	{z2.b}, p0/z, [x21, x2]
    KAI_ASM_INST(0xa4024281)  // ld1b	{z1.b}, p0/z, [x20, x2]
    KAI_ASM_INST(0xa4024240)  // ld1b	{z0.b}, p0/z, [x18, x2]
    KAI_ASM_INST(0x45c49875)  // ummla	z21.s, z3.b, z4.b
    KAI_ASM_INST(0x45c4984b)  // ummla	z11.s, z2.b, z4.b
    KAI_ASM_INST(0x45c49828)  // ummla	z8.s, z1.b, z4.b
    KAI_ASM_INST(0x45c49813)  // ummla	z19.s, z0.b, z4.b
    KAI_ASM_INST(0xa4024224)  // ld1b	{z4.b}, p0/z, [x17, x2]
    KAI_ASM_INST(0xa4024325)  // ld1b	{z5.b}, p0/z, [x25, x2]
    KAI_ASM_INST(0x45c49874)  // ummla	z20.s, z3.b, z4.b
    KAI_ASM_INST(0x45c4984a)  // ummla	z10.s, z2.b, z4.b
    KAI_ASM_INST(0x45c49827)  // ummla	z7.s, z1.b, z4.b
    KAI_ASM_INST(0x45c49812)  // ummla	z18.s, z0.b, z4.b
    KAI_ASM_INST(0x91000484)  // add	x4, x4, #0x1
    KAI_ASM_INST(0xa40240c4)  // ld1b	{z4.b}, p0/z, [x6, x2]
    KAI_ASM_INST(0x45c59870)  // ummla	z16.s, z3.b, z5.b
    KAI_ASM_INST(0x45c5985e)  // ummla	z30.s, z2.b, z5.b
    KAI_ASM_INST(0x45c5983b)  // ummla	z27.s, z1.b, z5.b
    KAI_ASM_INST(0x45c59817)  // ummla	z23.s, z0.b, z5.b
    KAI_ASM_INST(0x45c4986c)  // ummla	z12.s, z3.b, z4.b
    KAI_ASM_INST(0xa4024165)  // ld1b	{z5.b}, p0/z, [x11, x2]
    KAI_ASM_INST(0x45c49849)  // ummla	z9.s, z2.b, z4.b
    KAI_ASM_INST(0x45c49826)  // ummla	z6.s, z1.b, z4.b
    KAI_ASM_INST(0x45c49811)  // ummla	z17.s, z0.b, z4.b
    KAI_ASM_INST(0x45c5986d)  // ummla	z13.s, z3.b, z5.b
    KAI_ASM_INST(0xa40240a4)  // ld1b	{z4.b}, p0/z, [x5, x2]
    KAI_ASM_INST(0x45c5985d)  // ummla	z29.s, z2.b, z5.b
    KAI_ASM_INST(0x91010042)  // add	x2, x2, #0x40
    KAI_ASM_INST(0x45c5983a)  // ummla	z26.s, z1.b, z5.b
    KAI_ASM_INST(0x45c59816)  // ummla	z22.s, z0.b, z5.b
    KAI_ASM_INST(0x45c4987f)  // ummla	z31.s, z3.b, z4.b
    KAI_ASM_INST(0x45c4985c)  // ummla	z28.s, z2.b, z4.b
    KAI_ASM_INST(0x45c49839)  // ummla	z25.s, z1.b, z4.b
    KAI_ASM_INST(0x45c49818)  // ummla	z24.s, z0.b, z4.b
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54fffb61)  // b.ne	718 <MlasGemmU8X8KernelUmmlaSveImpl+0x718>
    KAI_ASM_INST(0xf90077f2)  // str	x18, [sp, #232]
    KAI_ASM_INST(0xb940abe2)  // ldr	w2, [sp, #168]
    KAI_ASM_INST(0x05f06aa0)  // uzp1	z0.d, z21.d, z16.d
    KAI_ASM_INST(0x05ed6a81)  // uzp1	z1.d, z20.d, z13.d
    KAI_ASM_INST(0x05ff6982)  // uzp1	z2.d, z12.d, z31.d
    KAI_ASM_INST(0x05f06eb0)  // uzp2	z16.d, z21.d, z16.d
    KAI_ASM_INST(0x05ed6e8d)  // uzp2	z13.d, z20.d, z13.d
    KAI_ASM_INST(0x05ff6d8c)  // uzp2	z12.d, z12.d, z31.d
    KAI_ASM_INST(0x350043a2)  // cbnz	w2, 1044 <MlasGemmU8X8KernelUmmlaSveImpl+0x1044>
    KAI_ASM_INST(0xd37ef402)  // lsl	x2, x0, #2
    KAI_ASM_INST(0x05fe696e)  // uzp1	z14.d, z11.d, z30.d
    KAI_ASM_INST(0x8b020205)  // add	x5, x16, x2
    KAI_ASM_INST(0xa540a4a3)  // ld1w	{z3.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00060)  // add	z0.s, z3.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa95117e6)  // ldp	x6, x5, [sp, #272]
    KAI_ASM_INST(0x05fd694f)  // uzp1	z15.d, z10.d, z29.d
    KAI_ASM_INST(0x05fe6d6b)  // uzp2	z11.d, z11.d, z30.d
    KAI_ASM_INST(0x05fd6d4a)  // uzp2	z10.d, z10.d, z29.d
    KAI_ASM_INST(0x05fc6934)  // uzp1	z20.d, z9.d, z28.d
    KAI_ASM_INST(0x05fc6d29)  // uzp2	z9.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6905)  // uzp1	z5.d, z8.d, z27.d
    KAI_ASM_INST(0x05fa68e4)  // uzp1	z4.d, z7.d, z26.d
    KAI_ASM_INST(0x8b180204)  // add	x4, x16, x24
    KAI_ASM_INST(0x05fb6d08)  // uzp2	z8.d, z8.d, z27.d
    KAI_ASM_INST(0x05fa6ce7)  // uzp2	z7.d, z7.d, z26.d
    KAI_ASM_INST(0x05f968c3)  // uzp1	z3.d, z6.d, z25.d
    KAI_ASM_INST(0x8b0200a5)  // add	x5, x5, x2
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xf94093e5)  // ldr	x5, [sp, #288]
    KAI_ASM_INST(0xa540a580)  // ld1w	{z0.s}, p1/z, [x12]
    KAI_ASM_INST(0x04b00000)  // add	z0.s, z0.s, z16.s
    KAI_ASM_INST(0xe540e580)  // st1w	{z0.s}, p1, [x12]
    KAI_ASM_INST(0x05f96cc6)  // uzp2	z6.d, z6.d, z25.d
    KAI_ASM_INST(0x8b0200a5)  // add	x5, x5, x2
    KAI_ASM_INST(0x8b0200c2)  // add	x2, x6, x2
    KAI_ASM_INST(0xf94063e6)  // ldr	x6, [sp, #192]
    KAI_ASM_INST(0xa540a4c0)  // ld1w	{z0.s}, p1/z, [x6]
    KAI_ASM_INST(0x04ad0000)  // add	z0.s, z0.s, z13.s
    KAI_ASM_INST(0xe540e4c0)  // st1w	{z0.s}, p1, [x6]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xf94087e5)  // ldr	x5, [sp, #264]
    KAI_ASM_INST(0xa54044a0)  // ld1w	{z0.s}, p1/z, [x5, x0, lsl #2]
    KAI_ASM_INST(0x04ac0000)  // add	z0.s, z0.s, z12.s
    KAI_ASM_INST(0xe54044a0)  // st1w	{z0.s}, p1, [x5, x0, lsl #2]
    KAI_ASM_INST(0xa540a760)  // ld1w	{z0.s}, p1/z, [x27]
    KAI_ASM_INST(0xf94067e5)  // ldr	x5, [sp, #200]
    KAI_ASM_INST(0x04a001ce)  // add	z14.s, z14.s, z0.s
    KAI_ASM_INST(0xe540e76e)  // st1w	{z14.s}, p1, [x27]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a001ef)  // add	z15.s, z15.s, z0.s
    KAI_ASM_INST(0xe540e4af)  // st1w	{z15.s}, p1, [x5]
    KAI_ASM_INST(0xa540a6c0)  // ld1w	{z0.s}, p1/z, [x22]
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0xe540e6cb)  // st1w	{z11.s}, p1, [x22]
    KAI_ASM_INST(0xf9406be5)  // ldr	x5, [sp, #208]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00140)  // add	z0.s, z10.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa540a520)  // ld1w	{z0.s}, p1/z, [x9]
    KAI_ASM_INST(0x04a00280)  // add	z0.s, z20.s, z0.s
    KAI_ASM_INST(0xe540e520)  // st1w	{z0.s}, p1, [x9]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a00120)  // add	z0.s, z9.s, z0.s
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a540)  // ld1w	{z0.s}, p1/z, [x10]
    KAI_ASM_INST(0x04a000a0)  // add	z0.s, z5.s, z0.s
    KAI_ASM_INST(0xf9406fe5)  // ldr	x5, [sp, #216]
    KAI_ASM_INST(0xe540e540)  // st1w	{z0.s}, p1, [x10]
    KAI_ASM_INST(0xa540a4a0)  // ld1w	{z0.s}, p1/z, [x5]
    KAI_ASM_INST(0x04a00080)  // add	z0.s, z4.s, z0.s
    KAI_ASM_INST(0xe540e4a0)  // st1w	{z0.s}, p1, [x5]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04a00100)  // add	z0.s, z8.s, z0.s
    KAI_ASM_INST(0x91004082)  // add	x2, x4, #0x10
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a000e0)  // add	z0.s, z7.s, z0.s
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a00060)  // add	z0.s, z3.s, z0.s
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04a000c6)  // add	z6.s, z6.s, z0.s
    KAI_ASM_INST(0x91008302)  // add	x2, x24, #0x20
    KAI_ASM_INST(0x05f86e21)  // uzp2	z1.d, z17.d, z24.d
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0xe540e446)  // st1w	{z6.s}, p1, [x2]
    KAI_ASM_INST(0xb940abe2)  // ldr	w2, [sp, #168]
    KAI_ASM_INST(0x05f76a63)  // uzp1	z3.d, z19.d, z23.d
    KAI_ASM_INST(0x05f66a42)  // uzp1	z2.d, z18.d, z22.d
    KAI_ASM_INST(0x05f86a31)  // uzp1	z17.d, z17.d, z24.d
    KAI_ASM_INST(0x04613020)  // mov	z0.d, z1.d
    KAI_ASM_INST(0x05f76e73)  // uzp2	z19.d, z19.d, z23.d
    KAI_ASM_INST(0x05f66e52)  // uzp2	z18.d, z18.d, z22.d
    KAI_ASM_INST(0x35003d22)  // cbnz	w2, 10e8 <MlasGemmU8X8KernelUmmlaSveImpl+0x10e8>
    KAI_ASM_INST(0xa540a7c0)  // ld1w	{z0.s}, p1/z, [x30]
    KAI_ASM_INST(0x04a30000)  // add	z0.s, z0.s, z3.s
    KAI_ASM_INST(0xe540e7c0)  // st1w	{z0.s}, p1, [x30]
    KAI_ASM_INST(0xa540a4e0)  // ld1w	{z0.s}, p1/z, [x7]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0x8b170202)  // add	x2, x16, x23
    KAI_ASM_INST(0xe540e4e0)  // st1w	{z0.s}, p1, [x7]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04b30000)  // add	z0.s, z0.s, z19.s
    KAI_ASM_INST(0x91004044)  // add	x4, x2, #0x10
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04b20000)  // add	z0.s, z0.s, z18.s
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xf9405fe4)  // ldr	x4, [sp, #184]
    KAI_ASM_INST(0xa540a480)  // ld1w	{z0.s}, p1/z, [x4]
    KAI_ASM_INST(0x04b10000)  // add	z0.s, z0.s, z17.s
    KAI_ASM_INST(0x91008042)  // add	x2, x2, #0x20
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xa540a440)  // ld1w	{z0.s}, p1/z, [x2]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0x910082e2)  // add	x2, x23, #0x20
    KAI_ASM_INST(0x91006000)  // add	x0, x0, #0x18
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0x9100c3de)  // add	x30, x30, #0x30
    KAI_ASM_INST(0x9100c14a)  // add	x10, x10, #0x30
    KAI_ASM_INST(0x9100c2d6)  // add	x22, x22, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9006fe2)  // str	x2, [sp, #216]
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0x9100c37b)  // add	x27, x27, #0x30
    KAI_ASM_INST(0x9100c18c)  // add	x12, x12, #0x30
    KAI_ASM_INST(0x9100c108)  // add	x8, x8, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9006be2)  // str	x2, [sp, #208]
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0x9100c1ef)  // add	x15, x15, #0x30
    KAI_ASM_INST(0x9100c0e7)  // add	x7, x7, #0x30
    KAI_ASM_INST(0x9100c129)  // add	x9, x9, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf90067e2)  // str	x2, [sp, #200]
    KAI_ASM_INST(0xf94063e2)  // ldr	x2, [sp, #192]
    KAI_ASM_INST(0x9100c318)  // add	x24, x24, #0x30
    KAI_ASM_INST(0x9100c2f7)  // add	x23, x23, #0x30
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0xf9405fe2)  // ldr	x2, [sp, #184]
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9005fe2)  // str	x2, [sp, #184]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0x9100c042)  // add	x2, x2, #0x30
    KAI_ASM_INST(0xf9005be2)  // str	x2, [sp, #176]
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0xeb00005f)  // cmp	x2, x0
    KAI_ASM_INST(0x54ffcf82)  // b.cs	418 <MlasGemmU8X8KernelUmmlaSveImpl+0x418>
    KAI_ASM_INST(0xf94073e9)  // ldr	x9, [sp, #224]
    KAI_ASM_INST(0xaa0203f1)  // mov	x17, x2
    KAI_ASM_INST(0xf94097e6)  // ldr	x6, [sp, #296]
    KAI_ASM_INST(0xf940a7fc)  // ldr	x28, [sp, #328]
    KAI_ASM_INST(0xeb09023f)  // cmp	x17, x9
    KAI_ASM_INST(0x54002ea9)  // b.ls	1014 <MlasGemmU8X8KernelUmmlaSveImpl+0x1014>
    KAI_ASM_INST(0xd37df0c7)  // lsl	x7, x6, #3
    KAI_ASM_INST(0xd37ff8c2)  // lsl	x2, x6, #1
    KAI_ASM_INST(0xcb0600e0)  // sub	x0, x7, x6
    KAI_ASM_INST(0x8b060045)  // add	x5, x2, x6
    KAI_ASM_INST(0x8b090017)  // add	x23, x0, x9
    KAI_ASM_INST(0x91001000)  // add	x0, x0, #0x4
    KAI_ASM_INST(0xcb0602f2)  // sub	x18, x23, x6
    KAI_ASM_INST(0x8b090000)  // add	x0, x0, x9
    KAI_ASM_INST(0xcb06024f)  // sub	x15, x18, x6
    KAI_ASM_INST(0x910010ab)  // add	x11, x5, #0x4
    KAI_ASM_INST(0x8b0608c8)  // add	x8, x6, x6, lsl #2
    KAI_ASM_INST(0x8b09016b)  // add	x11, x11, x9
    KAI_ASM_INST(0x8b000a00)  // add	x0, x16, x0, lsl #2
    KAI_ASM_INST(0x8b050525)  // add	x5, x9, x5, lsl #1
    KAI_ASM_INST(0x91001108)  // add	x8, x8, #0x4
    KAI_ASM_INST(0x910010a5)  // add	x5, x5, #0x4
    KAI_ASM_INST(0xf9005fe0)  // str	x0, [sp, #184]
    KAI_ASM_INST(0x8b0f0a00)  // add	x0, x16, x15, lsl #2
    KAI_ASM_INST(0x910010c6)  // add	x6, x6, #0x4
    KAI_ASM_INST(0x8b090108)  // add	x8, x8, x9
    KAI_ASM_INST(0xf9005be0)  // str	x0, [sp, #176]
    KAI_ASM_INST(0x8b0b0a00)  // add	x0, x16, x11, lsl #2
    KAI_ASM_INST(0xcb0201e4)  // sub	x4, x15, x2
    KAI_ASM_INST(0x8b0900ca)  // add	x10, x6, x9
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0x8b050a00)  // add	x0, x16, x5, lsl #2
    KAI_ASM_INST(0xcb020082)  // sub	x2, x4, x2
    KAI_ASM_INST(0xf90063e0)  // str	x0, [sp, #192]
    KAI_ASM_INST(0x8b080a00)  // add	x0, x16, x8, lsl #2
    KAI_ASM_INST(0x8b07020c)  // add	x12, x16, x7
    KAI_ASM_INST(0xb940fbf4)  // ldr	w20, [sp, #248]
    KAI_ASM_INST(0xd1000626)  // sub	x6, x17, #0x1
    KAI_ASM_INST(0x8b040a19)  // add	x25, x16, x4, lsl #2
    KAI_ASM_INST(0xf9006fe0)  // str	x0, [sp, #216]
    KAI_ASM_INST(0xf940a3e4)  // ldr	x4, [sp, #320]
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0x8b0a0a00)  // add	x0, x16, x10, lsl #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0x8b170a17)  // add	x23, x16, x23, lsl #2
    KAI_ASM_INST(0x8b120a1e)  // add	x30, x16, x18, lsl #2
    KAI_ASM_INST(0x8b020a1a)  // add	x26, x16, x2, lsl #2
    KAI_ASM_INST(0xcb090236)  // sub	x22, x17, x9
    KAI_ASM_INST(0x91001135)  // add	x21, x9, #0x4
    KAI_ASM_INST(0x927ff8cf)  // and	x15, x6, #0xfffffffffffffffe
    KAI_ASM_INST(0x910081ab)  // add	x11, x13, #0x20
    KAI_ASM_INST(0x9100c1aa)  // add	x10, x13, #0x30
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0x8b070180)  // add	x0, x12, x7
    KAI_ASM_INST(0xf9007bec)  // str	x12, [sp, #240]
    KAI_ASM_INST(0x910041ac)  // add	x12, x13, #0x10
    KAI_ASM_INST(0x12000294)  // and	w20, w20, #0x1
    KAI_ASM_INST(0xb900abf4)  // str	w20, [sp, #168]
    KAI_ASM_INST(0xa90e47f0)  // stp	x16, x17, [sp, #224]
    KAI_ASM_INST(0xf9007fe0)  // str	x0, [sp, #248]
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xf10022df)  // cmp	x22, #0x8
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0x4d40ca63)  // ld1r	{v3.4s}, [x19]
    KAI_ASM_INST(0x91000922)  // add	x2, x9, #0x2
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0x9a8092c7)  // csel	x7, x22, x0, ls
    KAI_ASM_INST(0xeb0901ff)  // cmp	x15, x9
    KAI_ASM_INST(0x91001920)  // add	x0, x9, #0x6
    KAI_ASM_INST(0x6e140403)  // mov	v3.s[2], v0.s[0]
    KAI_ASM_INST(0x9a8991e6)  // csel	x6, x15, x9, ls
    KAI_ASM_INST(0xeb0f005f)  // cmp	x2, x15
    KAI_ASM_INST(0x9a8f9042)  // csel	x2, x2, x15, ls
    KAI_ASM_INST(0xeb1501ff)  // cmp	x15, x21
    KAI_ASM_INST(0x9a9591e5)  // csel	x5, x15, x21, ls
    KAI_ASM_INST(0xeb0f001f)  // cmp	x0, x15
    KAI_ASM_INST(0x9a8f9000)  // csel	x0, x0, x15, ls
    KAI_ASM_INST(0xd37d0858)  // ubfiz	x24, x2, #3, #3
    KAI_ASM_INST(0xd343fc5b)  // lsr	x27, x2, #3
    KAI_ASM_INST(0xd343fcd1)  // lsr	x17, x6, #3
    KAI_ASM_INST(0xd37d0812)  // ubfiz	x18, x0, #3, #3
    KAI_ASM_INST(0xd343fc14)  // lsr	x20, x0, #3
    KAI_ASM_INST(0x6e1c0403)  // mov	v3.s[3], v0.s[0]
    KAI_ASM_INST(0xd37d08d0)  // ubfiz	x16, x6, #3, #3
    KAI_ASM_INST(0x9b0e637b)  // madd	x27, x27, x14, x24
    KAI_ASM_INST(0xd343fca8)  // lsr	x8, x5, #3
    KAI_ASM_INST(0xd37d08b8)  // ubfiz	x24, x5, #3, #3
    KAI_ASM_INST(0x9b0e4a94)  // madd	x20, x20, x14, x18
    KAI_ASM_INST(0xd37ef4d2)  // lsl	x18, x6, #2
    KAI_ASM_INST(0x9b0e4230)  // madd	x16, x17, x14, x16
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x91001252)  // add	x18, x18, #0x4
    KAI_ASM_INST(0x9b0e6111)  // madd	x17, x8, x14, x24
    KAI_ASM_INST(0xb4008abc)  // cbz	x28, 1cf4 <MlasGemmU8X8KernelUmmlaSveImpl+0x1cf4>
    KAI_ASM_INST(0xbc667b87)  // ldr	s7, [x28, x6, lsl #2]
    KAI_ASM_INST(0xbc726b80)  // ldr	s0, [x28, x18]
    KAI_ASM_INST(0x4e0404e7)  // dup	v7.4s, v7.s[0]
    KAI_ASM_INST(0x6e0c0407)  // mov	v7.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0407)  // mov	v7.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea39ce0)  // mul	v0.4s, v7.4s, v3.4s
    KAI_ASM_INST(0xbc667884)  // ldr	s4, [x4, x6, lsl #2]
    KAI_ASM_INST(0xd37ef446)  // lsl	x6, x2, #2
    KAI_ASM_INST(0xbc726881)  // ldr	s1, [x4, x18]
    KAI_ASM_INST(0x9104c3e8)  // add	x8, sp, #0x130
    KAI_ASM_INST(0x910010c6)  // add	x6, x6, #0x4
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x6e0c0424)  // mov	v4.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0424)  // mov	v4.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a50b)  // ld1w	{z11.s}, p1/z, [x8]
    KAI_ASM_INST(0xb40088dc)  // cbz	x28, 1d00 <MlasGemmU8X8KernelUmmlaSveImpl+0x1d00>
    KAI_ASM_INST(0xbc627b80)  // ldr	s0, [x28, x2, lsl #2]
    KAI_ASM_INST(0xbc666b81)  // ldr	s1, [x28, x6]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c08)  // mov	v8.16b, v0.16b
    KAI_ASM_INST(0x4ea69c01)  // mul	v1.4s, v0.4s, v6.4s
    KAI_ASM_INST(0xbc627880)  // ldr	s0, [x4, x2, lsl #2]
    KAI_ASM_INST(0xd37ef4a2)  // lsl	x2, x5, #2
    KAI_ASM_INST(0xbc666882)  // ldr	s2, [x4, x6]
    KAI_ASM_INST(0x91001042)  // add	x2, x2, #0x4
    KAI_ASM_INST(0x9104c3e6)  // add	x6, sp, #0x130
    KAI_ASM_INST(0x4e040403)  // dup	v3.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0443)  // mov	v3.s[1], v2.s[0]
    KAI_ASM_INST(0x6e1c0443)  // mov	v3.s[3], v2.s[0]
    KAI_ASM_INST(0x4ea38421)  // add	v1.4s, v1.4s, v3.4s
    KAI_ASM_INST(0x4ea31c69)  // mov	v9.16b, v3.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4d1)  // ld1w	{z17.s}, p1/z, [x6]
    KAI_ASM_INST(0xb400859c)  // cbz	x28, 1ce8 <MlasGemmU8X8KernelUmmlaSveImpl+0x1ce8>
    KAI_ASM_INST(0xbc657b80)  // ldr	s0, [x28, x5, lsl #2]
    KAI_ASM_INST(0xbc626b81)  // ldr	s1, [x28, x2]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c0e)  // mov	v14.16b, v0.16b
    KAI_ASM_INST(0x4ea69c01)  // mul	v1.4s, v0.4s, v6.4s
    KAI_ASM_INST(0xbc657882)  // ldr	s2, [x4, x5, lsl #2]
    KAI_ASM_INST(0xd37ef405)  // lsl	x5, x0, #2
    KAI_ASM_INST(0x4f00042d)  // movi	v13.4s, #0x1
    KAI_ASM_INST(0xbc626880)  // ldr	s0, [x4, x2]
    KAI_ASM_INST(0x910010a2)  // add	x2, x5, #0x4
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c0402)  // mov	v2.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0402)  // mov	v2.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea28421)  // add	v1.4s, v1.4s, v2.4s
    KAI_ASM_INST(0x4ea21c4f)  // mov	v15.16b, v2.16b
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0xb400011c)  // cbz	x28, cac <MlasGemmU8X8KernelUmmlaSveImpl+0xcac>
    KAI_ASM_INST(0xbc607b80)  // ldr	s0, [x28, x0, lsl #2]
    KAI_ASM_INST(0xbc626b81)  // ldr	s1, [x28, x2]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c0420)  // mov	v0.s[1], v1.s[0]
    KAI_ASM_INST(0x6e1c0420)  // mov	v0.s[3], v1.s[0]
    KAI_ASM_INST(0x4ea01c0d)  // mov	v13.16b, v0.16b
    KAI_ASM_INST(0x4ea09cc6)  // mul	v6.4s, v6.4s, v0.4s
    KAI_ASM_INST(0xbc607880)  // ldr	s0, [x4, x0, lsl #2]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0xbc626883)  // ldr	s3, [x4, x2]
    KAI_ASM_INST(0x3cc08262)  // ldur	q2, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x2d432a61)  // ldp	s1, s10, [x19, #24]
    KAI_ASM_INST(0x4e823845)  // zip1	v5.4s, v2.4s, v2.4s
    KAI_ASM_INST(0x6e0c0460)  // mov	v0.s[1], v3.s[0]
    KAI_ASM_INST(0x4ea41c93)  // mov	v19.16b, v4.16b
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0x4ea91d32)  // mov	v18.16b, v9.16b
    KAI_ASM_INST(0x6e1c0460)  // mov	v0.s[3], v3.s[0]
    KAI_ASM_INST(0x4ea794b3)  // mla	v19.4s, v5.4s, v7.4s
    KAI_ASM_INST(0x4eaf1df6)  // mov	v22.16b, v15.16b
    KAI_ASM_INST(0x4e827842)  // zip2	v2.4s, v2.4s, v2.4s
    KAI_ASM_INST(0x4ea084c3)  // add	v3.4s, v6.4s, v0.4s
    KAI_ASM_INST(0x4ea01c14)  // mov	v20.16b, v0.16b
    KAI_ASM_INST(0x4ea41c97)  // mov	v23.16b, v4.16b
    KAI_ASM_INST(0x6e140541)  // mov	v1.s[2], v10.s[0]
    KAI_ASM_INST(0x4ea59512)  // mla	v18.4s, v8.4s, v5.4s
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a40c)  // ld1w	{z12.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea91d35)  // mov	v21.16b, v9.16b
    KAI_ASM_INST(0x3d804ff3)  // str	q19, [sp, #304]
    KAI_ASM_INST(0xa540a406)  // ld1w	{z6.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea595b4)  // mla	v20.4s, v13.4s, v5.4s
    KAI_ASM_INST(0x3d804ff2)  // str	q18, [sp, #304]
    KAI_ASM_INST(0xa540a412)  // ld1w	{z18.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea595d6)  // mla	v22.4s, v14.4s, v5.4s
    KAI_ASM_INST(0x4eaf1df3)  // mov	v19.16b, v15.16b
    KAI_ASM_INST(0x4ea01c05)  // mov	v5.16b, v0.16b
    KAI_ASM_INST(0x6e1c0541)  // mov	v1.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea79457)  // mla	v23.4s, v2.4s, v7.4s
    KAI_ASM_INST(0x3d804ff6)  // str	q22, [sp, #304]
    KAI_ASM_INST(0xa540a416)  // ld1w	{z22.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea89455)  // mla	v21.4s, v2.4s, v8.4s
    KAI_ASM_INST(0x3d804ff4)  // str	q20, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea91d23)  // mov	v3.16b, v9.16b
    KAI_ASM_INST(0x3d804ff7)  // str	q23, [sp, #304]
    KAI_ASM_INST(0xa540a40a)  // ld1w	{z10.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ead9445)  // mla	v5.4s, v2.4s, v13.4s
    KAI_ASM_INST(0x3d804ff5)  // str	q21, [sp, #304]
    KAI_ASM_INST(0x4eae9453)  // mla	v19.4s, v2.4s, v14.4s
    KAI_ASM_INST(0x4eaf1de2)  // mov	v2.16b, v15.16b
    KAI_ASM_INST(0x4ea79424)  // mla	v4.4s, v1.4s, v7.4s
    KAI_ASM_INST(0x4ea89423)  // mla	v3.4s, v1.4s, v8.4s
    KAI_ASM_INST(0xa540a408)  // ld1w	{z8.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ead9420)  // mla	v0.4s, v1.4s, v13.4s
    KAI_ASM_INST(0x3d804ff3)  // str	q19, [sp, #304]
    KAI_ASM_INST(0xa540a415)  // ld1w	{z21.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eae9422)  // mla	v2.4s, v1.4s, v14.4s
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a409)  // ld1w	{z9.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a407)  // ld1w	{z7.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a40d)  // ld1w	{z13.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a417)  // ld1w	{z23.s}, p1/z, [x0]
    KAI_ASM_INST(0xb4000463)  // cbz	x3, e30 <MlasGemmU8X8KernelUmmlaSveImpl+0xe30>
    KAI_ASM_INST(0x8b100030)  // add	x16, x1, x16
    KAI_ASM_INST(0x8b1b0028)  // add	x8, x1, x27
    KAI_ASM_INST(0x8b110026)  // add	x6, x1, x17
    KAI_ASM_INST(0x8b140025)  // add	x5, x1, x20
    KAI_ASM_INST(0xd2800000)  // mov	x0, #0x0
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xa40041a0)  // ld1b	{z0.b}, p0/z, [x13, x0]
    KAI_ASM_INST(0xa4004204)  // ld1b	{z4.b}, p0/z, [x16, x0]
    KAI_ASM_INST(0xa4004103)  // ld1b	{z3.b}, p0/z, [x8, x0]
    KAI_ASM_INST(0xa40040c2)  // ld1b	{z2.b}, p0/z, [x6, x0]
    KAI_ASM_INST(0xa40040a1)  // ld1b	{z1.b}, p0/z, [x5, x0]
    KAI_ASM_INST(0x45c4980b)  // ummla	z11.s, z0.b, z4.b
    KAI_ASM_INST(0x45c39811)  // ummla	z17.s, z0.b, z3.b
    KAI_ASM_INST(0x45c29810)  // ummla	z16.s, z0.b, z2.b
    KAI_ASM_INST(0x45c1980c)  // ummla	z12.s, z0.b, z1.b
    KAI_ASM_INST(0x91000442)  // add	x2, x2, #0x1
    KAI_ASM_INST(0xa4004160)  // ld1b	{z0.b}, p0/z, [x11, x0]
    KAI_ASM_INST(0xa4004185)  // ld1b	{z5.b}, p0/z, [x12, x0]
    KAI_ASM_INST(0x45c4980a)  // ummla	z10.s, z0.b, z4.b
    KAI_ASM_INST(0x45c39808)  // ummla	z8.s, z0.b, z3.b
    KAI_ASM_INST(0x45c29815)  // ummla	z21.s, z0.b, z2.b
    KAI_ASM_INST(0x45c19813)  // ummla	z19.s, z0.b, z1.b
    KAI_ASM_INST(0x45c498a6)  // ummla	z6.s, z5.b, z4.b
    KAI_ASM_INST(0xa4004140)  // ld1b	{z0.b}, p0/z, [x10, x0]
    KAI_ASM_INST(0x45c398b2)  // ummla	z18.s, z5.b, z3.b
    KAI_ASM_INST(0x91010000)  // add	x0, x0, #0x40
    KAI_ASM_INST(0x45c298b6)  // ummla	z22.s, z5.b, z2.b
    KAI_ASM_INST(0x45c198b4)  // ummla	z20.s, z5.b, z1.b
    KAI_ASM_INST(0x45c49809)  // ummla	z9.s, z0.b, z4.b
    KAI_ASM_INST(0x45c39807)  // ummla	z7.s, z0.b, z3.b
    KAI_ASM_INST(0x45c2980d)  // ummla	z13.s, z0.b, z2.b
    KAI_ASM_INST(0x45c19817)  // ummla	z23.s, z0.b, z1.b
    KAI_ASM_INST(0xeb02007f)  // cmp	x3, x2
    KAI_ASM_INST(0x54fffca1)  // b.ne	dc0 <MlasGemmU8X8KernelUmmlaSveImpl+0xdc0>
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x9a8090e0)  // csel	x0, x7, x0, ls
    KAI_ASM_INST(0x05f16960)  // uzp1	z0.d, z11.d, z17.d
    KAI_ASM_INST(0x05ec6a02)  // uzp1	z2.d, z16.d, z12.d
    KAI_ASM_INST(0x05f16d6b)  // uzp2	z11.d, z11.d, z17.d
    KAI_ASM_INST(0x25a01fe2)  // whilelo	p2.s, xzr, x0
    KAI_ASM_INST(0xb940abe0)  // ldr	w0, [sp, #168]
    KAI_ASM_INST(0x05ec6e0c)  // uzp2	z12.d, z16.d, z12.d
    KAI_ASM_INST(0x35006f80)  // cbnz	w0, 1c44 <MlasGemmU8X8KernelUmmlaSveImpl+0x1c44>
    KAI_ASM_INST(0xf94073e6)  // ldr	x6, [sp, #224]
    KAI_ASM_INST(0xd37ef520)  // lsl	x0, x9, #2
    KAI_ASM_INST(0x8b0000c2)  // add	x2, x6, x0
    KAI_ASM_INST(0xa540a841)  // ld1w	{z1.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a00020)  // add	z0.s, z1.s, z0.s
    KAI_ASM_INST(0xe540e840)  // st1w	{z0.s}, p2, [x2]
    KAI_ASM_INST(0xf10012df)  // cmp	x22, #0x4
    KAI_ASM_INST(0x540074c9)  // b.ls	1d0c <MlasGemmU8X8KernelUmmlaSveImpl+0x1d0c>
    KAI_ASM_INST(0xd10010e7)  // sub	x7, x7, #0x4
    KAI_ASM_INST(0x25a71fe3)  // whilelo	p3.s, xzr, x7
    KAI_ASM_INST(0xa94f23e7)  // ldp	x7, x8, [sp, #240]
    KAI_ASM_INST(0x05e76d20)  // uzp2	z0.d, z9.d, z7.d
    KAI_ASM_INST(0xd37ef6a2)  // lsl	x2, x21, #2
    KAI_ASM_INST(0x05f268d0)  // uzp1	z16.d, z6.d, z18.d
    KAI_ASM_INST(0xaa0203e5)  // mov	x5, x2
    KAI_ASM_INST(0x8b0200c2)  // add	x2, x6, x2
    KAI_ASM_INST(0xa540ac51)  // ld1w	{z17.s}, p3/z, [x2]
    KAI_ASM_INST(0x04a20231)  // add	z17.s, z17.s, z2.s
    KAI_ASM_INST(0x05e76922)  // uzp1	z2.d, z9.d, z7.d
    KAI_ASM_INST(0xe540ec51)  // st1w	{z17.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab47)  // ld1w	{z7.s}, p2/z, [x26]
    KAI_ASM_INST(0x04ab00e7)  // add	z7.s, z7.s, z11.s
    KAI_ASM_INST(0x8b0000e2)  // add	x2, x7, x0
    KAI_ASM_INST(0x8b000106)  // add	x6, x8, x0
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0xe540eb47)  // st1w	{z7.s}, p2, [x26]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04ac00e7)  // add	z7.s, z7.s, z12.s
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540a847)  // ld1w	{z7.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a70207)  // add	z7.s, z16.s, z7.s
    KAI_ASM_INST(0x8b0500e0)  // add	x0, x7, x5
    KAI_ASM_INST(0x05f46acf)  // uzp1	z15.d, z22.d, z20.d
    KAI_ASM_INST(0xe540e847)  // st1w	{z7.s}, p2, [x2]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a701e7)  // add	z7.s, z15.s, z7.s
    KAI_ASM_INST(0x05f26cce)  // uzp2	z14.d, z6.d, z18.d
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab27)  // ld1w	{z7.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a701c7)  // add	z7.s, z14.s, z7.s
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x05f46ed4)  // uzp2	z20.d, z22.d, z20.d
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xa540ac07)  // ld1w	{z7.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a70287)  // add	z7.s, z20.s, z7.s
    KAI_ASM_INST(0x05e86946)  // uzp1	z6.d, z10.d, z8.d
    KAI_ASM_INST(0xe540ec07)  // st1w	{z7.s}, p3, [x0]
    KAI_ASM_INST(0xa540a8c7)  // ld1w	{z7.s}, p2/z, [x6]
    KAI_ASM_INST(0x04a700c6)  // add	z6.s, z6.s, z7.s
    KAI_ASM_INST(0x8b050105)  // add	x5, x8, x5
    KAI_ASM_INST(0x05f36aa5)  // uzp1	z5.d, z21.d, z19.d
    KAI_ASM_INST(0xe540e8c6)  // st1w	{z6.s}, p2, [x6]
    KAI_ASM_INST(0xa540aca6)  // ld1w	{z6.s}, p3/z, [x5]
    KAI_ASM_INST(0x04a600a5)  // add	z5.s, z5.s, z6.s
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86d44)  // uzp2	z4.d, z10.d, z8.d
    KAI_ASM_INST(0xe540eca5)  // st1w	{z5.s}, p3, [x5]
    KAI_ASM_INST(0xa540a805)  // ld1w	{z5.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a50084)  // add	z4.s, z4.s, z5.s
    KAI_ASM_INST(0x05f36ea3)  // uzp2	z3.d, z21.d, z19.d
    KAI_ASM_INST(0xe540e804)  // st1w	{z4.s}, p2, [x0]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0xa540ac04)  // ld1w	{z4.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a40063)  // add	z3.s, z3.s, z4.s
    KAI_ASM_INST(0xe540ec03)  // st1w	{z3.s}, p3, [x0]
    KAI_ASM_INST(0xa540abc3)  // ld1w	{z3.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0x05f769a1)  // uzp1	z1.d, z13.d, z23.d
    KAI_ASM_INST(0xe540ebc2)  // st1w	{z2.s}, p2, [x30]
    KAI_ASM_INST(0xa540ac02)  // ld1w	{z2.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540ec01)  // st1w	{z1.s}, p3, [x0]
    KAI_ASM_INST(0xa540aae1)  // ld1w	{z1.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0x05f76dad)  // uzp2	z13.d, z13.d, z23.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xa540ac00)  // ld1w	{z0.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a001ad)  // add	z13.s, z13.s, z0.s
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540f00d)  // st1w	{z13.s}, p4, [x0]
    KAI_ASM_INST(0x91002129)  // add	x9, x9, #0x8
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xd10022d6)  // sub	x22, x22, #0x8
    KAI_ASM_INST(0x910022b5)  // add	x21, x21, #0x8
    KAI_ASM_INST(0x910082f7)  // add	x23, x23, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005fe0)  // str	x0, [sp, #184]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x910083de)  // add	x30, x30, #0x20
    KAI_ASM_INST(0x91008339)  // add	x25, x25, #0x20
    KAI_ASM_INST(0x9100835a)  // add	x26, x26, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9005be0)  // str	x0, [sp, #176]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9006fe0)  // str	x0, [sp, #216]
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xf90063e0)  // str	x0, [sp, #192]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0xeb09001f)  // cmp	x0, x9
    KAI_ASM_INST(0x54ffd888)  // b.hi	b20 <MlasGemmU8X8KernelUmmlaSveImpl+0xb20>
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x6d493fee)  // ldp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0xf9408fe4)  // ldr	x4, [sp, #280]
    KAI_ASM_INST(0xe5404600)  // st1w	{z0.s}, p1, [x16, x0, lsl #2]
    KAI_ASM_INST(0xe5404481)  // st1w	{z1.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e590)  // st1w	{z16.s}, p1, [x12]
    KAI_ASM_INST(0xf94063e4)  // ldr	x4, [sp, #192]
    KAI_ASM_INST(0xe540e48d)  // st1w	{z13.s}, p1, [x4]
    KAI_ASM_INST(0x05fe6965)  // uzp1	z5.d, z11.d, z30.d
    KAI_ASM_INST(0x05fd694e)  // uzp1	z14.d, z10.d, z29.d
    KAI_ASM_INST(0xf94093e4)  // ldr	x4, [sp, #288]
    KAI_ASM_INST(0xe5404482)  // st1w	{z2.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0x8b180202)  // add	x2, x16, x24
    KAI_ASM_INST(0x05fe6d6b)  // uzp2	z11.d, z11.d, z30.d
    KAI_ASM_INST(0xf94087e4)  // ldr	x4, [sp, #264]
    KAI_ASM_INST(0xe540448c)  // st1w	{z12.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e765)  // st1w	{z5.s}, p1, [x27]
    KAI_ASM_INST(0x05fd6d4a)  // uzp2	z10.d, z10.d, z29.d
    KAI_ASM_INST(0xf94067e4)  // ldr	x4, [sp, #200]
    KAI_ASM_INST(0xe540e48e)  // st1w	{z14.s}, p1, [x4]
    KAI_ASM_INST(0xe540e6cb)  // st1w	{z11.s}, p1, [x22]
    KAI_ASM_INST(0x05fa68e0)  // uzp1	z0.d, z7.d, z26.d
    KAI_ASM_INST(0xf9406be4)  // ldr	x4, [sp, #208]
    KAI_ASM_INST(0xe540e48a)  // st1w	{z10.s}, p1, [x4]
    KAI_ASM_INST(0x05fc6924)  // uzp1	z4.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6903)  // uzp1	z3.d, z8.d, z27.d
    KAI_ASM_INST(0xf9408be4)  // ldr	x4, [sp, #272]
    KAI_ASM_INST(0x05fc6d29)  // uzp2	z9.d, z9.d, z28.d
    KAI_ASM_INST(0x05fb6d08)  // uzp2	z8.d, z8.d, z27.d
    KAI_ASM_INST(0xe540e524)  // st1w	{z4.s}, p1, [x9]
    KAI_ASM_INST(0xe5404489)  // st1w	{z9.s}, p1, [x4, x0, lsl #2]
    KAI_ASM_INST(0xe540e543)  // st1w	{z3.s}, p1, [x10]
    KAI_ASM_INST(0xf9406fe4)  // ldr	x4, [sp, #216]
    KAI_ASM_INST(0xe540e480)  // st1w	{z0.s}, p1, [x4]
    KAI_ASM_INST(0xe540e448)  // st1w	{z8.s}, p1, [x2]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x05f968c0)  // uzp1	z0.d, z6.d, z25.d
    KAI_ASM_INST(0x05fa6ce7)  // uzp2	z7.d, z7.d, z26.d
    KAI_ASM_INST(0x05f96cc6)  // uzp2	z6.d, z6.d, z25.d
    KAI_ASM_INST(0xe540e447)  // st1w	{z7.s}, p1, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xe540e440)  // st1w	{z0.s}, p1, [x2]
    KAI_ASM_INST(0x17fffe0d)  // b	918 <MlasGemmU8X8KernelUmmlaSveImpl+0x918>
    KAI_ASM_INST(0x8b170202)  // add	x2, x16, x23
    KAI_ASM_INST(0xe540e7c3)  // st1w	{z3.s}, p1, [x30]
    KAI_ASM_INST(0xe540e4e2)  // st1w	{z2.s}, p1, [x7]
    KAI_ASM_INST(0xe540e453)  // st1w	{z19.s}, p1, [x2]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0xe540e452)  // st1w	{z18.s}, p1, [x2]
    KAI_ASM_INST(0xf9405fe2)  // ldr	x2, [sp, #184]
    KAI_ASM_INST(0xe540e451)  // st1w	{z17.s}, p1, [x2]
    KAI_ASM_INST(0x17fffe25)  // b	99c <MlasGemmU8X8KernelUmmlaSveImpl+0x99c>
    KAI_ASM_INST(0x2d40910a)  // ldp	s10, s4, [x8, #4]
    KAI_ASM_INST(0x9104c3e5)  // add	x5, sp, #0x130
    KAI_ASM_INST(0x2d419509)  // ldp	s9, s5, [x8, #12]
    KAI_ASM_INST(0x2d428d07)  // ldp	s7, s3, [x8, #20]
    KAI_ASM_INST(0x2d438906)  // ldp	s6, s2, [x8, #28]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x4d40c901)  // ld1r	{v1.4s}, [x8]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0xbd402500)  // ldr	s0, [x8, #36]
    KAI_ASM_INST(0x4e040463)  // dup	v3.4s, v3.s[0]
    KAI_ASM_INST(0x6e0c0541)  // mov	v1.s[1], v10.s[0]
    KAI_ASM_INST(0x6e0c0524)  // mov	v4.s[1], v9.s[0]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x6e0c04e5)  // mov	v5.s[1], v7.s[0]
    KAI_ASM_INST(0x6e1c0541)  // mov	v1.s[3], v10.s[0]
    KAI_ASM_INST(0x6e0c04c3)  // mov	v3.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c0402)  // mov	v2.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c0524)  // mov	v4.s[3], v9.s[0]
    KAI_ASM_INST(0x6e1c04e5)  // mov	v5.s[3], v7.s[0]
    KAI_ASM_INST(0x6e1c04c3)  // mov	v3.s[3], v6.s[0]
    KAI_ASM_INST(0x4ea88429)  // add	v9.4s, v1.4s, v8.4s
    KAI_ASM_INST(0x4ea88486)  // add	v6.4s, v4.4s, v8.4s
    KAI_ASM_INST(0x6e1c0402)  // mov	v2.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea884a7)  // add	v7.4s, v5.4s, v8.4s
    KAI_ASM_INST(0x4ea11c3a)  // mov	v26.16b, v1.16b
    KAI_ASM_INST(0x3d804fe9)  // str	q9, [sp, #304]
    KAI_ASM_INST(0xa540a4b5)  // ld1w	{z21.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea38501)  // add	v1.4s, v8.4s, v3.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a4b0)  // ld1w	{z16.s}, p1/z, [x5]
    KAI_ASM_INST(0x4f000426)  // movi	v6.4s, #0x1
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a4b4)  // ld1w	{z20.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea28500)  // add	v0.4s, v8.4s, v2.4s
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a4ad)  // ld1w	{z13.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea61cd9)  // mov	v25.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd7)  // mov	v23.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd3)  // mov	v19.16b, v6.16b
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a4ac)  // ld1w	{z12.s}, p1/z, [x5]
    KAI_ASM_INST(0x4ea61cd6)  // mov	v22.16b, v6.16b
    KAI_ASM_INST(0x4ea61cd2)  // mov	v18.16b, v6.16b
    KAI_ASM_INST(0x17fffcfc)  // b	5a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x5a8>
    KAI_ASM_INST(0x8b0e01a9)  // add	x9, x13, x14
    KAI_ASM_INST(0xb4003785)  // cbz	x5, 18b0 <MlasGemmU8X8KernelUmmlaSveImpl+0x18b0>
    KAI_ASM_INST(0x8b0604c7)  // add	x7, x6, x6, lsl #1
    KAI_ASM_INST(0xcb060be0)  // neg	x0, x6, lsl #2
    KAI_ASM_INST(0xa90363f7)  // stp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xd37df0c5)  // lsl	x5, x6, #3
    KAI_ASM_INST(0xd37ef4e7)  // lsl	x7, x7, #2
    KAI_ASM_INST(0x2598e081)  // ptrue	p1.s, vl4
    KAI_ASM_INST(0xcb0600e2)  // sub	x2, x7, x6
    KAI_ASM_INST(0xa9046bf9)  // stp	x25, x26, [sp, #64]
    KAI_ASM_INST(0xd37ef4c8)  // lsl	x8, x6, #2
    KAI_ASM_INST(0x2518e120)  // ptrue	p0.b, vl16
    KAI_ASM_INST(0xd37ef442)  // lsl	x2, x2, #2
    KAI_ASM_INST(0xa9025bf5)  // stp	x21, x22, [sp, #32]
    KAI_ASM_INST(0x8b02020c)  // add	x12, x16, x2
    KAI_ASM_INST(0x8b000197)  // add	x23, x12, x0
    KAI_ASM_INST(0x8b060104)  // add	x4, x8, x6
    KAI_ASM_INST(0x8b0002fb)  // add	x27, x23, x0
    KAI_ASM_INST(0x8b0600ab)  // add	x11, x5, x6
    KAI_ASM_INST(0x8b00037a)  // add	x26, x27, x0
    KAI_ASM_INST(0xcb0600aa)  // sub	x10, x5, x6
    KAI_ASM_INST(0x8b000359)  // add	x25, x26, x0
    KAI_ASM_INST(0x910040af)  // add	x15, x5, #0x10
    KAI_ASM_INST(0x8b000336)  // add	x22, x25, x0
    KAI_ASM_INST(0x8b040a04)  // add	x4, x16, x4, lsl #2
    KAI_ASM_INST(0x8b0002d8)  // add	x24, x22, x0
    KAI_ASM_INST(0x91004108)  // add	x8, x8, #0x10
    KAI_ASM_INST(0x8b00031e)  // add	x30, x24, x0
    KAI_ASM_INST(0x8b0b0a0b)  // add	x11, x16, x11, lsl #2
    KAI_ASM_INST(0x8b0003c5)  // add	x5, x30, x0
    KAI_ASM_INST(0x8b0a0a0a)  // add	x10, x16, x10, lsl #2
    KAI_ASM_INST(0x8b0000b4)  // add	x20, x5, x0
    KAI_ASM_INST(0xaa1003f5)  // mov	x21, x16
    KAI_ASM_INST(0x8b000280)  // add	x0, x20, x0
    KAI_ASM_INST(0xa90b33e0)  // stp	x0, x12, [sp, #176]
    KAI_ASM_INST(0x91004080)  // add	x0, x4, #0x10
    KAI_ASM_INST(0xf90073e0)  // str	x0, [sp, #224]
    KAI_ASM_INST(0x8b080200)  // add	x0, x16, x8
    KAI_ASM_INST(0xf9006be0)  // str	x0, [sp, #208]
    KAI_ASM_INST(0x91004160)  // add	x0, x11, #0x10
    KAI_ASM_INST(0xf9007be0)  // str	x0, [sp, #240]
    KAI_ASM_INST(0x91004140)  // add	x0, x10, #0x10
    KAI_ASM_INST(0x910004d2)  // add	x18, x6, #0x1
    KAI_ASM_INST(0x8b061606)  // add	x6, x16, x6, lsl #5
    KAI_ASM_INST(0xf90077e0)  // str	x0, [sp, #232]
    KAI_ASM_INST(0x8b0f02a0)  // add	x0, x21, x15
    KAI_ASM_INST(0xb940fbec)  // ldr	w12, [sp, #248]
    KAI_ASM_INST(0x910040e7)  // add	x7, x7, #0x10
    KAI_ASM_INST(0xf90067e0)  // str	x0, [sp, #200]
    KAI_ASM_INST(0x910040c0)  // add	x0, x6, #0x10
    KAI_ASM_INST(0xa91117f1)  // stp	x17, x5, [sp, #272]
    KAI_ASM_INST(0x91004042)  // add	x2, x2, #0x10
    KAI_ASM_INST(0x8b070207)  // add	x7, x16, x7
    KAI_ASM_INST(0xf9007fe0)  // str	x0, [sp, #248]
    KAI_ASM_INST(0x8b1212a0)  // add	x0, x21, x18, lsl #4
    KAI_ASM_INST(0xf940a3e5)  // ldr	x5, [sp, #320]
    KAI_ASM_INST(0x8b020202)  // add	x2, x16, x2
    KAI_ASM_INST(0x1200018c)  // and	w12, w12, #0x1
    KAI_ASM_INST(0xaa1c03f0)  // mov	x16, x28
    KAI_ASM_INST(0x6d0627e8)  // stp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x91004028)  // add	x8, x1, #0x10
    KAI_ASM_INST(0x9100c1ab)  // add	x11, x13, #0x30
    KAI_ASM_INST(0x6d072fea)  // stp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x9100412a)  // add	x10, x9, #0x10
    KAI_ASM_INST(0x910041af)  // add	x15, x13, #0x10
    KAI_ASM_INST(0x6d0837ec)  // stp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x9100c026)  // add	x6, x1, #0x30
    KAI_ASM_INST(0xaa1103f2)  // mov	x18, x17
    KAI_ASM_INST(0x6d093fee)  // stp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xb900abec)  // str	w12, [sp, #168]
    KAI_ASM_INST(0x910081ac)  // add	x12, x13, #0x20
    KAI_ASM_INST(0xf90063e2)  // str	x2, [sp, #192]
    KAI_ASM_INST(0xf9006fe7)  // str	x7, [sp, #216]
    KAI_ASM_INST(0x91008027)  // add	x7, x1, #0x20
    KAI_ASM_INST(0xf90083e0)  // str	x0, [sp, #256]
    KAI_ASM_INST(0x910042a0)  // add	x0, x21, #0x10
    KAI_ASM_INST(0xf90087e0)  // str	x0, [sp, #264]
    KAI_ASM_INST(0xf900a7fc)  // str	x28, [sp, #328]
    KAI_ASM_INST(0xf9408ffc)  // ldr	x28, [sp, #280]
    KAI_ASM_INST(0xd2800100)  // mov	x0, #0x8
    KAI_ASM_INST(0xbd400660)  // ldr	s0, [x19, #4]
    KAI_ASM_INST(0xf100225f)  // cmp	x18, #0x8
    KAI_ASM_INST(0x4d40ca64)  // ld1r	{v4.4s}, [x19]
    KAI_ASM_INST(0x9a809251)  // csel	x17, x18, x0, ls
    KAI_ASM_INST(0xf940a7e0)  // ldr	x0, [sp, #328]
    KAI_ASM_INST(0x6e140404)  // mov	v4.s[2], v0.s[0]
    KAI_ASM_INST(0x6e1c0404)  // mov	v4.s[3], v0.s[0]
    KAI_ASM_INST(0xb40030c0)  // cbz	x0, 1930 <MlasGemmU8X8KernelUmmlaSveImpl+0x1930>
    KAI_ASM_INST(0x2d4088ab)  // ldp	s11, s2, [x5, #4]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4184a8)  // ldp	s8, s1, [x5, #12]
    KAI_ASM_INST(0x2d41a60a)  // ldp	s10, s9, [x16, #12]
    KAI_ASM_INST(0x2d40ca0c)  // ldp	s12, s18, [x16, #4]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x4d40c8a3)  // ld1r	{v3.4s}, [x5]
    KAI_ASM_INST(0x2d429607)  // ldp	s7, s5, [x16, #20]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd4014a6)  // ldr	s6, [x5, #20]
    KAI_ASM_INST(0x6e0c0563)  // mov	v3.s[1], v11.s[0]
    KAI_ASM_INST(0x4d40ca15)  // ld1r	{v21.4s}, [x16]
    KAI_ASM_INST(0x4e040529)  // dup	v9.4s, v9.s[0]
    KAI_ASM_INST(0xbd401e00)  // ldr	s0, [x16, #28]
    KAI_ASM_INST(0x6e0c0502)  // mov	v2.s[1], v8.s[0]
    KAI_ASM_INST(0x4e040652)  // dup	v18.4s, v18.s[0]
    KAI_ASM_INST(0x6e0c04c1)  // mov	v1.s[1], v6.s[0]
    KAI_ASM_INST(0x4e0404a5)  // dup	v5.4s, v5.s[0]
    KAI_ASM_INST(0x6e0c04e9)  // mov	v9.s[1], v7.s[0]
    KAI_ASM_INST(0x6e1c0563)  // mov	v3.s[3], v11.s[0]
    KAI_ASM_INST(0x6e0c0595)  // mov	v21.s[1], v12.s[0]
    KAI_ASM_INST(0x6e1c0502)  // mov	v2.s[3], v8.s[0]
    KAI_ASM_INST(0x6e0c0552)  // mov	v18.s[1], v10.s[0]
    KAI_ASM_INST(0x6e1c04c1)  // mov	v1.s[3], v6.s[0]
    KAI_ASM_INST(0x6e0c0405)  // mov	v5.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c04e9)  // mov	v9.s[3], v7.s[0]
    KAI_ASM_INST(0x6e1c0595)  // mov	v21.s[3], v12.s[0]
    KAI_ASM_INST(0x4ea31c67)  // mov	v7.16b, v3.16b
    KAI_ASM_INST(0x4ea21c46)  // mov	v6.16b, v2.16b
    KAI_ASM_INST(0x6e1c0552)  // mov	v18.s[3], v10.s[0]
    KAI_ASM_INST(0x6e1c0405)  // mov	v5.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea11c20)  // mov	v0.16b, v1.16b
    KAI_ASM_INST(0x4ea496a7)  // mla	v7.4s, v21.4s, v4.4s
    KAI_ASM_INST(0x4ea49646)  // mla	v6.4s, v18.4s, v4.4s
    KAI_ASM_INST(0x4ea49520)  // mla	v0.4s, v9.4s, v4.4s
    KAI_ASM_INST(0x4ea59c84)  // mul	v4.4s, v4.4s, v5.4s
    KAI_ASM_INST(0x3d804fe7)  // str	q7, [sp, #304]
    KAI_ASM_INST(0xa540a41b)  // ld1w	{z27.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4328a0)  // ldp	s0, s10, [x5, #24]
    KAI_ASM_INST(0x3cc08266)  // ldur	q6, [x19, #8]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x4ea31c71)  // mov	v17.16b, v3.16b
    KAI_ASM_INST(0x4e8638c8)  // zip1	v8.4s, v6.4s, v6.4s
    KAI_ASM_INST(0x4ea21c4d)  // mov	v13.16b, v2.16b
    KAI_ASM_INST(0x6e0c0540)  // mov	v0.s[1], v10.s[0]
    KAI_ASM_INST(0x4ea11c2c)  // mov	v12.16b, v1.16b
    KAI_ASM_INST(0x4eb59511)  // mla	v17.4s, v8.4s, v21.4s
    KAI_ASM_INST(0x4eb2950d)  // mla	v13.4s, v8.4s, v18.4s
    KAI_ASM_INST(0x6e1c0540)  // mov	v0.s[3], v10.s[0]
    KAI_ASM_INST(0x4ea9950c)  // mla	v12.4s, v8.4s, v9.4s
    KAI_ASM_INST(0x3cc18267)  // ldur	q7, [x19, #24]
    KAI_ASM_INST(0x4ea08484)  // add	v4.4s, v4.4s, v0.4s
    KAI_ASM_INST(0x4ea01c0a)  // mov	v10.16b, v0.16b
    KAI_ASM_INST(0x4e8678c6)  // zip2	v6.4s, v6.4s, v6.4s
    KAI_ASM_INST(0x4ea31c70)  // mov	v16.16b, v3.16b
    KAI_ASM_INST(0x4ea21c4f)  // mov	v15.16b, v2.16b
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41a)  // ld1w	{z26.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea894aa)  // mla	v10.4s, v5.4s, v8.4s
    KAI_ASM_INST(0x3d804ff1)  // str	q17, [sp, #304]
    KAI_ASM_INST(0xa540a411)  // ld1w	{z17.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea11c2e)  // mov	v14.16b, v1.16b
    KAI_ASM_INST(0x3d804fed)  // str	q13, [sp, #304]
    KAI_ASM_INST(0xa540a408)  // ld1w	{z8.s}, p1/z, [x0]
    KAI_ASM_INST(0x4e8738eb)  // zip1	v11.4s, v7.4s, v7.4s
    KAI_ASM_INST(0x3d804fec)  // str	q12, [sp, #304]
    KAI_ASM_INST(0xa540a419)  // ld1w	{z25.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea01c0c)  // mov	v12.16b, v0.16b
    KAI_ASM_INST(0x3d804fea)  // str	q10, [sp, #304]
    KAI_ASM_INST(0xa540a40d)  // ld1w	{z13.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb594d0)  // mla	v16.4s, v6.4s, v21.4s
    KAI_ASM_INST(0x4eb294cf)  // mla	v15.4s, v6.4s, v18.4s
    KAI_ASM_INST(0x4ea994ce)  // mla	v14.4s, v6.4s, v9.4s
    KAI_ASM_INST(0x4ea594cc)  // mla	v12.4s, v6.4s, v5.4s
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x3d804ff0)  // str	q16, [sp, #304]
    KAI_ASM_INST(0xa540a410)  // ld1w	{z16.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea21c44)  // mov	v4.16b, v2.16b
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a41f)  // ld1w	{z31.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb59566)  // mla	v6.4s, v11.4s, v21.4s
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a418)  // ld1w	{z24.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb29564)  // mla	v4.4s, v11.4s, v18.4s
    KAI_ASM_INST(0x3d804fec)  // str	q12, [sp, #304]
    KAI_ASM_INST(0xa540a40c)  // ld1w	{z12.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea11c2f)  // mov	v15.16b, v1.16b
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0x4ea01c0e)  // mov	v14.16b, v0.16b
    KAI_ASM_INST(0x4e8778ea)  // zip2	v10.4s, v7.4s, v7.4s
    KAI_ASM_INST(0xa540a407)  // ld1w	{z7.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea31c66)  // mov	v6.16b, v3.16b
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41e)  // ld1w	{z30.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea21c44)  // mov	v4.16b, v2.16b
    KAI_ASM_INST(0x4ea9956f)  // mla	v15.4s, v11.4s, v9.4s
    KAI_ASM_INST(0x4ea5956e)  // mla	v14.4s, v11.4s, v5.4s
    KAI_ASM_INST(0x4eb59546)  // mla	v6.4s, v10.4s, v21.4s
    KAI_ASM_INST(0x4eb29544)  // mla	v4.4s, v10.4s, v18.4s
    KAI_ASM_INST(0x4ea11c36)  // mov	v22.16b, v1.16b
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a417)  // ld1w	{z23.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea01c0f)  // mov	v15.16b, v0.16b
    KAI_ASM_INST(0x3d804fee)  // str	q14, [sp, #304]
    KAI_ASM_INST(0xa540a40b)  // ld1w	{z11.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea99556)  // mla	v22.4s, v10.4s, v9.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a406)  // ld1w	{z6.s}, p1/z, [x0]
    KAI_ASM_INST(0x4ea5954f)  // mla	v15.4s, v10.4s, v5.4s
    KAI_ASM_INST(0x3d804fe4)  // str	q4, [sp, #304]
    KAI_ASM_INST(0xa540a41d)  // ld1w	{z29.s}, p1/z, [x0]
    KAI_ASM_INST(0x2d453a64)  // ldp	s4, s14, [x19, #40]
    KAI_ASM_INST(0x3d804ff6)  // str	q22, [sp, #304]
    KAI_ASM_INST(0xa540a416)  // ld1w	{z22.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fef)  // str	q15, [sp, #304]
    KAI_ASM_INST(0xa540a40a)  // ld1w	{z10.s}, p1/z, [x0]
    KAI_ASM_INST(0x4e040484)  // dup	v4.4s, v4.s[0]
    KAI_ASM_INST(0x6e1405c4)  // mov	v4.s[2], v14.s[0]
    KAI_ASM_INST(0x6e1c05c4)  // mov	v4.s[3], v14.s[0]
    KAI_ASM_INST(0x4eb59483)  // mla	v3.4s, v4.4s, v21.4s
    KAI_ASM_INST(0x4eb29482)  // mla	v2.4s, v4.4s, v18.4s
    KAI_ASM_INST(0x4ea99481)  // mla	v1.4s, v4.4s, v9.4s
    KAI_ASM_INST(0x4ea59480)  // mla	v0.4s, v4.4s, v5.4s
    KAI_ASM_INST(0x3d804fe3)  // str	q3, [sp, #304]
    KAI_ASM_INST(0xa540a412)  // ld1w	{z18.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe2)  // str	q2, [sp, #304]
    KAI_ASM_INST(0xa540a41c)  // ld1w	{z28.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe1)  // str	q1, [sp, #304]
    KAI_ASM_INST(0xa540a415)  // ld1w	{z21.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a409)  // ld1w	{z9.s}, p1/z, [x0]
    KAI_ASM_INST(0xb4000583)  // cbz	x3, 15ec <MlasGemmU8X8KernelUmmlaSveImpl+0x15ec>
    KAI_ASM_INST(0xd2800002)  // mov	x2, #0x0
    KAI_ASM_INST(0xd2800000)  // mov	x0, #0x0
    KAI_ASM_INST(0xd2800004)  // mov	x4, #0x0
    KAI_ASM_INST(0xd503201f)  // nop
    KAI_ASM_INST(0xa40041a4)  // ld1b	{z4.b}, p0/z, [x13, x0]
    KAI_ASM_INST(0xa4004023)  // ld1b	{z3.b}, p0/z, [x1, x0]
    KAI_ASM_INST(0xa4004102)  // ld1b	{z2.b}, p0/z, [x8, x0]
    KAI_ASM_INST(0xa40040e1)  // ld1b	{z1.b}, p0/z, [x7, x0]
    KAI_ASM_INST(0xa40040c0)  // ld1b	{z0.b}, p0/z, [x6, x0]
    KAI_ASM_INST(0x45c3989b)  // ummla	z27.s, z4.b, z3.b
    KAI_ASM_INST(0x45c29893)  // ummla	z19.s, z4.b, z2.b
    KAI_ASM_INST(0x45c19894)  // ummla	z20.s, z4.b, z1.b
    KAI_ASM_INST(0x45c0989a)  // ummla	z26.s, z4.b, z0.b
    KAI_ASM_INST(0xa4004184)  // ld1b	{z4.b}, p0/z, [x12, x0]
    KAI_ASM_INST(0xa40041e5)  // ld1b	{z5.b}, p0/z, [x15, x0]
    KAI_ASM_INST(0x45c39890)  // ummla	z16.s, z4.b, z3.b
    KAI_ASM_INST(0x45c2989f)  // ummla	z31.s, z4.b, z2.b
    KAI_ASM_INST(0x45c19898)  // ummla	z24.s, z4.b, z1.b
    KAI_ASM_INST(0x45c0988c)  // ummla	z12.s, z4.b, z0.b
    KAI_ASM_INST(0x91000484)  // add	x4, x4, #0x1
    KAI_ASM_INST(0xa4024124)  // ld1b	{z4.b}, p0/z, [x9, x2]
    KAI_ASM_INST(0x45c398b1)  // ummla	z17.s, z5.b, z3.b
    KAI_ASM_INST(0x45c298a8)  // ummla	z8.s, z5.b, z2.b
    KAI_ASM_INST(0x45c198b9)  // ummla	z25.s, z5.b, z1.b
    KAI_ASM_INST(0x45c098ad)  // ummla	z13.s, z5.b, z0.b
    KAI_ASM_INST(0x45c39886)  // ummla	z6.s, z4.b, z3.b
    KAI_ASM_INST(0xa4004165)  // ld1b	{z5.b}, p0/z, [x11, x0]
    KAI_ASM_INST(0x45c2989d)  // ummla	z29.s, z4.b, z2.b
    KAI_ASM_INST(0x45c19896)  // ummla	z22.s, z4.b, z1.b
    KAI_ASM_INST(0x45c0988a)  // ummla	z10.s, z4.b, z0.b
    KAI_ASM_INST(0x91010000)  // add	x0, x0, #0x40
    KAI_ASM_INST(0xa4024144)  // ld1b	{z4.b}, p0/z, [x10, x2]
    KAI_ASM_INST(0x45c398a7)  // ummla	z7.s, z5.b, z3.b
    KAI_ASM_INST(0x91008042)  // add	x2, x2, #0x20
    KAI_ASM_INST(0x45c298be)  // ummla	z30.s, z5.b, z2.b
    KAI_ASM_INST(0x45c198b7)  // ummla	z23.s, z5.b, z1.b
    KAI_ASM_INST(0x45c098ab)  // ummla	z11.s, z5.b, z0.b
    KAI_ASM_INST(0x45c39892)  // ummla	z18.s, z4.b, z3.b
    KAI_ASM_INST(0x45c2989c)  // ummla	z28.s, z4.b, z2.b
    KAI_ASM_INST(0x45c19895)  // ummla	z21.s, z4.b, z1.b
    KAI_ASM_INST(0x45c09889)  // ummla	z9.s, z4.b, z0.b
    KAI_ASM_INST(0xeb04007f)  // cmp	x3, x4
    KAI_ASM_INST(0x54fffb41)  // b.ne	1550 <MlasGemmU8X8KernelUmmlaSveImpl+0x1550>
    KAI_ASM_INST(0xf100123f)  // cmp	x17, #0x4
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x9a809220)  // csel	x0, x17, x0, ls
    KAI_ASM_INST(0x05f36b60)  // uzp1	z0.d, z27.d, z19.d
    KAI_ASM_INST(0x05fa6a82)  // uzp1	z2.d, z20.d, z26.d
    KAI_ASM_INST(0x05f36f73)  // uzp2	z19.d, z27.d, z19.d
    KAI_ASM_INST(0x25a01fe2)  // whilelo	p2.s, xzr, x0
    KAI_ASM_INST(0xb940abe0)  // ldr	w0, [sp, #168]
    KAI_ASM_INST(0x05fa6e94)  // uzp2	z20.d, z20.d, z26.d
    KAI_ASM_INST(0x350015a0)  // cbnz	w0, 18c4 <MlasGemmU8X8KernelUmmlaSveImpl+0x18c4>
    KAI_ASM_INST(0xa540aaa1)  // ld1w	{z1.s}, p2/z, [x21]
    KAI_ASM_INST(0x04a00020)  // add	z0.s, z1.s, z0.s
    KAI_ASM_INST(0xe540eaa0)  // st1w	{z0.s}, p2, [x21]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54002529)  // b.ls	1ac8 <MlasGemmU8X8KernelUmmlaSveImpl+0x1ac8>
    KAI_ASM_INST(0xf94087e2)  // ldr	x2, [sp, #264]
    KAI_ASM_INST(0x05fd68c3)  // uzp1	z3.d, z6.d, z29.d
    KAI_ASM_INST(0xd1001220)  // sub	x0, x17, #0x4
    KAI_ASM_INST(0x05fd6cc1)  // uzp2	z1.d, z6.d, z29.d
    KAI_ASM_INST(0x25a01fe3)  // whilelo	p3.s, xzr, x0
    KAI_ASM_INST(0xa540ac4f)  // ld1w	{z15.s}, p3/z, [x2]
    KAI_ASM_INST(0x04a201ef)  // add	z15.s, z15.s, z2.s
    KAI_ASM_INST(0xe540ec4f)  // st1w	{z15.s}, p3, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xa540a846)  // ld1w	{z6.s}, p2/z, [x2]
    KAI_ASM_INST(0x04b300c6)  // add	z6.s, z6.s, z19.s
    KAI_ASM_INST(0xe540e846)  // st1w	{z6.s}, p2, [x2]
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x04b400c6)  // add	z6.s, z6.s, z20.s
    KAI_ASM_INST(0xe540ec46)  // st1w	{z6.s}, p3, [x2]
    KAI_ASM_INST(0xa540aa86)  // ld1w	{z6.s}, p2/z, [x20]
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0x05e86a2e)  // uzp1	z14.d, z17.d, z8.d
    KAI_ASM_INST(0x04a601ce)  // add	z14.s, z14.s, z6.s
    KAI_ASM_INST(0xe540ea8e)  // st1w	{z14.s}, p2, [x20]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ed6b28)  // uzp1	z8.d, z25.d, z13.d
    KAI_ASM_INST(0x04a60108)  // add	z8.s, z8.s, z6.s
    KAI_ASM_INST(0xe540ec48)  // st1w	{z8.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab86)  // ld1w	{z6.s}, p2/z, [x28]
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0x04a60231)  // add	z17.s, z17.s, z6.s
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ed6f39)  // uzp2	z25.d, z25.d, z13.d
    KAI_ASM_INST(0x04a60339)  // add	z25.s, z25.s, z6.s
    KAI_ASM_INST(0xe540ec59)  // st1w	{z25.s}, p3, [x2]
    KAI_ASM_INST(0xa540abc6)  // ld1w	{z6.s}, p2/z, [x30]
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0x05ff6a0d)  // uzp1	z13.d, z16.d, z31.d
    KAI_ASM_INST(0x04a601ad)  // add	z13.s, z13.s, z6.s
    KAI_ASM_INST(0xe540ebcd)  // st1w	{z13.s}, p2, [x30]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ec6b1a)  // uzp1	z26.d, z24.d, z12.d
    KAI_ASM_INST(0x04a6035a)  // add	z26.s, z26.s, z6.s
    KAI_ASM_INST(0xe540ec5a)  // st1w	{z26.s}, p3, [x2]
    KAI_ASM_INST(0xa540ab06)  // ld1w	{z6.s}, p2/z, [x24]
    KAI_ASM_INST(0xf94073e2)  // ldr	x2, [sp, #224]
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x04a60210)  // add	z16.s, z16.s, z6.s
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xa540ac46)  // ld1w	{z6.s}, p3/z, [x2]
    KAI_ASM_INST(0x05ec6f18)  // uzp2	z24.d, z24.d, z12.d
    KAI_ASM_INST(0x04a60318)  // add	z24.s, z24.s, z6.s
    KAI_ASM_INST(0xe540ec58)  // st1w	{z24.s}, p3, [x2]
    KAI_ASM_INST(0xa540aac6)  // ld1w	{z6.s}, p2/z, [x22]
    KAI_ASM_INST(0x910042c0)  // add	x0, x22, #0x10
    KAI_ASM_INST(0x05fe68ec)  // uzp1	z12.d, z7.d, z30.d
    KAI_ASM_INST(0x04a6018c)  // add	z12.s, z12.s, z6.s
    KAI_ASM_INST(0xe540eacc)  // st1w	{z12.s}, p2, [x22]
    KAI_ASM_INST(0xa540ac06)  // ld1w	{z6.s}, p3/z, [x0]
    KAI_ASM_INST(0x05eb6afb)  // uzp1	z27.d, z23.d, z11.d
    KAI_ASM_INST(0x04a6037b)  // add	z27.s, z27.s, z6.s
    KAI_ASM_INST(0xe540ec1b)  // st1w	{z27.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab26)  // ld1w	{z6.s}, p2/z, [x25]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0x05eb6ef7)  // uzp2	z23.d, z23.d, z11.d
    KAI_ASM_INST(0x04a600e7)  // add	z7.s, z7.s, z6.s
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xa540ac06)  // ld1w	{z6.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a602e6)  // add	z6.s, z23.s, z6.s
    KAI_ASM_INST(0xe540ec06)  // st1w	{z6.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab46)  // ld1w	{z6.s}, p2/z, [x26]
    KAI_ASM_INST(0x04a60063)  // add	z3.s, z3.s, z6.s
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05ea6ac2)  // uzp1	z2.d, z22.d, z10.d
    KAI_ASM_INST(0xe540eb43)  // st1w	{z3.s}, p2, [x26]
    KAI_ASM_INST(0xa540ac03)  // ld1w	{z3.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xe540ec02)  // st1w	{z2.s}, p3, [x0]
    KAI_ASM_INST(0xa540ab62)  // ld1w	{z2.s}, p2/z, [x27]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05ea6ec0)  // uzp2	z0.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6a44)  // uzp1	z4.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96aa5)  // uzp1	z5.d, z21.d, z9.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540eb61)  // st1w	{z1.s}, p2, [x27]
    KAI_ASM_INST(0xa540ac01)  // ld1w	{z1.s}, p3/z, [x0]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xa540aae0)  // ld1w	{z0.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a00080)  // add	z0.s, z4.s, z0.s
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54001788)  // b.hi	1a90 <MlasGemmU8X8KernelUmmlaSveImpl+0x1a90>
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xa540a800)  // ld1w	{z0.s}, p2/z, [x0]
    KAI_ASM_INST(0x540031c0)  // b.eq	1de4 <MlasGemmU8X8KernelUmmlaSveImpl+0x1de4>
    KAI_ASM_INST(0x04a00252)  // add	z18.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0xa94b93e0)  // ldp	x0, x4, [sp, #184]
    KAI_ASM_INST(0xd1002252)  // sub	x18, x18, #0x8
    KAI_ASM_INST(0xf9408be2)  // ldr	x2, [sp, #272]
    KAI_ASM_INST(0x910082f7)  // add	x23, x23, #0x20
    KAI_ASM_INST(0x9100837b)  // add	x27, x27, #0x20
    KAI_ASM_INST(0x9100835a)  // add	x26, x26, #0x20
    KAI_ASM_INST(0x91008339)  // add	x25, x25, #0x20
    KAI_ASM_INST(0x910082d6)  // add	x22, x22, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0x91008000)  // add	x0, x0, #0x20
    KAI_ASM_INST(0xa90b93e0)  // stp	x0, x4, [sp, #184]
    KAI_ASM_INST(0xcb120040)  // sub	x0, x2, x18
    KAI_ASM_INST(0x91008318)  // add	x24, x24, #0x20
    KAI_ASM_INST(0xf9407be4)  // ldr	x4, [sp, #240]
    KAI_ASM_INST(0x910083de)  // add	x30, x30, #0x20
    KAI_ASM_INST(0x9100839c)  // add	x28, x28, #0x20
    KAI_ASM_INST(0x91008294)  // add	x20, x20, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9007be4)  // str	x4, [sp, #240]
    KAI_ASM_INST(0xf94077e4)  // ldr	x4, [sp, #232]
    KAI_ASM_INST(0x910082b5)  // add	x21, x21, #0x20
    KAI_ASM_INST(0x910080a5)  // add	x5, x5, #0x20
    KAI_ASM_INST(0x91008210)  // add	x16, x16, #0x20
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90077e4)  // str	x4, [sp, #232]
    KAI_ASM_INST(0xf94073e4)  // ldr	x4, [sp, #224]
    KAI_ASM_INST(0x8b0e0021)  // add	x1, x1, x14
    KAI_ASM_INST(0x8b0e0108)  // add	x8, x8, x14
    KAI_ASM_INST(0x8b0e00e7)  // add	x7, x7, x14
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90073e4)  // str	x4, [sp, #224]
    KAI_ASM_INST(0xf9406fe4)  // ldr	x4, [sp, #216]
    KAI_ASM_INST(0x8b0e00c6)  // add	x6, x6, x14
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9006fe4)  // str	x4, [sp, #216]
    KAI_ASM_INST(0xf9405be4)  // ldr	x4, [sp, #176]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9005be4)  // str	x4, [sp, #176]
    KAI_ASM_INST(0xf9406be4)  // ldr	x4, [sp, #208]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9006be4)  // str	x4, [sp, #208]
    KAI_ASM_INST(0xf9407fe4)  // ldr	x4, [sp, #248]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf9007fe4)  // str	x4, [sp, #248]
    KAI_ASM_INST(0xf94083e4)  // ldr	x4, [sp, #256]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90083e4)  // str	x4, [sp, #256]
    KAI_ASM_INST(0xf94067e4)  // ldr	x4, [sp, #200]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90067e4)  // str	x4, [sp, #200]
    KAI_ASM_INST(0xf94087e4)  // ldr	x4, [sp, #264]
    KAI_ASM_INST(0x91008084)  // add	x4, x4, #0x20
    KAI_ASM_INST(0xf90087e4)  // str	x4, [sp, #264]
    KAI_ASM_INST(0xeb00005f)  // cmp	x2, x0
    KAI_ASM_INST(0x54ffd348)  // b.hi	12f8 <MlasGemmU8X8KernelUmmlaSveImpl+0x12f8>
    KAI_ASM_INST(0xa9425bf5)  // ldp	x21, x22, [sp, #32]
    KAI_ASM_INST(0xa94363f7)  // ldp	x23, x24, [sp, #48]
    KAI_ASM_INST(0xa9446bf9)  // ldp	x25, x26, [sp, #64]
    KAI_ASM_INST(0x6d4627e8)  // ldp	d8, d9, [sp, #96]
    KAI_ASM_INST(0x6d472fea)  // ldp	d10, d11, [sp, #112]
    KAI_ASM_INST(0x6d4837ec)  // ldp	d12, d13, [sp, #128]
    KAI_ASM_INST(0x6d493fee)  // ldp	d14, d15, [sp, #144]
    KAI_ASM_INST(0xd2800180)  // mov	x0, #0xc
    KAI_ASM_INST(0xa94153f3)  // ldp	x19, x20, [sp, #16]
    KAI_ASM_INST(0xa94573fb)  // ldp	x27, x28, [sp, #80]
    KAI_ASM_INST(0xa8d47bfd)  // ldp	x29, x30, [sp], #320
    KAI_ASM_INST(0xd65f03c0)  // ret
    KAI_ASM_INST(0xe540eaa0)  // st1w	{z0.s}, p2, [x21]
    KAI_ASM_INST(0xf100125f)  // cmp	x18, #0x4
    KAI_ASM_INST(0x54000688)  // b.hi	199c <MlasGemmU8X8KernelUmmlaSveImpl+0x199c>
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86a24)  // uzp1	z4.d, z17.d, z8.d
    KAI_ASM_INST(0x05ff6a03)  // uzp1	z3.d, z16.d, z31.d
    KAI_ASM_INST(0x05fe68e2)  // uzp1	z2.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd68c1)  // uzp1	z1.d, z6.d, z29.d
    KAI_ASM_INST(0x05fc6a40)  // uzp1	z0.d, z18.d, z28.d
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd6cc6)  // uzp2	z6.d, z6.d, z29.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0xe540e813)  // st1w	{z19.s}, p2, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540ea84)  // st1w	{z4.s}, p2, [x20]
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xe540ebc3)  // st1w	{z3.s}, p2, [x30]
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xe540eac2)  // st1w	{z2.s}, p2, [x22]
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xe540eb41)  // st1w	{z1.s}, p2, [x26]
    KAI_ASM_INST(0xe540eb66)  // st1w	{z6.s}, p2, [x27]
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0x17ffffa3)  // b	17b8 <MlasGemmU8X8KernelUmmlaSveImpl+0x17b8>
    KAI_ASM_INST(0x2d4088a6)  // ldp	s6, s2, [x5, #4]
    KAI_ASM_INST(0x9104c3e0)  // add	x0, sp, #0x130
    KAI_ASM_INST(0x2d4184a5)  // ldp	s5, s1, [x5, #12]
    KAI_ASM_INST(0x4e040442)  // dup	v2.4s, v2.s[0]
    KAI_ASM_INST(0x4d40c8a3)  // ld1r	{v3.4s}, [x5]
    KAI_ASM_INST(0x4f000435)  // movi	v21.4s, #0x1
    KAI_ASM_INST(0xbd4014a0)  // ldr	s0, [x5, #20]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0x6e0c04c3)  // mov	v3.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c04a2)  // mov	v2.s[1], v5.s[0]
    KAI_ASM_INST(0x4eb51eb2)  // mov	v18.16b, v21.16b
    KAI_ASM_INST(0x6e0c0401)  // mov	v1.s[1], v0.s[0]
    KAI_ASM_INST(0x6e1c04c3)  // mov	v3.s[3], v6.s[0]
    KAI_ASM_INST(0x6e1c04a2)  // mov	v2.s[3], v5.s[0]
    KAI_ASM_INST(0x4eb51ea9)  // mov	v9.16b, v21.16b
    KAI_ASM_INST(0x6e1c0401)  // mov	v1.s[3], v0.s[0]
    KAI_ASM_INST(0x4ea38486)  // add	v6.4s, v4.4s, v3.4s
    KAI_ASM_INST(0x4ea48445)  // add	v5.4s, v2.4s, v4.4s
    KAI_ASM_INST(0x4ea48420)  // add	v0.4s, v1.4s, v4.4s
    KAI_ASM_INST(0x3d804fe6)  // str	q6, [sp, #304]
    KAI_ASM_INST(0xa540a41b)  // ld1w	{z27.s}, p1/z, [x0]
    KAI_ASM_INST(0x3d804fe5)  // str	q5, [sp, #304]
    KAI_ASM_INST(0xa540a413)  // ld1w	{z19.s}, p1/z, [x0]
    KAI_ASM_INST(0x4eb51ea5)  // mov	v5.16b, v21.16b
    KAI_ASM_INST(0x3d804fe0)  // str	q0, [sp, #304]
    KAI_ASM_INST(0xa540a414)  // ld1w	{z20.s}, p1/z, [x0]
    KAI_ASM_INST(0x17fffe8b)  // b	13c4 <MlasGemmU8X8KernelUmmlaSveImpl+0x13c4>
    KAI_ASM_INST(0xf94087e2)  // ldr	x2, [sp, #264]
    KAI_ASM_INST(0x05e86a24)  // uzp1	z4.d, z17.d, z8.d
    KAI_ASM_INST(0xd1001231)  // sub	x17, x17, #0x4
    KAI_ASM_INST(0x25b11fe3)  // whilelo	p3.s, xzr, x17
    KAI_ASM_INST(0xe540ec42)  // st1w	{z2.s}, p3, [x2]
    KAI_ASM_INST(0xf9405be2)  // ldr	x2, [sp, #176]
    KAI_ASM_INST(0xe540e853)  // st1w	{z19.s}, p2, [x2]
    KAI_ASM_INST(0x05ed6b25)  // uzp1	z5.d, z25.d, z13.d
    KAI_ASM_INST(0xf9406be2)  // ldr	x2, [sp, #208]
    KAI_ASM_INST(0xe540ec54)  // st1w	{z20.s}, p3, [x2]
    KAI_ASM_INST(0xe540ea84)  // st1w	{z4.s}, p2, [x20]
    KAI_ASM_INST(0x910042c0)  // add	x0, x22, #0x10
    KAI_ASM_INST(0xf94067e2)  // ldr	x2, [sp, #200]
    KAI_ASM_INST(0xe540ec45)  // st1w	{z5.s}, p3, [x2]
    KAI_ASM_INST(0x05e86e31)  // uzp2	z17.d, z17.d, z8.d
    KAI_ASM_INST(0x05ed6f39)  // uzp2	z25.d, z25.d, z13.d
    KAI_ASM_INST(0xf9406fe2)  // ldr	x2, [sp, #216]
    KAI_ASM_INST(0xe540eb91)  // st1w	{z17.s}, p2, [x28]
    KAI_ASM_INST(0xe540ec59)  // st1w	{z25.s}, p3, [x2]
    KAI_ASM_INST(0x05eb6ae1)  // uzp1	z1.d, z23.d, z11.d
    KAI_ASM_INST(0xf94083e2)  // ldr	x2, [sp, #256]
    KAI_ASM_INST(0x05ff6a0d)  // uzp1	z13.d, z16.d, z31.d
    KAI_ASM_INST(0x05ec6b08)  // uzp1	z8.d, z24.d, z12.d
    KAI_ASM_INST(0x05fe68e3)  // uzp1	z3.d, z7.d, z30.d
    KAI_ASM_INST(0x05ff6e10)  // uzp2	z16.d, z16.d, z31.d
    KAI_ASM_INST(0x05ec6f18)  // uzp2	z24.d, z24.d, z12.d
    KAI_ASM_INST(0xe540ebcd)  // st1w	{z13.s}, p2, [x30]
    KAI_ASM_INST(0xe540ec48)  // st1w	{z8.s}, p3, [x2]
    KAI_ASM_INST(0xf94073e2)  // ldr	x2, [sp, #224]
    KAI_ASM_INST(0xe540eb10)  // st1w	{z16.s}, p2, [x24]
    KAI_ASM_INST(0xe540ec58)  // st1w	{z24.s}, p3, [x2]
    KAI_ASM_INST(0xe540eac3)  // st1w	{z3.s}, p2, [x22]
    KAI_ASM_INST(0xe540ec01)  // st1w	{z1.s}, p3, [x0]
    KAI_ASM_INST(0xf94077e0)  // ldr	x0, [sp, #232]
    KAI_ASM_INST(0x05fd68c0)  // uzp1	z0.d, z6.d, z29.d
    KAI_ASM_INST(0x05fe6ce7)  // uzp2	z7.d, z7.d, z30.d
    KAI_ASM_INST(0xf9407fe2)  // ldr	x2, [sp, #248]
    KAI_ASM_INST(0x05eb6ef7)  // uzp2	z23.d, z23.d, z11.d
    KAI_ASM_INST(0xe540eb27)  // st1w	{z7.s}, p2, [x25]
    KAI_ASM_INST(0xe540ec17)  // st1w	{z23.s}, p3, [x0]
    KAI_ASM_INST(0x910042e0)  // add	x0, x23, #0x10
    KAI_ASM_INST(0x05ea6ac2)  // uzp1	z2.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6a41)  // uzp1	z1.d, z18.d, z28.d
    KAI_ASM_INST(0x05fd6cc6)  // uzp2	z6.d, z6.d, z29.d
    KAI_ASM_INST(0x05ea6ed6)  // uzp2	z22.d, z22.d, z10.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0xe540eb40)  // st1w	{z0.s}, p2, [x26]
    KAI_ASM_INST(0x05e96aa0)  // uzp1	z0.d, z21.d, z9.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540ec42)  // st1w	{z2.s}, p3, [x2]
    KAI_ASM_INST(0xf9407be2)  // ldr	x2, [sp, #240]
    KAI_ASM_INST(0xe540eb66)  // st1w	{z6.s}, p2, [x27]
    KAI_ASM_INST(0xe540ec56)  // st1w	{z22.s}, p3, [x2]
    KAI_ASM_INST(0xe540eae1)  // st1w	{z1.s}, p2, [x23]
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xe540e812)  // st1w	{z18.s}, p2, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xe540f015)  // st1w	{z21.s}, p4, [x0]
    KAI_ASM_INST(0x17ffff4b)  // b	17b8 <MlasGemmU8X8KernelUmmlaSveImpl+0x17b8>
    KAI_ASM_INST(0x910042e0)  // add	x0, x23, #0x10
    KAI_ASM_INST(0xd1001231)  // sub	x17, x17, #0x4
    KAI_ASM_INST(0x25b11fe4)  // whilelo	p4.s, xzr, x17
    KAI_ASM_INST(0xa540b000)  // ld1w	{z0.s}, p4/z, [x0]
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0xe540f005)  // st1w	{z5.s}, p4, [x0]
    KAI_ASM_INST(0xf9405fe0)  // ldr	x0, [sp, #184]
    KAI_ASM_INST(0xa540a800)  // ld1w	{z0.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a00240)  // add	z0.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e800)  // st1w	{z0.s}, p2, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xa540b000)  // ld1w	{z0.s}, p4/z, [x0]
    KAI_ASM_INST(0x04a002b5)  // add	z21.s, z21.s, z0.s
    KAI_ASM_INST(0x17fffff0)  // b	1a84 <MlasGemmU8X8KernelUmmlaSveImpl+0x1a84>
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86a2b)  // uzp1	z11.d, z17.d, z8.d
    KAI_ASM_INST(0xa540a80c)  // ld1w	{z12.s}, p2/z, [x0]
    KAI_ASM_INST(0x05e86e23)  // uzp2	z3.d, z17.d, z8.d
    KAI_ASM_INST(0x05fe6ce1)  // uzp2	z1.d, z7.d, z30.d
    KAI_ASM_INST(0x05fe68e8)  // uzp1	z8.d, z7.d, z30.d
    KAI_ASM_INST(0x05fd6cc0)  // uzp2	z0.d, z6.d, z29.d
    KAI_ASM_INST(0x05fd68c7)  // uzp1	z7.d, z6.d, z29.d
    KAI_ASM_INST(0x04b3018c)  // add	z12.s, z12.s, z19.s
    KAI_ASM_INST(0xe540e80c)  // st1w	{z12.s}, p2, [x0]
    KAI_ASM_INST(0xa540aa86)  // ld1w	{z6.s}, p2/z, [x20]
    KAI_ASM_INST(0x04a60166)  // add	z6.s, z11.s, z6.s
    KAI_ASM_INST(0xe540ea86)  // st1w	{z6.s}, p2, [x20]
    KAI_ASM_INST(0xa540ab86)  // ld1w	{z6.s}, p2/z, [x28]
    KAI_ASM_INST(0x04a60063)  // add	z3.s, z3.s, z6.s
    KAI_ASM_INST(0x05ff6a0a)  // uzp1	z10.d, z16.d, z31.d
    KAI_ASM_INST(0xe540eb83)  // st1w	{z3.s}, p2, [x28]
    KAI_ASM_INST(0xa540abc3)  // ld1w	{z3.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a30143)  // add	z3.s, z10.s, z3.s
    KAI_ASM_INST(0x05ff6e02)  // uzp2	z2.d, z16.d, z31.d
    KAI_ASM_INST(0xe540ebc3)  // st1w	{z3.s}, p2, [x30]
    KAI_ASM_INST(0xa540ab03)  // ld1w	{z3.s}, p2/z, [x24]
    KAI_ASM_INST(0x04a30042)  // add	z2.s, z2.s, z3.s
    KAI_ASM_INST(0xe540eb02)  // st1w	{z2.s}, p2, [x24]
    KAI_ASM_INST(0xa540aac2)  // ld1w	{z2.s}, p2/z, [x22]
    KAI_ASM_INST(0x04a20102)  // add	z2.s, z8.s, z2.s
    KAI_ASM_INST(0xe540eac2)  // st1w	{z2.s}, p2, [x22]
    KAI_ASM_INST(0xa540ab22)  // ld1w	{z2.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540eb21)  // st1w	{z1.s}, p2, [x25]
    KAI_ASM_INST(0xa540ab41)  // ld1w	{z1.s}, p2/z, [x26]
    KAI_ASM_INST(0x04a100e1)  // add	z1.s, z7.s, z1.s
    KAI_ASM_INST(0x05fc6a44)  // uzp1	z4.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96aa5)  // uzp1	z5.d, z21.d, z9.d
    KAI_ASM_INST(0x05fc6e52)  // uzp2	z18.d, z18.d, z28.d
    KAI_ASM_INST(0x05e96eb5)  // uzp2	z21.d, z21.d, z9.d
    KAI_ASM_INST(0xe540eb41)  // st1w	{z1.s}, p2, [x26]
    KAI_ASM_INST(0xa540ab61)  // ld1w	{z1.s}, p2/z, [x27]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540eb60)  // st1w	{z0.s}, p2, [x27]
    KAI_ASM_INST(0x17ffff0a)  // b	1790 <MlasGemmU8X8KernelUmmlaSveImpl+0x1790>
    KAI_ASM_INST(0xd2800208)  // mov	x8, #0x10
    KAI_ASM_INST(0xd2800022)  // mov	x2, #0x1
    KAI_ASM_INST(0xd2800040)  // mov	x0, #0x2
    KAI_ASM_INST(0x2518e121)  // ptrue	p1.b, vl16
    KAI_ASM_INST(0x17fff940)  // b	7c <MlasGemmU8X8KernelUmmlaSveImpl+0x7c>
    KAI_ASM_INST(0xe540ed64)  // st1w	{z4.s}, p3, [x11]
    KAI_ASM_INST(0xf10010ff)  // cmp	x7, #0x4
    KAI_ASM_INST(0x54000369)  // b.ls	1bf4 <MlasGemmU8X8KernelUmmlaSveImpl+0x1bf4>
    KAI_ASM_INST(0xd10011cc)  // sub	x12, x14, #0x4
    KAI_ASM_INST(0x25ac1fe4)  // whilelo	p4.s, xzr, x12
    KAI_ASM_INST(0x9100416c)  // add	x12, x11, #0x10
    KAI_ASM_INST(0xe540f181)  // st1w	{z1.s}, p4, [x12]
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff3849)  // b.ls	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xe540ed82)  // st1w	{z2.s}, p3, [x12]
    KAI_ASM_INST(0xe540f2a5)  // st1w	{z5.s}, p4, [x21]
    KAI_ASM_INST(0x17fff9be)  // b	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0x2d408486)  // ldp	s6, s1, [x4, #4]
    KAI_ASM_INST(0x2d418085)  // ldp	s5, s0, [x4, #12]
    KAI_ASM_INST(0x4e040421)  // dup	v1.4s, v1.s[0]
    KAI_ASM_INST(0xbd401483)  // ldr	s3, [x4, #20]
    KAI_ASM_INST(0x4d40c882)  // ld1r	{v2.4s}, [x4]
    KAI_ASM_INST(0x4e040400)  // dup	v0.4s, v0.s[0]
    KAI_ASM_INST(0x6e0c04c2)  // mov	v2.s[1], v6.s[0]
    KAI_ASM_INST(0x6e0c04a1)  // mov	v1.s[1], v5.s[0]
    KAI_ASM_INST(0x6e0c0460)  // mov	v0.s[1], v3.s[0]
    KAI_ASM_INST(0x6e1c04c2)  // mov	v2.s[3], v6.s[0]
    KAI_ASM_INST(0x6e1c04a1)  // mov	v1.s[3], v5.s[0]
    KAI_ASM_INST(0x6e1c0460)  // mov	v0.s[3], v3.s[0]
    KAI_ASM_INST(0x4ea48442)  // add	v2.4s, v2.4s, v4.4s
    KAI_ASM_INST(0x4ea48421)  // add	v1.4s, v1.4s, v4.4s
    KAI_ASM_INST(0x4ea48400)  // add	v0.4s, v0.4s, v4.4s
    KAI_ASM_INST(0x17fff976)  // b	1c8 <MlasGemmU8X8KernelUmmlaSveImpl+0x1c8>
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff3589)  // b.ls	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xe540ed82)  // st1w	{z2.s}, p3, [x12]
    KAI_ASM_INST(0x17fff9a9)  // b	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xeb19001f)  // cmp	x0, x25
    KAI_ASM_INST(0x54ff34e9)  // b.ls	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xa540ad80)  // ld1w	{z0.s}, p3/z, [x12]
    KAI_ASM_INST(0x04a20000)  // add	z0.s, z0.s, z2.s
    KAI_ASM_INST(0xe540ed80)  // st1w	{z0.s}, p3, [x12]
    KAI_ASM_INST(0x17fff9a2)  // b	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xd10042ac)  // sub	x12, x21, #0x10
    KAI_ASM_INST(0xa540ad81)  // ld1w	{z1.s}, p3/z, [x12]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540ed81)  // st1w	{z1.s}, p3, [x12]
    KAI_ASM_INST(0xa540b2a5)  // ld1w	{z5.s}, p4/z, [x21]
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0xe540f2a5)  // st1w	{z5.s}, p4, [x21]
    KAI_ASM_INST(0x17fff99a)  // b	2a8 <MlasGemmU8X8KernelUmmlaSveImpl+0x2a8>
    KAI_ASM_INST(0xf94073e0)  // ldr	x0, [sp, #224]
    KAI_ASM_INST(0xe5494800)  // st1w	{z0.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf10012df)  // cmp	x22, #0x4
    KAI_ASM_INST(0x540009e9)  // b.ls	1d8c <MlasGemmU8X8KernelUmmlaSveImpl+0x1d8c>
    KAI_ASM_INST(0xd10010e7)  // sub	x7, x7, #0x4
    KAI_ASM_INST(0x25a71fe3)  // whilelo	p3.s, xzr, x7
    KAI_ASM_INST(0xe5554c02)  // st1w	{z2.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf94067e0)  // ldr	x0, [sp, #200]
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0xe540ec0c)  // st1w	{z12.s}, p3, [x0]
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05f268ce)  // uzp1	z14.d, z6.d, z18.d
    KAI_ASM_INST(0x05f46ac5)  // uzp1	z5.d, z22.d, z20.d
    KAI_ASM_INST(0xe549480e)  // st1w	{z14.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xe5554c05)  // st1w	{z5.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf9406be0)  // ldr	x0, [sp, #208]
    KAI_ASM_INST(0x05f26cc6)  // uzp2	z6.d, z6.d, z18.d
    KAI_ASM_INST(0x05f46ed6)  // uzp2	z22.d, z22.d, z20.d
    KAI_ASM_INST(0xe540eb26)  // st1w	{z6.s}, p2, [x25]
    KAI_ASM_INST(0xe540ec16)  // st1w	{z22.s}, p3, [x0]
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05e86944)  // uzp1	z4.d, z10.d, z8.d
    KAI_ASM_INST(0x05f36aa3)  // uzp1	z3.d, z21.d, z19.d
    KAI_ASM_INST(0xe5494804)  // st1w	{z4.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xe5554c03)  // st1w	{z3.s}, p3, [x0, x21, lsl #2]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0x05e86d48)  // uzp2	z8.d, z10.d, z8.d
    KAI_ASM_INST(0xe540e808)  // st1w	{z8.s}, p2, [x0]
    KAI_ASM_INST(0xf9406fe0)  // ldr	x0, [sp, #216]
    KAI_ASM_INST(0x05e76921)  // uzp1	z1.d, z9.d, z7.d
    KAI_ASM_INST(0x05f769a0)  // uzp1	z0.d, z13.d, z23.d
    KAI_ASM_INST(0x05f36eb5)  // uzp2	z21.d, z21.d, z19.d
    KAI_ASM_INST(0x05e76d27)  // uzp2	z7.d, z9.d, z7.d
    KAI_ASM_INST(0x25834c64)  // mov	p4.b, p3.b
    KAI_ASM_INST(0x05f76dad)  // uzp2	z13.d, z13.d, z23.d
    KAI_ASM_INST(0xe540ec15)  // st1w	{z21.s}, p3, [x0]
    KAI_ASM_INST(0xf94063e0)  // ldr	x0, [sp, #192]
    KAI_ASM_INST(0xe540ebc1)  // st1w	{z1.s}, p2, [x30]
    KAI_ASM_INST(0xe540ec00)  // st1w	{z0.s}, p3, [x0]
    KAI_ASM_INST(0xe540eae7)  // st1w	{z7.s}, p2, [x23]
    KAI_ASM_INST(0x17fffcae)  // b	f9c <MlasGemmU8X8KernelUmmlaSveImpl+0xf9c>
    KAI_ASM_INST(0x4ea61cc1)  // mov	v1.16b, v6.16b
    KAI_ASM_INST(0x4f00042e)  // movi	v14.4s, #0x1
    KAI_ASM_INST(0x17fffbda)  // b	c58 <MlasGemmU8X8KernelUmmlaSveImpl+0xc58>
    KAI_ASM_INST(0x4ea31c60)  // mov	v0.16b, v3.16b
    KAI_ASM_INST(0x4f000427)  // movi	v7.4s, #0x1
    KAI_ASM_INST(0x17fffbb0)  // b	bbc <MlasGemmU8X8KernelUmmlaSveImpl+0xbbc>
    KAI_ASM_INST(0x4ea61cc1)  // mov	v1.16b, v6.16b
    KAI_ASM_INST(0x4f000428)  // movi	v8.4s, #0x1
    KAI_ASM_INST(0x17fffbc0)  // b	c08 <MlasGemmU8X8KernelUmmlaSveImpl+0xc08>
    KAI_ASM_INST(0xa94f17e2)  // ldp	x2, x5, [sp, #240]
    KAI_ASM_INST(0xa540ab40)  // ld1w	{z0.s}, p2/z, [x26]
    KAI_ASM_INST(0x05f268c5)  // uzp1	z5.d, z6.d, z18.d
    KAI_ASM_INST(0x04ab000b)  // add	z11.s, z0.s, z11.s
    KAI_ASM_INST(0x05f26cc2)  // uzp2	z2.d, z6.d, z18.d
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0x05e86944)  // uzp1	z4.d, z10.d, z8.d
    KAI_ASM_INST(0x05e86d41)  // uzp2	z1.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76923)  // uzp1	z3.d, z9.d, z7.d
    KAI_ASM_INST(0x05e76d20)  // uzp2	z0.d, z9.d, z7.d
    KAI_ASM_INST(0x8b000042)  // add	x2, x2, x0
    KAI_ASM_INST(0xa540a846)  // ld1w	{z6.s}, p2/z, [x2]
    KAI_ASM_INST(0x04a600a5)  // add	z5.s, z5.s, z6.s
    KAI_ASM_INST(0xe540e845)  // st1w	{z5.s}, p2, [x2]
    KAI_ASM_INST(0xa540ab25)  // ld1w	{z5.s}, p2/z, [x25]
    KAI_ASM_INST(0x04a50042)  // add	z2.s, z2.s, z5.s
    KAI_ASM_INST(0x8b0000a0)  // add	x0, x5, x0
    KAI_ASM_INST(0xe540eb22)  // st1w	{z2.s}, p2, [x25]
    KAI_ASM_INST(0xa540a802)  // ld1w	{z2.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a20082)  // add	z2.s, z4.s, z2.s
    KAI_ASM_INST(0xe540e802)  // st1w	{z2.s}, p2, [x0]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0xa540a802)  // ld1w	{z2.s}, p2/z, [x0]
    KAI_ASM_INST(0x04a20021)  // add	z1.s, z1.s, z2.s
    KAI_ASM_INST(0xe540e801)  // st1w	{z1.s}, p2, [x0]
    KAI_ASM_INST(0xa540abc1)  // ld1w	{z1.s}, p2/z, [x30]
    KAI_ASM_INST(0x04a10061)  // add	z1.s, z3.s, z1.s
    KAI_ASM_INST(0xe540ebc1)  // st1w	{z1.s}, p2, [x30]
    KAI_ASM_INST(0xa540aae1)  // ld1w	{z1.s}, p2/z, [x23]
    KAI_ASM_INST(0x04a10000)  // add	z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe540eae0)  // st1w	{z0.s}, p2, [x23]
    KAI_ASM_INST(0x17fffc87)  // b	fa4 <MlasGemmU8X8KernelUmmlaSveImpl+0xfa4>
    KAI_ASM_INST(0xf9407be0)  // ldr	x0, [sp, #240]
    KAI_ASM_INST(0x05f268c2)  // uzp1	z2.d, z6.d, z18.d
    KAI_ASM_INST(0xe540eb4b)  // st1w	{z11.s}, p2, [x26]
    KAI_ASM_INST(0xe5494802)  // st1w	{z2.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf9407fe0)  // ldr	x0, [sp, #248]
    KAI_ASM_INST(0x05e86941)  // uzp1	z1.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76920)  // uzp1	z0.d, z9.d, z7.d
    KAI_ASM_INST(0x05f26cc6)  // uzp2	z6.d, z6.d, z18.d
    KAI_ASM_INST(0x05e86d48)  // uzp2	z8.d, z10.d, z8.d
    KAI_ASM_INST(0x05e76d27)  // uzp2	z7.d, z9.d, z7.d
    KAI_ASM_INST(0xe540eb26)  // st1w	{z6.s}, p2, [x25]
    KAI_ASM_INST(0xe5494801)  // st1w	{z1.s}, p2, [x0, x9, lsl #2]
    KAI_ASM_INST(0xf9405be0)  // ldr	x0, [sp, #176]
    KAI_ASM_INST(0xe540e808)  // st1w	{z8.s}, p2, [x0]
    KAI_ASM_INST(0xe540ebc0)  // st1w	{z0.s}, p2, [x30]
    KAI_ASM_INST(0xe540eae7)  // st1w	{z7.s}, p2, [x23]
    KAI_ASM_INST(0x17fffc76)  // b	fa4 <MlasGemmU8X8KernelUmmlaSveImpl+0xfa4>
    KAI_ASM_INST(0xd2800408)  // mov	x8, #0x20
    KAI_ASM_INST(0xd2800042)  // mov	x2, #0x2
    KAI_ASM_INST(0xd2800080)  // mov	x0, #0x4
    KAI_ASM_INST(0x2518e121)  // ptrue	p1.b, vl16
    KAI_ASM_INST(0x17fff8a7)  // b	7c <MlasGemmU8X8KernelUmmlaSveImpl+0x7c>
    KAI_ASM_INST(0x04a00240)  // add	z0.s, z18.s, z0.s
    KAI_ASM_INST(0xe540e800)  // st1w	{z0.s}, p2, [x0]
    KAI_ASM_INST(0x17fffe73)  // b	17b8 <MlasGemmU8X8KernelUmmlaSveImpl+0x17b8>
    KAI_ASM_FUNCTION_END(MlasGemmU8X8KernelUmmlaSveImpl)

    KAI_ASM_END
